"""
Writer Agent Module

This module is responsible for generating notebook content based on the plan created by the Planner LLM.
It uses langgraph to create a workflow that includes the Writer LLM and the Critic LLM.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import os
import re
import shutil

import openai
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.prompts.chat import ChatPromptTemplate

# Use absolute imports for proper package imports when running from root
from src.models import (
    NotebookPlanModel,
    Section,
    MultipleSearchQuestions,
    NotebookSectionContent,
    NotebookCell,
    CriticEvaluation,
    SearchDecision,
)
from src.prompts.writer_prompts import (
    WRITER_SYSTEM_PROMPT,
    WRITER_CONTENT_PROMPT,
    WRITER_REVISION_PROMPT,
    SEARCH_DECISION_INITIAL_PROMPT,
    SEARCH_DECISION_POST_CRITIQUE_PROMPT,
    WRITER_FINAL_REVISION_PROMPT,
)
from src.prompts.critic_prompts import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_EVALUATION_PROMPT,
    FINAL_CRITIQUE_PROMPT,
)
from src.format.prompt_utils import (
    format_subsections_details,
    format_additional_requirements,
    format_previous_content,
    format_cells_for_evaluation,
)
from src.format.markdown_utils import (
    notebook_content_to_markdown,
    notebook_section_to_markdown,
    format_notebook_for_critique,
    markdown_to_notebook_content,
    notebook_plan_to_markdown,
    json_string_to_markdown,
)
from src.format.notebook_utils import (
    save_notebook_versions,
    writer_output_to_files,
    writer_output_to_notebook,
    notebook_to_writer_output,
    notebook_dict_to_writer_output,
)
from src.searcher import (
    search_with_tavily,
    search_with_openai,
    format_tavily_search_results,
    format_openai_search_results,
)

# Get a logger for this module
logger = logging.getLogger(__name__)

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    # If dotenv is not installed, just print a warning
    print("Warning: python-dotenv not installed. Using environment variables directly.")


class WriterAgent:
    """
    Agent for generating notebook content based on the plan created by the Planner LLM.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3,
        search_enabled: bool = True,
        final_critique_enabled: bool = True,
        search_context_size: str = "low",
    ):
        """
        Initialize the Writer Agent.

        Args:
            model (str, optional): The model to use for generating content. Defaults to "gpt-4o".
            max_retries (int, optional): Maximum number of retries for content generation. Defaults to 3.
            search_enabled (bool, optional): Whether to enable search functionality. Defaults to True.
            final_critique_enabled (bool, optional): Whether to perform a final critique of the complete notebook. Defaults to True.
            search_context_size (str, optional): Size of search context to use ("low", "medium", or "high"). Defaults to "low".
        """
        logger.info(f"Initializing WriterAgent with model: {model}")

        # Set the model and configuration
        self.model = model
        self.max_retries = max_retries
        self.search_enabled = search_enabled
        self.final_critique_enabled = final_critique_enabled
        self.search_context_size = search_context_size

        # Create the OpenAI client
        try:
            self.client = OpenAI()
            logger.debug("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        # Create the critic LLM
        try:
            self.critic_llm = ChatOpenAI(model=model)
            logger.debug("Critic LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Critic LLM: {e}")
            raise

        # Create the workflow
        self.workflow = self._create_workflow()
        logger.debug("Writer workflow created successfully")

    def _create_workflow(self) -> StateGraph:
        """
        Create the workflow for generating notebook content.

        Returns:
            StateGraph: The workflow graph.
        """
        logger.debug("Creating writer workflow")

        # Define the state schema
        class State(dict):
            """State for the writer workflow."""

            notebook_plan: NotebookPlanModel
            section_index: int
            current_section: Section
            additional_requirements: Dict[str, Any]
            search_results: Optional[str]
            generated_content: Optional[NotebookSectionContent]
            evaluation: Optional[CriticEvaluation]
            final_content: Optional[NotebookSectionContent]
            previous_content: Dict[str, str]
            max_retries: int
            current_retries: int
            has_critic_feedback: bool
            search_decision_results: Optional[SearchDecision]

        # Create the workflow graph to manage the content generation process
        workflow = StateGraph(State)

        # Add nodes to the workflow
        # 1. Initial decision node to determine if search is needed
        workflow.add_node("search_decision", self._search_decision_node)
        # 2. Search node to gather information from external sources
        workflow.add_node("search", self._search_node)
        # 3. Generate node to create initial content based on requirements and search results
        workflow.add_node("generate", self._generate_node)
        # 4. Critic node to evaluate the quality of generated content
        workflow.add_node("critic", self._critic_node)
        # 5. Revise node to improve content based on critic feedback
        workflow.add_node("revise", self._revise_node)
        # 6. Second search decision node that runs after critic feedback
        workflow.add_node("search_decision_after_critic", self._search_decision_node)
        # 7. Second search node that runs after critic feedback if needed
        workflow.add_node("search_after_critic", self._search_node)

        # Define the workflow paths with edges
        # Initial path: Decide whether to search first or go straight to content generation
        workflow.add_conditional_edges(
            "search_decision",
            self._needs_search,  # Function that determines if search is needed
            {
                True: "search",  # If search needed, go to search node
                False: "generate",  # If no search needed, go directly to generate node
            },
        )
        # After search, always proceed to content generation
        workflow.add_edge("search", "generate")

        # After generation, always evaluate the content with the critic
        workflow.add_edge("generate", "critic")

        # After critic evaluation, either finish (if content is good) or continue revision process
        workflow.add_conditional_edges(
            "critic",
            self._should_revise,  # Function that determines if revision is needed
            {
                True: "search_decision_after_critic",  # If revision needed, decide if more search is required
                False: END,  # If no revision needed, end the workflow
            },
        )

        # For revision path: First decide if additional search is needed based on critic feedback
        workflow.add_conditional_edges(
            "search_decision_after_critic",
            self._needs_search,  # Reusing the same function to check if search is needed
            {
                True: "search_after_critic",  # If more information needed, do another search
                False: "revise",  # If no additional search needed, proceed directly to revision
            },
        )

        # After post-critic search, proceed to revision
        workflow.add_edge("search_after_critic", "revise")

        # After revision, go back to critic for re-evaluation (creating a feedback loop)
        workflow.add_edge("revise", "critic")

        # Set the workflow's starting point
        workflow.set_entry_point("search_decision")

        # Finalize the workflow configuration
        logger.debug(f"Workflow created with nodes: {workflow.nodes}")
        return workflow

    def _call_llm_with_parser(
        self,
        system_prompt: str,
        user_prompt: str,
        output_parser_cls: Any,
        temperature: float = 0.5,
    ) -> Any:
        """
        Generic method to call an LLM with structured output parsing.

        Args:
            system_prompt (str): The system prompt to use
            user_prompt (str): The user prompt to use
            output_parser_cls (Any): The Pydantic class to parse the output into
            temperature (float, optional): Sampling temperature. Defaults to 0.5.

        Returns:
            Any: The parsed output object or None if parsing failed
        """
        try:
            messages = [
                ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                ChatCompletionUserMessageParam(content=user_prompt, role="user"),
            ]

            logger.debug(f"Calling OpenAI API with {output_parser_cls.__name__} parser")
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=output_parser_cls,
                temperature=temperature,
            )

            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(
                f"Error calling LLM with {output_parser_cls.__name__} parser: {e}"
            )
            return None

    def _search_decision_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide whether additional searches are needed before generating content.

        This function analyzes the current section requirements, any existing search results,
        and (if available) critic feedback to determine if more searches are needed.
        If so, it generates specific search queries to fill information gaps.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state with search decision information.
        """
        logger.info(
            f"Executing search decision node for section: {state['current_section'].title}"
        )

        # Get the current section and notebook plan
        section = state["current_section"]
        notebook_plan = state["notebook_plan"]

        # Format existing search results (if any)
        existing_search_results = state.get(
            "search_results", "No existing search results"
        )

        # Get current month and year for time-relevant search queries
        current_month_year = datetime.now().strftime("%B %Y")
        logger.info(f"Current month/year for search queries: {current_month_year}")

        # Determine which prompt to use based on presence of critic feedback
        has_critic_feedback = state.get("has_critic_feedback", False)

        # Prepare the user prompt based on whether we have critic feedback
        if (
            has_critic_feedback
            and "evaluation" in state
            and state["evaluation"] is not None
        ):
            logger.info("Using post-critique prompt for search decision")
            # Format generated content for the prompt
            generated_content = ""
            if "generated_content" in state and state["generated_content"] is not None:
                generated_content = self._format_cells_for_evaluation(
                    state["generated_content"].cells
                )

            # Format the prompt with critic feedback
            user_prompt = SEARCH_DECISION_POST_CRITIQUE_PROMPT.format(
                current_month_year=current_month_year,
                notebook_title=notebook_plan.title,
                notebook_description=notebook_plan.description,
                notebook_purpose=notebook_plan.purpose,
                notebook_target_audience=notebook_plan.target_audience,
                section_title=section.title,
                section_description=section.description,
                subsections_details=format_subsections_details(section.subsections),
                additional_requirements=format_additional_requirements(
                    state.get("additional_requirements", {})
                ),
                existing_search_results=existing_search_results,
                generated_content=generated_content,
                critic_evaluation=state["evaluation"].rationale,
            )
        else:
            logger.info("Using initial prompt for search decision")
            # Format the prompt for initial search decision
            user_prompt = SEARCH_DECISION_INITIAL_PROMPT.format(
                current_month_year=current_month_year,
                notebook_title=notebook_plan.title,
                notebook_description=notebook_plan.description,
                notebook_purpose=notebook_plan.purpose,
                notebook_target_audience=notebook_plan.target_audience,
                section_title=section.title,
                section_description=section.description,
                subsections_details=format_subsections_details(section.subsections),
                additional_requirements=format_additional_requirements(
                    state.get("additional_requirements", {})
                ),
                existing_search_results=existing_search_results,
                previous_content=format_previous_content(
                    state.get("previous_content", {})
                ),
            )

        # Create the system prompt for search decision
        system_prompt = "You are an AI assistant that analyzes information needs and determines when additional searches are required to generate high-quality educational content about OpenAI APIs."

        # Get the search decision using the generic LLM caller
        search_decision_results = self._call_llm_with_parser(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_parser_cls=SearchDecision,
            temperature=0,
        )

        # Create a default decision as fallback if needed
        if not search_decision_results:
            search_decision_results = SearchDecision(
                needs_additional_search=True,
                reasoning="Error generating search decision, defaulting to performing a search",
                search_queries=None,
            )
        else:
            # Log the decision results
            logger.info(
                f"Search decision: needs_additional_search={search_decision_results.needs_additional_search}"
            )
            if (
                search_decision_results.needs_additional_search
                and search_decision_results.search_queries
            ):
                logger.info(
                    f"Generated {len(search_decision_results.search_queries)} search queries"
                )
                for i, query in enumerate(search_decision_results.search_queries):
                    logger.info(f"Query {i+1}: {query.search_query}")
                    logger.debug(f"Justification: {query.justification}")

        # Store the decision in the state
        state["search_decision_results"] = search_decision_results

        # If coming from critic, update the has_critic_feedback flag to true
        # This ensures subsequent runs will use the post-critique prompt
        if "evaluation" in state and state["evaluation"] is not None:
            state["has_critic_feedback"] = True

        logger.debug("Search decision node completed successfully")
        return state

    def _needs_search(self, state: Dict[str, Any]) -> bool:
        """
        Determine whether to perform a search based on the search decision.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            bool: True if a search should be performed, False otherwise.
        """
        # First check if search is globally disabled
        if not self.search_enabled:
            logger.info("Search is disabled by configuration, skipping search")
            return False

        # Check if we have a search decision
        if (
            "search_decision_results" not in state
            or state["search_decision_results"] is None
        ):
            logger.debug("No search decision found, defaulting to performing search")
            return True

        # Get the decision
        decision = state["search_decision_results"]

        # Log the decision
        logger.info(f"Search decision: {decision.needs_additional_search}")
        logger.debug(f"Search reasoning: {decision.reasoning}")

        # Return the decision
        return decision.needs_additional_search

    def _search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for information related to the current section.

        This function uses search queries from the search decision if available,
        or generates new questions and searches for information.
        It appends new search results to existing ones.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
        logger.info(
            f"Executing search node for section: {state['current_section'].title}"
        )

        # Get the current section
        section = state["current_section"]

        all_search_results = []
        search_queries = []

        # Check if we have search queries from the search decision
        if (
            "search_decision_results" in state
            and state["search_decision_results"] is not None
            and state["search_decision_results"].search_queries is not None
        ):
            # Use the search queries from the decision
            logger.info("Using search queries from search decision")
            search_queries = state["search_decision_results"].search_queries
        else:
            # Generate new questions if no search queries provided
            logger.info("No search queries from search decision, generating questions")
            search_queries = self._generate_search_questions(section)

        # Execute searches for all queries
        for i, query in enumerate(search_queries):
            query_text = getattr(query, "search_query", None) or getattr(
                query, "question", None
            )
            justification = getattr(query, "justification", "No justification provided")

            if not query_text:
                logger.warning(f"Query {i+1} has no searchable text, skipping")
                continue

            logger.info(f"Searching for information with query {i+1}: {query_text}")

            try:
                # Search for information using OpenAI
                search_results = search_with_openai(
                    search_query=query_text,
                    model=self.model,
                    search_context_size=self.search_context_size,
                    client=self.client,  # Pass the existing client
                )
                formatted_results = format_openai_search_results(search_results)
                logger.debug(f"OpenAI search completed for query {i+1}")

                # Add the query to the results
                query_with_results = {
                    "question": query_text,
                    "justification": justification,
                    "search_results": formatted_results,
                }

                all_search_results.append(query_with_results)
            except Exception as e:
                logger.error(f"Failed to search for information for query {i+1}: {e}")
                query_with_results = {
                    "question": query_text,
                    "justification": justification,
                    "search_results": f"Error searching for information: {str(e)}",
                }
                all_search_results.append(query_with_results)

        # Format new search results
        new_results = self._format_combined_search_results(all_search_results)

        # Get existing search results, if any
        existing_results = state.get("search_results", "")

        # Append new results to existing results if there are existing results
        if (
            existing_results
            and existing_results != "No search results found"
            and existing_results != "No existing search results"
        ):
            # Strip the "# Combined Search Results" header from the new results to avoid duplication
            if new_results.startswith("# Search Results"):
                new_results_content = (
                    new_results.split("\n\n", 1)[1] if "\n\n" in new_results else ""
                )
            else:
                new_results_content = new_results

            # Add a separator between existing and new results
            combined_results = (
                existing_results.rstrip()
                + "\n\n## Additional Search Results\n\n"
                + new_results_content
            )
        else:
            combined_results = new_results

        # Update the state with combined results
        state["search_results"] = combined_results
        logger.debug(
            f"Search node completed successfully with new search results appended"
        )
        return state

    def _format_combined_search_results(self, all_results: List[Dict[str, str]]) -> str:
        """
        Format combined search results from multiple questions.

        Args:
            all_results (List[Dict[str, str]]): List of dictionaries containing questions and their search results.

        Returns:
            str: Formatted combined search results.
        """
        if not all_results:
            return "No search results found"

        formatted = "# Search Results\n\n"

        # Format each search result
        for i, result in enumerate(all_results, 1):
            # Format question with number
            formatted += f"## Question {i}: {result['question']}\n"
            formatted += f"*Justification: {result['justification']}*\n\n"
            formatted += f"{result['search_results']}\n\n"
            formatted += "---\n\n"

        return formatted

    def _generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content for the current section.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
        logger.info(f"Generating content for section: {state['current_section'].title}")

        # Get the current section and notebook plan
        section = state["current_section"]
        notebook_plan = state["notebook_plan"]
        previous_content = state.get("previous_content", {})

        # Format the prompt
        user_prompt = WRITER_CONTENT_PROMPT.format(
            notebook_title=notebook_plan.title,
            notebook_description=notebook_plan.description,
            notebook_purpose=notebook_plan.purpose,
            notebook_target_audience=notebook_plan.target_audience,
            section_title=section.title,
            section_description=section.description,
            subsections_details=format_subsections_details(section.subsections),
            additional_requirements=format_additional_requirements(
                state.get("additional_requirements", {})
            ),
            search_results=state.get("search_results", "No search results available"),
            previous_content=format_previous_content(previous_content),
        )
        logger.debug(
            f"Formatted content generation prompt with {len(user_prompt)} characters"
        )

        # Generate the content using the generic LLM call method
        section_content = self._call_llm_with_parser(
            system_prompt=WRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_parser_cls=NotebookSectionContent,
            temperature=0.5,
        )

        # Create an empty section content as fallback if needed
        if not section_content or not hasattr(section_content, "cells"):
            logger.error(
                f"Failed to generate valid content for section: {section.title}"
            )
            section_content = NotebookSectionContent(
                section_title=section.title,
                cells=[
                    NotebookCell(
                        cell_type="markdown",
                        content=f"Error generating content for section: {section.title}",
                    )
                ],
            )
        else:
            logger.info(f"Generated content with {len(section_content.cells)} cells")
            logger.debug(
                f"Content types: {[cell.cell_type for cell in section_content.cells]}"
            )

        # Update the state
        state["generated_content"] = section_content
        logger.debug("Generate node completed successfully")
        return state

    def _critic_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the generated content.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
        logger.info(f"Evaluating content for section: {state['current_section'].title}")

        # Get the current section, notebook plan, and generated content
        section = state["current_section"]
        notebook_plan = state["notebook_plan"]
        generated_content = state["generated_content"]

        # Format the generated content for evaluation
        formatted_content = self._format_cells_for_evaluation(generated_content.cells)
        logger.debug(
            f"Formatted content for evaluation with {len(formatted_content)} characters"
        )

        # Format the prompt
        user_prompt = CRITIC_EVALUATION_PROMPT.format(
            notebook_title=notebook_plan.title,
            notebook_description=notebook_plan.description,
            notebook_purpose=notebook_plan.purpose,
            notebook_target_audience=notebook_plan.target_audience,
            section_title=section.title,
            section_description=section.description,
            subsections_details=format_subsections_details(section.subsections),
            additional_requirements=format_additional_requirements(
                state.get("additional_requirements", {})
            ),
            generated_content=formatted_content,
        )

        # Evaluate the content using the generic LLM call method
        evaluation = self._call_llm_with_parser(
            system_prompt=CRITIC_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_parser_cls=CriticEvaluation,
            temperature=0,
        )

        # Create a default evaluation as fallback if needed
        if not evaluation:
            logger.error(f"Failed to get valid evaluation for section: {section.title}")
            evaluation = CriticEvaluation(
                rationale="Error evaluating content", passed=False
            )
        else:
            logger.info(f"Evaluation: {evaluation}")

        # Update the state with evaluation
        state["evaluation"] = evaluation

        # If the content passes evaluation, set it as the final content
        if evaluation and evaluation.passed:
            logger.info("Content passed evaluation")
            state["final_content"] = generated_content
        # Check if this attempt would reach the maximum retries (after incrementing)
        elif state.get("current_retries", 0) + 1 >= state["max_retries"]:
            logger.warning(
                f"Maximum number of retries reached. "
                f"Using the last generated content despite failing evaluation."
            )
            state["current_retries"] = state["max_retries"]
            state["final_content"] = generated_content
        else:
            # Increment the retry counter
            state["current_retries"] = state.get("current_retries", 0) + 1
            logger.info(
                f"Content failed evaluation. Retry {state['current_retries']} of {state['max_retries']}"
            )

        logger.debug("Critic node completed successfully")
        return state

    def _revise_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revise the generated content based on the evaluation.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
        logger.info(f"Revising content for section: {state['current_section'].title}")

        # Get the current section, notebook plan, generated content, and evaluation
        section = state["current_section"]
        notebook_plan = state["notebook_plan"]
        generated_content = state["generated_content"]
        evaluation = state["evaluation"]

        # Format the generated content for revision
        formatted_content = self._format_cells_for_evaluation(generated_content.cells)
        logger.debug(
            f"Formatted content for revision with {len(formatted_content)} characters"
        )

        # Create the revision prompt
        user_prompt = WRITER_REVISION_PROMPT.format(
            notebook_title=notebook_plan.title,
            notebook_description=notebook_plan.description,
            notebook_purpose=notebook_plan.purpose,
            notebook_target_audience=notebook_plan.target_audience,
            section_title=section.title,
            section_description=section.description,
            original_content=formatted_content,
            evaluation_feedback=evaluation.rationale,
            search_results=state.get("search_results", "No search results available"),
        )

        # Generate the revised content using the generic LLM call method
        section_content = self._call_llm_with_parser(
            system_prompt=WRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_parser_cls=NotebookSectionContent,
            temperature=0.3,
        )

        # Use original content as fallback if revision fails
        if not section_content or not hasattr(section_content, "cells"):
            logger.error(f"Failed to revise content for section: {section.title}")
            logger.warning("Using original content as fallback due to revision error")
            section_content = generated_content
        else:
            logger.info(f"Revised content with {len(section_content.cells)} cells")
            logger.debug(
                f"Revised content types: {[cell.cell_type for cell in section_content.cells]}"
            )

        # Update the state
        state["generated_content"] = section_content
        logger.debug("Revise node completed successfully")
        return state

    def _should_revise(self, state: Dict[str, Any]) -> bool:
        """
        Determine whether to revise the content based on the evaluation.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            bool: True if the content should be revised, False otherwise.
        """
        # Check if the content passed evaluation
        if state["evaluation"] and state["evaluation"].passed:
            logger.debug("Content passed evaluation, no revision needed")
            return False

        # Check if we've reached the maximum number of retries
        if state.get("current_retries", 0) >= state["max_retries"]:
            logger.warning(
                f"Maximum number of retries reached ({state['max_retries']}). "
                f"Using the last generated content despite failing evaluation."
            )
            return False

        # If we're here, we should revise the content
        return True

    def _format_cells_for_evaluation(self, cells: List[NotebookCell]) -> str:
        """
        Format cells for evaluation.

        Args:
            cells (List[NotebookCell]): The cells to format.

        Returns:
            str: The formatted cells.
        """
        # Use the imported function from format.prompt_utils
        return format_cells_for_evaluation(cells)

    def generate_section_content(
        self,
        notebook_plan: NotebookPlanModel,
        section_index: int,
        additional_requirements: Optional[Union[List[str], Dict[str, Any]]] = None,
        previous_content: Optional[Dict[str, str]] = None,
    ) -> NotebookSectionContent:
        """
        Generate content for a specific section of the notebook.

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            section_index (int): The index of the section to generate content for.
            additional_requirements (Optional[Union[List[str], Dict[str, Any]]], optional): Additional requirements for the content. Defaults to None.
            previous_content (Optional[Dict[str, str]], optional): Content from previous sections. Defaults to None.

        Returns:
            NotebookSectionContent: The generated content.
        """
        logger.info(
            f"Generating content for section {section_index + 1}: {notebook_plan.sections[section_index].title}"
        )

        # Get the section
        section = notebook_plan.sections[section_index]

        # Create the initial state
        state = {
            "notebook_plan": notebook_plan,
            "section_index": section_index,
            "current_section": section,
            "additional_requirements": additional_requirements or {},
            "search_results": None,
            "generated_content": None,
            "evaluation": None,
            "final_content": None,
            "previous_content": previous_content or {},
            "max_retries": self.max_retries,
            "current_retries": 0,
            "has_critic_feedback": False,
            "search_decision_results": None,
        }

        # Run the workflow
        try:
            logger.debug("Running writer workflow")
            compiled_workflow = self.workflow.compile(debug=False)
            result = compiled_workflow.invoke(state)
            logger.info(
                f"Workflow completed successfully for section {section_index + 1}"
            )
        except Exception as e:
            logger.error(f"Error running workflow for section {section_index + 1}: {e}")
            # Create an empty section content as fallback
            return NotebookSectionContent(
                section_title=section.title,
                cells=[
                    NotebookCell(
                        cell_type="markdown",
                        content=f"Error generating content for section: {section.title}",
                    )
                ],
            )

        # Return the final content
        if "final_content" not in result or result["final_content"] is None:
            logger.error(
                f"Final content is missing or None for section {section_index + 1}. Using fallback content."
            )
            # Try to use the generated_content if available
            if (
                "generated_content" in result
                and result["generated_content"] is not None
            ):
                logger.info("Using generated_content as fallback")
                return result["generated_content"]
            else:
                # Create an empty section content as fallback
                return NotebookSectionContent(
                    section_title=section.title,
                    cells=[
                        NotebookCell(
                            cell_type="markdown",
                            content=f"Error generating content for section: {section.title}. The content generation process did not produce a valid result.",
                        )
                    ],
                )

        final_content = result["final_content"]

        logger.info(
            f"Generated content for section {section_index + 1} with "
            f"{len(final_content.cells)} cells"
        )
        return final_content

    def _final_revision(
        self,
        notebook_plan: NotebookPlanModel,
        section_contents: List[NotebookSectionContent],
        final_critique: str,
    ) -> List[NotebookSectionContent]:
        """
        Apply final revisions to the notebook based on critique using an .ipynb format string.

        Instead of saving to temporary files, this method:
        1. Converts the content to an ipynb format string directly using writer_output_to_notebook
        2. Passes that string to the LLM with WRITER_FINAL_REVISION_PROMPT
        3. Directly parses the LLM response back to NotebookSectionContent objects
        4. Also saves a markdown version of the response regardless of parsing success
        5. If parsing fails, saves the raw LLM response and returns original content

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            section_contents (List[NotebookSectionContent]): The generated content for all sections.
            final_critique (str): The critique of the notebook from the final review.

        Returns:
            List[NotebookSectionContent]: The revised notebook content or original content if revision fails.
        """
        import json
        import os
        from src.format.notebook_utils import (
            writer_output_to_notebook,
            notebook_dict_to_writer_output,
        )
        from src.format.markdown_utils import json_string_to_markdown

        logger.info("Performing final revision of the notebook based on critique")

        # Create a temp directory to store processing artifacts
        temp_dir = os.path.join(os.getcwd(), "output", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        logger.debug(f"Created temp directory at {temp_dir}")

        try:
            # Step 1: Convert section_contents to notebook dictionary
            logger.debug(
                f"Converting {len(section_contents)} sections to notebook dictionary"
            )
            notebook = writer_output_to_notebook(
                writer_output=section_contents,
                notebook_title=notebook_plan.title,
                return_notebook=True,
            )

            # Step 2: Convert notebook to JSON string
            notebook_content = json.dumps(notebook, indent=2)
            logger.debug(
                f"Converted notebook to JSON string ({len(notebook_content)} characters)"
            )

            # Step 3: Format the prompt with notebook content and critique
            user_prompt = WRITER_FINAL_REVISION_PROMPT.format(
                notebook_title=notebook_plan.title,
                notebook_description=notebook_plan.description,
                notebook_purpose=notebook_plan.purpose,
                notebook_target_audience=notebook_plan.target_audience,
                notebook_content=notebook_content,
                final_critique=final_critique,
            )
            logger.debug(
                f"Formatted prompt for final revision ({len(user_prompt)} characters)"
            )

            # Step 4: Create message and call the OpenAI API
            messages = [
                ChatCompletionUserMessageParam(content=user_prompt, role="user"),
            ]

            logger.debug("Calling OpenAI API with model o1-2024-12-17")
            response = self.client.chat.completions.create(
                model="o1-2024-12-17",
                messages=messages,
                response_format={"type": "json_object"},
            )
            logger.debug("Received response from OpenAI API")

            # Step 5: Extract and validate the revised JSON content
            revised_content = response.choices[0].message.content

            # Check for empty response
            if revised_content is None or not revised_content.strip():
                logger.warning(
                    "Received empty response from API, using original notebook"
                )
                return section_contents

            # Step 6: Save the raw response for debugging and recovery
            raw_response_path = os.path.join(temp_dir, "final_revision_response.json")
            with open(raw_response_path, "w") as f:
                f.write(revised_content)
            logger.info(f"Saved raw LLM response to {raw_response_path}")

            # Step 7: Clean up and parse the JSON
            logger.debug("Starting JSON cleanup and parsing process")
            cleaned_content = revised_content.strip()

            # Remove any markdown code block indicators if present
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json", 1)[1]
            if cleaned_content.startswith("```"):
                cleaned_content = cleaned_content.split("```", 1)[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()

            # Attempt to parse JSON
            revised_notebook = json.loads(cleaned_content)
            logger.debug("Successfully parsed JSON content")

            # Step 8: Convert the parsed JSON to NotebookSectionContent objects
            revised_sections = notebook_dict_to_writer_output(revised_notebook)
            logger.info(
                f"Successfully revised notebook with {len(revised_sections)} sections"
            )

            return revised_sections

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")

            # Save the problematic content for debugging
            error_path = os.path.join(temp_dir, "json_parse_error.txt")
            with open(error_path, "w") as f:
                f.write(f"Error: {str(e)}\n\n")
                if "cleaned_content" in locals() and cleaned_content is not None:
                    f.write("Content that failed to parse:\n\n")
                    f.write(cleaned_content)
                elif "revised_content" in locals() and revised_content is not None:
                    f.write("Raw content from API:\n\n")
                    f.write(revised_content)
            logger.info(f"Saved error details to {error_path}")

            # Return the original content
            logger.warning(
                "Using original notebook content due to JSON parsing failure"
            )
            return section_contents

        except Exception as e:
            logger.error(f"Unexpected error during final revision: {e}")

            # Save error information
            error_path = os.path.join(temp_dir, "unexpected_error.txt")
            with open(error_path, "w") as f:
                f.write(f"Error: {str(e)}\n")
            logger.info(f"Saved error details to {error_path}")

            # Return the original content in case of any errors
            logger.warning("Using original notebook content due to unexpected error")
            return section_contents

    def generate_content(
        self,
        notebook_plan: NotebookPlanModel,
        additional_requirements: Optional[Union[List[str], Dict[str, Any]]] = None,
    ) -> Union[
        List[NotebookSectionContent],
        Tuple[List[NotebookSectionContent], str, List[NotebookSectionContent]],
    ]:
        """
        Generate content for all sections of the notebook.

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            additional_requirements (Optional[Union[List[str], Dict[str, Any]]], optional): Additional requirements for the content. Defaults to None.

        Returns:
            Union[
                List[NotebookSectionContent],
                Tuple[List[NotebookSectionContent], str, List[NotebookSectionContent]]
            ]:
                If final_critique_enabled is False: The generated content for all sections.
                If final_critique_enabled is True: A tuple of (original content, final critique, revised content).
        """
        logger.info(f"Generating content for notebook: {notebook_plan.title}")
        logger.info(f"Notebook has {len(notebook_plan.sections)} sections")

        # Initialize the list of section contents
        section_contents = []

        # Initialize the previous content dictionary
        previous_content = {}

        # Generate content for each section
        for i, section in enumerate(notebook_plan.sections):
            logger.info(
                f"Generating content for section {i + 1}/{len(notebook_plan.sections)}: {section.title}"
            )

            # Generate content for the section
            section_content = self.generate_section_content(
                notebook_plan=notebook_plan,
                section_index=i,
                additional_requirements=additional_requirements,
                previous_content=previous_content,
            )

            # Add the section content to the list
            section_contents.append(section_content)

            # Add the full section content to the previous content dictionary
            previous_content[section.title] = section_content

            logger.info(f"Completed section {i + 1}/{len(notebook_plan.sections)}")

        logger.info(
            f"Content generation completed for all {len(notebook_plan.sections)} sections"
        )

        # If final critique is enabled, generate it and apply final revisions
        if self.final_critique_enabled:
            # Generate the final critique
            final_critique = self._final_critique(notebook_plan, section_contents)
            logger.info("Final critique completed")

            # Print the critique to the console for review
            print("\n==== FINAL NOTEBOOK CRITIQUE ====\n")
            print(final_critique)
            print("\n================================\n")

            # Save the original content before applying revisions
            original_content = section_contents.copy()

            # Apply final revisions based on the critique
            revised_content = self._final_revision(
                notebook_plan, section_contents, final_critique
            )
            logger.info("Final revision completed")

            # Return the original content, critique, and revised content as a tuple
            return original_content, final_critique, revised_content

        return section_contents

    def _final_critique(
        self,
        notebook_plan: NotebookPlanModel,
        section_contents: List[NotebookSectionContent],
    ) -> str:
        """
        Perform a final critique of the complete notebook.

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            section_contents (List[NotebookSectionContent]): The generated content for all sections.

        Returns:
            str: The final critique of the notebook.
        """
        logger.info("Performing final critique of the complete notebook")

        # Format all sections into a single notebook representation
        formatted_notebook = format_notebook_for_critique(
            notebook_plan, section_contents
        )

        # Prepare input for the critique
        input_text = f"{FINAL_CRITIQUE_PROMPT}\n\n### Notebook to Evaluate:\n{formatted_notebook}"

        # Use the Responses API for deeper reasoning
        try:
            logger.debug("Calling OpenAI Responses API for final critique")
            response = self.client.responses.create(
                model="o1",  # Use default model o1 as specified
                input=input_text,
                reasoning={"effort": "high"},
            )

            # Extract the critique - ensure it's a string
            final_critique = (
                str(response.output_text)
                if response.output_text is not None
                else "No critique was generated"
            )
            logger.info("Final critique generated successfully")
            return final_critique

        except Exception as e:
            logger.error(f"Failed to generate final critique: {e}")
            return f"Error generating final critique: {str(e)}"

    def save_to_directory(
        self,
        notebook_plan: NotebookPlanModel,
        section_contents: List[NotebookSectionContent],
        output_dir: str = "/output",
        formats: List[str] = ["ipynb", "py", "md", "json"],
        include_critique: bool = True,
        original_content: Optional[List[NotebookSectionContent]] = None,
        critique: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save all important files related to the notebook under a specified directory.

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            section_contents (List[NotebookSectionContent]): The generated content for all sections.
            output_dir (str, optional): Directory to save files. Defaults to "/output".
            formats (List[str], optional): Formats to save in. Defaults to ["ipynb", "py", "md", "json"].
            include_critique (bool, optional): Whether to include critique if available. Defaults to True.
            original_content (Optional[List[NotebookSectionContent]], optional): Original content before revision. Defaults to None.
            critique (Optional[str], optional): Critique text. Defaults to None.

        Returns:
            Dict[str, str]: Dictionary mapping file types to their paths.
        """
        logger.info(f"Saving notebook files to directory: {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Dictionary to store paths to all generated files
        saved_files = {}

        # Save notebook plan as markdown and JSON
        if notebook_plan:
            try:
                # Save notebook plan as markdown
                plan_md_path = os.path.join(output_dir, "notebook_plan.md")
                plan_markdown = notebook_plan_to_markdown(notebook_plan)
                with open(plan_md_path, "w") as f:
                    f.write(plan_markdown)
                saved_files["plan_markdown"] = plan_md_path
                logger.info(f"Saved notebook plan as markdown to {plan_md_path}")

                # Save notebook plan as JSON
                plan_json_path = os.path.join(output_dir, "notebook_plan.json")
                with open(plan_json_path, "w") as f:
                    f.write(notebook_plan.model_dump_json(indent=2))
                saved_files["plan_json"] = plan_json_path
                logger.info(f"Saved notebook plan as JSON to {plan_json_path}")
            except Exception as e:
                logger.error(f"Error saving notebook plan: {e}")

        # If both original content and critique are provided, use save_notebook_versions
        if include_critique and original_content is not None and critique is not None:
            version_files = save_notebook_versions(
                original_content=original_content,
                revised_content=section_contents,
                critique=critique,
                output_dir=output_dir,
                notebook_title=notebook_plan.title if notebook_plan else None,
                formats=formats,
            )
            saved_files.update(version_files)
            logger.info("Saved original and revised versions with critique")
        else:
            # Otherwise, just save the final content
            content_files = writer_output_to_files(
                writer_output=section_contents,
                output_dir=output_dir,
                notebook_title=notebook_plan.title if notebook_plan else None,
                formats=formats,
            )
            saved_files.update(content_files)
            logger.info("Saved final notebook content")

            # If critique is provided but not original content, save critique separately
            if include_critique and critique is not None:
                critique_path = os.path.join(output_dir, "critique.md")
                with open(critique_path, "w") as f:
                    f.write("# Notebook Critique\n\n")
                    f.write(critique)
                saved_files["critique"] = critique_path
                logger.info(f"Saved critique to {critique_path}")

        # If "json" is in formats, save section contents as individual JSON files
        if "json" in formats:
            json_dir = os.path.join(output_dir, "sections_json")
            os.makedirs(json_dir, exist_ok=True)

            for i, section in enumerate(section_contents):
                section_filename = (
                    f"section_{i+1}_{section.section_title.replace(' ', '_')}.json"
                )
                section_path = os.path.join(json_dir, section_filename)
                with open(section_path, "w") as f:
                    f.write(section.model_dump_json(indent=2))

            saved_files["json_sections_dir"] = json_dir
            logger.info(f"Saved individual section JSON files to {json_dir}")

        # Create a README with information about all saved files
        readme_path = os.path.join(output_dir, "README.md")
        try:
            with open(readme_path, "w") as f:
                f.write("# Generated Notebook Files\n\n")
                f.write(
                    f"Notebook Title: {notebook_plan.title if notebook_plan else 'Untitled'}\n\n"
                )
                f.write("## Files Generated\n\n")

                for file_type, file_path in saved_files.items():
                    relative_path = os.path.relpath(file_path, output_dir)
                    f.write(f"- **{file_type}**: [{relative_path}]({relative_path})\n")

                f.write("\n## Generation Information\n\n")
                f.write(
                    f"- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"- Model used: {self.model}\n")
                f.write(f"- Search enabled: {self.search_enabled}\n")
                f.write(f"- Final critique enabled: {self.final_critique_enabled}\n")

            saved_files["readme"] = readme_path
            logger.info(f"Saved README file to {readme_path}")
        except Exception as e:
            logger.error(f"Error creating README file: {e}")

        logger.info(f"Successfully saved {len(saved_files)} files to {output_dir}")
        return saved_files

    def _generate_search_questions(self, section: Section) -> List[Any]:
        """
        Generate questions for searching information about a topic.

        Args:
            section (Section): The section to generate questions for.

        Returns:
            List[Any]: List of SearchQuestion objects.
        """
        # Number of questions to generate
        num_questions = 3  # Can be adjusted as needed

        # Generate specific questions about the topic
        questions_prompt = [
            ChatCompletionSystemMessageParam(
                content="You are an AI assistant that helps create specific questions for finding information about various topics.",
                role="system",
            ),
            ChatCompletionUserMessageParam(
                content=f"Create {num_questions} specific questions to find detailed information about the following topic: {section.title}. The section description is: {section.description}.\n\n{format_subsections_details(section.subsections)}\n\nGenerate questions that would help gather comprehensive information for creating educational content about this topic and its subsections.",
                role="user",
            ),
        ]

        # Generate the questions
        try:
            structured_questions = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=questions_prompt,
                response_format=MultipleSearchQuestions,
                temperature=0.3,
            )
            search_questions = structured_questions.choices[0].message.parsed
            if search_questions and search_questions.questions:
                logger.info(f"Generated {len(search_questions.questions)} questions")
                for i, q in enumerate(search_questions.questions):
                    logger.info(f"Question {i+1}: {q.question}")
                    logger.debug(f"Justification: {q.justification}")
                return search_questions.questions
            else:
                logger.error("Failed to generate valid search questions")
                return []
        except Exception as e:
            logger.error(f"Failed to generate search questions: {e}")
            return []
