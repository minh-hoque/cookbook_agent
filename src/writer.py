"""
Writer Agent Module

This module is responsible for generating notebook content based on the plan created by the Planner LLM.
It uses langgraph to create a workflow that includes the Writer LLM and the Critic LLM.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Use absolute imports for proper package imports when running from root
from src.models import (
    NotebookPlanModel,
    Section,
    SubSection,
    SearchQuery,
    NotebookSectionContent,
    NotebookCell,
    CriticEvaluation,
)
from src.prompts.writer_prompts import (
    WRITER_SYSTEM_PROMPT,
    WRITER_CONTENT_PROMPT,
    WRITER_REVISION_PROMPT,
)
from src.prompts.critic_prompts import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_EVALUATION_PROMPT,
)
from src.prompts.prompt_helpers import (
    format_subsections_details,
    format_additional_requirements,
    format_previous_content,
)
from src.searcher import (
    search_topic,
    format_search_results as format_search_results_from_searcher,
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

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Writer Agent.

        Args:
            model (str, optional): The model to use for generating content. Defaults to "gpt-4o".
        """
        logger.info(f"Initializing WriterAgent with model: {model}")

        # Set the model
        self.model = model

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

        # Create the graph
        workflow = StateGraph(State)

        # Add nodes
        workflow.add_node("search", self._search_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("revise", self._revise_node)

        # Add edges
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", "critic")
        workflow.add_conditional_edges(
            "critic",
            self._should_revise,
            {
                True: "revise",
                False: END,
            },
        )
        workflow.add_edge("revise", "critic")

        # Set the entry point
        workflow.set_entry_point("search")

        # Compile the workflow
        logger.debug(f"Workflow created with nodes: {workflow.nodes}")
        return workflow

    def _search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for information related to the current section.

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

        # Create a search query
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps create search queries for finding information about OpenAI API topics.",
            },
            {
                "role": "user",
                "content": f"Create a search query to find information about the following OpenAI API topic: {section.title}. The section description is: {section.description}",
            },
        ]

        # Generate the search query
        try:
            structured_search = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=SearchQuery,
                temperature=0.1,
            )
            search_query = structured_search.choices[0].message.parsed
            logger.info(f"Search query: {search_query.search_query}")
            logger.info(f"Justification: {search_query.justification}")
        except Exception as e:
            logger.error(f"Failed to generate search query: {e}")
            state["search_results"] = "Error generating search query"
            return state

        # Search for information
        try:
            search_results = search_topic(search_query.search_query)
            formatted_results = format_search_results_from_searcher(search_results)
            logger.debug(f"Search returned {len(search_results)} results")
        except Exception as e:
            logger.error(f"Failed to search for information: {e}")
            formatted_results = "Error searching for information"

        # Update the state
        state["search_results"] = formatted_results
        logger.debug("Search node completed successfully")
        return state

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
        prompt = WRITER_CONTENT_PROMPT.format(
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
            f"Formatted content generation prompt with {len(prompt)} characters"
        )

        # Generate the content
        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Use structured output to get the notebook section content
        try:
            logger.debug("Calling OpenAI API to generate content")
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=NotebookSectionContent,
                temperature=0.5,
            )

            # Get the parsed section content
            section_content = response.choices[0].message.parsed
            if section_content and hasattr(section_content, "cells"):
                logger.info(
                    f"Generated content with {len(section_content.cells)} cells"
                )
                logger.debug(
                    f"Content types: {[cell.cell_type for cell in section_content.cells]}"
                )
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            # Create an empty section content as fallback
            section_content = NotebookSectionContent(
                section_title=section.title,
                cells=[
                    NotebookCell(
                        cell_type="markdown",
                        content=f"Error generating content for section: {section.title}",
                    )
                ],
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
        prompt = CRITIC_EVALUATION_PROMPT.format(
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

        # Evaluate the content
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Get the evaluation
        try:
            logger.debug("Calling Critic LLM to evaluate content")
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=CriticEvaluation,
                temperature=0,
            )
            evaluation = response.choices[0].message.parsed
            logger.info(f"Evaluation: {evaluation}")
        except Exception as e:
            logger.error(f"Failed to evaluate content: {e}")
            # Create a default evaluation as fallback
            evaluation = CriticEvaluation(
                rationale="Error evaluating content", passed=False
            )

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
        revision_prompt = WRITER_REVISION_PROMPT.format(
            notebook_title=notebook_plan.title,
            notebook_description=notebook_plan.description,
            notebook_purpose=notebook_plan.purpose,
            notebook_target_audience=notebook_plan.target_audience,
            section_title=section.title,
            section_description=section.description,
            original_content=formatted_content,
            evaluation_feedback=evaluation.rationale,
        )

        # Generate the revised content
        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": revision_prompt},
        ]

        # Use structured output to get the revised notebook section content
        try:
            logger.debug("Calling OpenAI API to revise content")
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=NotebookSectionContent,
                temperature=0.3,
            )

            # Get the parsed section content
            section_content = response.choices[0].message.parsed
            if section_content and hasattr(section_content, "cells"):
                logger.info(f"Revised content with {len(section_content.cells)} cells")
                logger.debug(
                    f"Revised content types: {[cell.cell_type for cell in section_content.cells]}"
                )
        except Exception as e:
            logger.error(f"Failed to revise content: {e}")
            # Keep the original content as fallback
            section_content = generated_content
            logger.warning("Using original content as fallback due to revision error")

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
        logger.debug(f"Formatting {len(cells)} cells for evaluation")

        formatted = ""

        for cell in cells:
            if cell.cell_type == "markdown":
                formatted += f"```markdown\n{cell.content}\n```\n\n"
            else:
                formatted += f"```python\n{cell.content}\n```\n\n"

        formatted_content = formatted.strip()
        logger.debug(
            f"Formatted content for evaluation with {len(formatted_content)} characters"
        )

        return formatted_content

    def generate_section_content(
        self,
        notebook_plan: NotebookPlanModel,
        section_index: int,
        additional_requirements: Optional[Dict[str, Any]] = None,
        previous_content: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
    ) -> NotebookSectionContent:
        """
        Generate content for a specific section of the notebook.

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            section_index (int): The index of the section to generate content for.
            additional_requirements (Optional[Dict[str, Any]], optional): Additional requirements for the content. Defaults to None.
            previous_content (Optional[Dict[str, str]], optional): Content from previous sections. Defaults to None.
            max_retries (int, optional): Maximum number of retries for content generation. Defaults to 3.

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
            "max_retries": max_retries,
            "current_retries": 0,
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
        # print("final_content", final_content)

        logger.info(
            f"Generated content for section {section_index + 1} with "
            f"{len(final_content.cells)} cells"
        )
        return final_content

    def generate_content(
        self,
        notebook_plan: NotebookPlanModel,
        additional_requirements: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> List[NotebookSectionContent]:
        """
        Generate content for all sections of the notebook.

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            additional_requirements (Optional[Dict[str, Any]], optional): Additional requirements for the content. Defaults to None.
            max_retries (int, optional): Maximum number of retries for content generation. Defaults to 3.

        Returns:
            List[NotebookSectionContent]: The generated content for all sections.
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
                max_retries=max_retries,
            )

            # Add the section content to the list
            section_contents.append(section_content)

            # Add the full section content to the previous content dictionary
            previous_content[section.title] = section_content

            logger.info(f"Completed section {i + 1}/{len(notebook_plan.sections)}")

        logger.info(
            f"Content generation completed for all {len(notebook_plan.sections)} sections"
        )
        return section_contents
