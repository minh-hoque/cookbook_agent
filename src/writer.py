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
from langgraph.checkpoint import MemorySaver
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
    WRITER_SUBSECTION_PROMPT,
    WRITER_REVISION_PROMPT,
)
from src.prompts.critic_prompts import (
    CRITIC_SYSTEM_PROMPT,
    CRITIC_EVALUATION_PROMPT,
)
from src.prompts.prompt_helpers import (
    format_subsections_details,
    format_additional_requirements,
    format_search_results,
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
    Writer Agent class for generating notebook content.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the Writer Agent.

        Args:
            model (str, optional): The OpenAI model to use. Defaults to "gpt-4o".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.critic_llm = ChatOpenAI(model=model, temperature=0.2)

        # Create the workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """
        Create the langgraph workflow for the Writer Agent.

        Returns:
            StateGraph: The langgraph workflow.
        """

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
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("revise", self._revise_node)

        # Add edges
        workflow.add_edge("search", "generate")
        workflow.add_edge("generate", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            self._should_revise,
            {
                True: "revise",
                False: END,
            },
        )
        workflow.add_edge("revise", "evaluate")

        # Set the entry point
        workflow.set_entry_point("search")

        # Compile the workflow
        return workflow.compile()

    def _search_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for information about the current section.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
        # Get the current section
        section = state["current_section"]

        # Create a search query
        messages = [
            SystemMessage(
                content="You are an AI assistant that helps create search queries for finding information about OpenAI API topics."
            ),
            HumanMessage(
                content=f"Create a search query to find information about the following OpenAI API topic: {section.title}. The section description is: {section.description}"
            ),
        ]

        # Get the search query
        structured_llm = self.llm.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke(messages)

        # Log the search query
        logger.info(f"Search query: {search_query.search_query}")
        logger.info(f"Justification: {search_query.justification}")

        # Search for information
        search_results = search_topic(search_query.search_query)
        formatted_results = format_search_results_from_searcher(search_results)

        # Update the state
        state["search_results"] = formatted_results

        return state

    def _generate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content for the current section.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
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

        # Generate the content
        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        # Parse the generated content
        content = response.choices[0].message.content

        # Extract cells from the content
        cells = self._extract_cells(content)

        # Create the notebook section content
        section_content = NotebookSectionContent(
            section_title=section.title,
            cells=cells,
        )

        # Update the state
        state["generated_content"] = section_content

        return state

    def _evaluate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the generated content.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
        # Get the current section, notebook plan, and generated content
        section = state["current_section"]
        notebook_plan = state["notebook_plan"]
        generated_content = state["generated_content"]

        # Format the generated content for evaluation
        formatted_content = self._format_cells_for_evaluation(generated_content.cells)

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
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # Get the evaluation
        structured_critic = self.critic_llm.with_structured_output(CriticEvaluation)
        evaluation = structured_critic.invoke(messages)

        # Log the evaluation
        logger.info(f"Evaluation: {evaluation.model_dump_json(indent=2)}")

        # Update the state
        state["evaluation"] = evaluation

        # If the content passes evaluation, set it as the final content
        if evaluation.pass_content:
            state["final_content"] = generated_content
        else:
            # Increment the retry counter
            state["current_retries"] = state.get("current_retries", 0) + 1

        return state

    def _revise_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revise the generated content based on the evaluation.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            Dict[str, Any]: The updated state.
        """
        # Get the current section, notebook plan, generated content, and evaluation
        section = state["current_section"]
        notebook_plan = state["notebook_plan"]
        generated_content = state["generated_content"]
        evaluation = state["evaluation"]

        # Format the generated content for revision
        formatted_content = self._format_cells_for_evaluation(generated_content.cells)

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
            revision_instructions="Improve the content based on the evaluation feedback.",
        )

        # Generate the revised content
        messages = [
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": revision_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        # Parse the revised content
        content = response.choices[0].message.content

        # Extract cells from the content
        cells = self._extract_cells(content)

        # Create the notebook section content
        section_content = NotebookSectionContent(
            section_title=section.title,
            cells=cells,
        )

        # Update the state
        state["generated_content"] = section_content

        return state

    def _should_revise(self, state: Dict[str, Any]) -> bool:
        """
        Determine if the content should be revised.

        Args:
            state (Dict[str, Any]): The current state.

        Returns:
            bool: True if the content should be revised, False otherwise.
        """
        # Check if we've reached the maximum number of retries
        if state.get("current_retries", 0) >= state.get("max_retries", 3):
            logger.warning(
                f"Reached maximum number of retries ({state.get('max_retries', 3)}). Accepting content as is."
            )
            state["final_content"] = state["generated_content"]
            return False

        return not state["evaluation"].pass_content

    def _extract_cells(self, content: str) -> List[NotebookCell]:
        """
        Extract cells from the generated content.

        Args:
            content (str): The generated content.

        Returns:
            List[NotebookCell]: The extracted cells.
        """
        cells = []

        # Split the content by code blocks
        parts = content.split("```")

        # Skip the first part if it's empty
        start_index = 1 if parts[0].strip() == "" else 0

        # Process each part
        i = start_index
        while i < len(parts):
            # Check if this is a code block
            if i % 2 == start_index:
                # This is a code block header
                block_type = parts[i].strip().lower()

                # Check if the next part exists
                if i + 1 < len(parts):
                    # This is the code block content
                    block_content = parts[i + 1].strip()

                    # Determine the cell type
                    if block_type == "markdown":
                        cell_type = "markdown"
                    elif block_type.startswith("python"):
                        cell_type = "code"
                    else:
                        # Default to code for other types
                        cell_type = "code"

                    # Create the cell
                    cell = NotebookCell(
                        cell_type=cell_type,
                        content=block_content,
                    )

                    # Add the cell to the list
                    cells.append(cell)

                    # Skip the next part
                    i += 2
                else:
                    # No more parts
                    break
            else:
                # This is not a code block header, treat it as markdown
                if parts[i].strip():
                    cell = NotebookCell(
                        cell_type="markdown",
                        content=parts[i].strip(),
                    )
                    cells.append(cell)

                # Move to the next part
                i += 1

        return cells

    def _format_cells_for_evaluation(self, cells: List[NotebookCell]) -> str:
        """
        Format cells for evaluation.

        Args:
            cells (List[NotebookCell]): The cells to format.

        Returns:
            str: The formatted cells.
        """
        formatted = ""

        for cell in cells:
            if cell.cell_type == "markdown":
                formatted += f"```markdown\n{cell.content}\n```\n\n"
            else:
                formatted += f"```python\n{cell.content}\n```\n\n"

        return formatted.strip()

    def _summarize_section_content(
        self, section_content: NotebookSectionContent
    ) -> str:
        """
        Summarize section content for use as context in subsequent sections.

        Args:
            section_content (NotebookSectionContent): The section content to summarize.

        Returns:
            str: A summary of the section content.
        """
        # Create a summary of the content
        summary = ""

        # Add code cells to the summary
        for cell in section_content.cells:
            if cell.cell_type == "code":
                summary += f"Code snippet: {cell.content[:200]}...\n\n"

        return summary

    def generate_section_content(
        self,
        notebook_plan: NotebookPlanModel,
        section_index: int,
        additional_requirements: Optional[Dict[str, Any]] = None,
        previous_content: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
    ) -> NotebookSectionContent:
        """
        Generate content for a section of the notebook.

        Args:
            notebook_plan (NotebookPlanModel): The notebook plan.
            section_index (int): The index of the section to generate content for.
            additional_requirements (Optional[Dict[str, Any]], optional): Additional requirements. Defaults to None.
            previous_content (Optional[Dict[str, str]], optional): Previously generated content. Defaults to None.
            max_retries (int, optional): Maximum number of revision attempts. Defaults to 3.

        Returns:
            NotebookSectionContent: The generated content.
        """
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
        result = self.workflow.invoke(state)

        # Return the final content
        return result["final_content"]

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
            additional_requirements (Optional[Dict[str, Any]], optional): Additional requirements. Defaults to None.
            max_retries (int, optional): Maximum number of revision attempts per section. Defaults to 3.

        Returns:
            List[NotebookSectionContent]: The generated content for all sections.
        """
        # Initialize the list of section contents
        section_contents = []

        # Initialize the previous content dictionary
        previous_content = {}

        # Generate content for each section
        for i, section in enumerate(notebook_plan.sections):
            logger.info(
                f"Generating content for section {i+1}/{len(notebook_plan.sections)}: {section.title}"
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

            # Add the section content to the previous content dictionary
            previous_content[section.title] = self._summarize_section_content(
                section_content
            )

            logger.info(
                f"Completed section {i+1}/{len(notebook_plan.sections)}: {section.title}"
            )

        return section_contents
