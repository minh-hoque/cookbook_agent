"""
Utility functions for formatting prompt elements.

This module provides functions specifically for formatting various components
of prompts used by the models.
"""

import logging
from typing import Dict, List, Any, Union

# Set up logger
logger = logging.getLogger(__name__)


def format_subsections_details(subsections):
    """
    Format subsections details for the prompt.

    Args:
        subsections: List of subsections to format.

    Returns:
        str: Formatted subsections details.
    """
    logger.info(f"Formatting {len(subsections) if subsections else 0} subsections")

    if not subsections:
        logger.debug("No subsections to format")
        return "No subsections specified"

    formatted = ""
    for i, subsection in enumerate(subsections, 1):
        formatted += f"### Subsection {i}: {subsection.title}\n"
        formatted += f"Description: {subsection.description}\n\n"

        if subsection.subsections:
            logger.debug(
                f"Found {len(subsection.subsections)} sub-subsections for {subsection.title}"
            )
            formatted += "Sub-subsections:\n"
            for j, subsubsection in enumerate(subsection.subsections, 1):
                formatted += (
                    f"- {j}. {subsubsection.title}: {subsubsection.description}\n"
                )
            formatted += "\n"

    return formatted.strip() or "No subsections specified"


def format_additional_requirements(requirements):
    """
    Format additional requirements for the prompt.

    Args:
        requirements: Additional requirements to format. Can be a list, dict, or string.

    Returns:
        str: Formatted additional requirements.
    """
    logger.info(
        f"Formatting additional requirements of type: {type(requirements).__name__}"
    )

    if not requirements:
        logger.debug("No requirements to format")
        return "None specified"

    formatted = ""

    # Handle both list and dict inputs
    if isinstance(requirements, list):
        for req in requirements:
            formatted += f"- {req}\n"
    elif isinstance(requirements, dict):
        for key, value in requirements.items():
            if key not in [
                "description",
                "purpose",
                "target_audience",
                "code_snippets",
            ]:
                formatted += f"- {key}: {value}\n"
    else:
        # logger.debug(f"Processing requirements as string: {requirements}")
        formatted = f"- {str(requirements)}\n"

    return formatted.strip() or "None specified"


def format_previous_content(previous_content):
    """
    Format previously generated content for the prompt.

    Args:
        previous_content: Dictionary mapping section titles to NotebookSectionContent objects or strings.

    Returns:
        str: Formatted previous content.
    """
    logger.info(
        f"Formatting {len(previous_content) if previous_content else 0} previous content sections"
    )

    if not previous_content:
        logger.debug("No previous content to format")
        return "No previously generated content available"

    formatted = "### Previously Generated Sections:\n\n"

    for section_title, content in previous_content.items():
        # logger.debug(f"Formatting section: {section_title}")
        formatted += f"#### {section_title}\n"

        # Check if content is a NotebookSectionContent object
        if hasattr(content, "cells"):
            # Format each cell in the section
            for cell in content.cells:
                if cell.cell_type == "markdown":
                    formatted += f"**Markdown:**\n{cell.content}\n\n"
                elif cell.cell_type == "code":
                    formatted += f"**Code:**\n```python\n{cell.content}\n```\n\n"
        else:
            # Handle the case where content is a string (for backward compatibility)
            formatted += f"{content}\n\n"

    return formatted.strip() or "No previously generated content available"


def format_code_snippets(snippets):
    """
    Format code snippets for the prompt.

    Args:
        snippets: List of code snippets to format.

    Returns:
        str: Formatted code snippets.
    """
    logger.info(f"Formatting {len(snippets) if snippets else 0} code snippets")

    if not snippets:
        logger.debug("No code snippets to format")
        return "None provided"

    formatted = ""
    for i, snippet in enumerate(snippets, 1):
        formatted += f"Snippet {i}:\n```python\n{snippet}\n```\n\n"

    return formatted.strip() or "None provided"


def format_clarifications(clarifications):
    """
    Format clarifications for the prompt.

    Args:
        clarifications: Dictionary mapping questions to answers.

    Returns:
        str: Formatted clarifications.
    """
    logger.info(
        f"Formatting {len(clarifications) if clarifications else 0} clarifications"
    )

    if not clarifications:
        logger.debug("No clarifications to format")
        return "No clarifications provided"

    formatted = ""
    for question, answer in clarifications.items():
        formatted += f"Q: {question}\nA: {answer}\n\n"

    return formatted.strip() or "No clarifications provided"


def format_cells_for_evaluation(cells):
    """
    Format cells for evaluation.

    Args:
        cells: The cells to format.

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
