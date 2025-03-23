"""
Plan formatting utilities.

This module provides functions for formatting notebook plans.
"""

import logging
from typing import Optional
from src.models import NotebookPlanModel

# Get a logger for this module
logger = logging.getLogger(__name__)


def format_notebook_plan(plan: NotebookPlanModel) -> str:
    """
    Format the notebook plan as a markdown string.

    Args:
        plan: The notebook plan to format

    Returns:
        A markdown string representation of the plan
    """
    logger.debug("Formatting notebook plan")

    # Header section
    result = f"# {plan.title}\n\n"
    result += f"**Description:** {plan.description}\n\n"
    result += f"**Purpose:** {plan.purpose}\n\n"
    result += f"**Target Audience:** {plan.target_audience}\n\n"
    result += "## Outline\n\n"

    # Process sections
    for i, section in enumerate(plan.sections, 1):
        result += f"### {i}. {section.title}\n\n"
        result += f"{section.description}\n\n"

        # Process subsections if they exist
        if section.subsections:
            for j, subsection in enumerate(section.subsections, 1):
                result += f"#### {i}.{j}. {subsection.title}\n\n"
                result += f"{subsection.description}\n\n"

                # Process sub-subsections if they exist
                if subsection.subsections:
                    for k, sub_subsection in enumerate(subsection.subsections, 1):
                        result += f"##### {i}.{j}.{k}. {sub_subsection.title}\n\n"
                        result += f"{sub_subsection.description}\n\n"

    logger.debug("Notebook plan formatting complete")
    return result


def save_plan_to_file(plan: NotebookPlanModel, output_file: str) -> None:
    """
    Save the formatted notebook plan to a file.

    Args:
        plan: The notebook plan to save
        output_file: Path to the output file
    """
    logger.debug(f"Saving notebook plan to file: {output_file}")
    try:
        formatted_plan = format_notebook_plan(plan)
        with open(output_file, "w") as f:
            f.write(formatted_plan)
        logger.info(f"Notebook plan saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving notebook plan: {str(e)}")
        raise
