"""
Example script to demonstrate how to use the Writer Agent.

This script shows how to use the Writer Agent to generate content for a notebook.
"""

import os
import json
import logging
from typing import Dict, Any, List

from src.planner import PlannerLLM
from src.writer import WriterAgent
from src.models import NotebookPlanModel, NotebookSectionContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Get a logger for this module
logger = logging.getLogger(__name__)


def load_notebook_plan(plan_file: str) -> NotebookPlanModel:
    """
    Load a notebook plan from a file.

    Args:
        plan_file (str): The path to the plan file.

    Returns:
        NotebookPlanModel: The notebook plan.
    """
    with open(plan_file, "r") as f:
        plan_data = json.load(f)

    return NotebookPlanModel.model_validate(plan_data)


def save_section_content(
    section_content: NotebookSectionContent, output_file: str
) -> None:
    """
    Save section content to a file.

    Args:
        section_content (NotebookSectionContent): The section content to save.
        output_file (str): The path to the output file.
    """
    with open(output_file, "w") as f:
        json.dump(section_content.model_dump(), f, indent=2)


def save_all_sections(
    section_contents: List[NotebookSectionContent], output_dir: str
) -> None:
    """
    Save all section contents to files.

    Args:
        section_contents (List[NotebookSectionContent]): The section contents to save.
        output_dir (str): The directory to save the files to.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each section content
    for i, section_content in enumerate(section_contents):
        output_file = os.path.join(
            output_dir,
            f"section_{i+1}_{section_content.section_title.replace(' ', '_')}.json",
        )
        save_section_content(section_content, output_file)
        logger.info(f"Saved section {i+1} to {output_file}")


def main():
    """
    Main function to demonstrate the Writer Agent.
    """
    # Load the notebook plan
    plan_file = "notebook_plan.json"
    if not os.path.exists(plan_file):
        # Generate a plan if it doesn't exist
        logger.info("Generating a notebook plan...")
        planner = PlannerLLM()
        requirements = {
            "description": "A notebook to demonstrate how to use the OpenAI Chat Completions API",
            "purpose": "To help developers understand how to use the Chat Completions API effectively",
            "target_audience": "Python developers who are new to OpenAI APIs",
        }
        plan = planner.plan_notebook(requirements)

        # Save the plan
        with open(plan_file, "w") as f:
            json.dump(plan.model_dump(), f, indent=2)
    else:
        # Load the existing plan
        logger.info(f"Loading notebook plan from {plan_file}...")
        plan = load_notebook_plan(plan_file)

    # Print the plan sections
    logger.info(f"Notebook plan: {plan.title}")
    for i, section in enumerate(plan.sections):
        logger.info(f"Section {i+1}: {section.title}")

    # Initialize the Writer Agent
    writer = WriterAgent()

    # Additional requirements (optional)
    additional_requirements = {
        "code_style": "Include detailed comments in code examples",
        "error_handling": "Show proper error handling for API calls",
    }

    # Generate content for all sections
    logger.info("Generating content for all sections...")
    section_contents = writer.generate_content(
        notebook_plan=plan,
        additional_requirements=additional_requirements,
        max_retries=2,  # Limit retries for faster example
    )

    # Save all section contents
    output_dir = "notebook_sections"
    logger.info(f"Saving all section contents to {output_dir}...")
    save_all_sections(section_contents, output_dir)

    logger.info(f"Generated content for {len(section_contents)} sections. Done!")


if __name__ == "__main__":
    main()
