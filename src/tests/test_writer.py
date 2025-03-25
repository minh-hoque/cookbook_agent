#!/usr/bin/env python3
"""
Run Writer Test Script

This script provides a simple command-line interface to run the WriterAgent test.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional

# Configure logging
# Set root logger to WARNING to suppress logs from other libraries
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s",
)

# Get a logger for this module and set it to DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------
# UPDATED CODE: More comprehensive reduction of external library logging
# Reduce logging for external libraries to WARNING or higher:
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# Silence other noisy libraries
for logger_name in logging.root.manager.loggerDict:
    if any(
        name in logger_name.lower()
        for name in ["http", "urllib", "requests", "openai", "aiohttp"]
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
# ---------------------------------------------------------------------

# Import format_utils functions at the top level
from src.format.format_utils import (
    notebook_content_to_markdown,
    save_markdown_to_file,
    writer_output_to_notebook,
    save_notebook_versions,
)
from src.models import NotebookSectionContent


def parse_markdown_to_plan(markdown_file: str) -> Dict[str, Any]:
    """
    Parse a markdown file into a dictionary representing a notebook plan.

    This function reads a markdown file containing a notebook plan and converts it
    into a dictionary that can be used to create a NotebookPlanModel.

    Args:
        markdown_file (str): Path to the markdown file containing the notebook plan.

    Returns:
        Dict[str, Any]: A dictionary representing the notebook plan.
    """
    with open(markdown_file, "r") as f:
        content = f.read()

    lines = content.split("\n")

    # Extract title (first line starting with #)
    title = lines[0].replace("# ", "").strip()

    # Extract description, purpose, and target audience
    description = ""
    purpose = ""
    target_audience = ""

    for line in lines:
        if line.startswith("**Description:**"):
            description = line.replace("**Description:**", "").strip()
        elif line.startswith("**Purpose:**"):
            purpose = line.replace("**Purpose:**", "").strip()
        elif line.startswith("**Target Audience:**"):
            target_audience = line.replace("**Target Audience:**", "").strip()

    # Parse sections
    sections = []
    current_section = None
    current_subsection = None

    for line in lines:
        # Section (starts with ###)
        if line.startswith("### "):
            # Remove any numbering (e.g., "### 1. Title" -> "Title")
            section_title = line.replace("### ", "").strip()
            if "." in section_title and section_title.split(".", 1)[0].isdigit():
                section_title = section_title.split(".", 1)[1].strip()

            current_section = {
                "title": section_title,
                "description": "",
                "subsections": [],
            }
            sections.append(current_section)
            current_subsection = None

        # Subsection (starts with ####)
        elif line.startswith("#### "):
            if current_section is None:
                continue

            # Remove any numbering
            subsection_title = line.replace("#### ", "").strip()
            if (
                "." in subsection_title
                and subsection_title.split(".")[-1].strip().isdigit()
            ):
                subsection_title = ".".join(subsection_title.split(".")[:-1]).strip()
            elif "." in subsection_title and subsection_title.split(".")[0].isdigit():
                subsection_title = subsection_title.split(".", 1)[1].strip()

            current_subsection = {
                "title": subsection_title,
                "description": "",
                "subsections": [],
            }
            current_section["subsections"].append(current_subsection)

        # Sub-subsection (starts with #####)
        elif line.startswith("##### "):
            if current_section is None or current_subsection is None:
                continue

            # Remove any numbering
            subsubsection_title = line.replace("##### ", "").strip()
            if "." in subsubsection_title:
                parts = subsubsection_title.split(".")
                if parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                    subsubsection_title = ".".join(parts[3:]).strip()

            subsubsection = {"title": subsubsection_title, "description": ""}
            current_subsection["subsections"].append(subsubsection)

        # Description text (not a header)
        elif not line.startswith("#") and line.strip() and current_section is not None:
            if current_subsection is not None:
                # Check if we're in a sub-subsection
                if current_subsection["subsections"] and not line.startswith("**"):
                    current_subsection["subsections"][-1]["description"] += line + "\n"
                else:
                    current_subsection["description"] += line + "\n"
            else:
                current_section["description"] += line + "\n"

    # Create and return the plan dictionary
    return {
        "title": title,
        "description": description,
        "purpose": purpose,
        "target_audience": target_audience,
        "sections": sections,
    }


def save_notebook_content(
    content_list: List[Dict[str, Any]],
    output_dir: str,
    is_single_section: bool = False,
    notebook_title: Optional[str] = None,
    section_index: Optional[int] = None,
) -> None:
    """
    Save the generated notebook content to files.

    This function saves notebook content in multiple formats:
    1. Individual JSON files for each section
    2. A combined markdown file
    3. A Jupyter notebook (.ipynb) file

    Args:
        content_list (List[Dict[str, Any]]): List of notebook section contents.
        output_dir (str): Directory to save the content to.
        is_single_section (bool): Whether this is saving content for a single section.
        notebook_title (Optional[str]): Title of the notebook (used for the complete notebook).
        section_index (Optional[int]): Index of the section if is_single_section is True.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, section_content in enumerate(content_list):
        # Create a sanitized filename
        section_title = section_content.get("section_title", f"section_{i+1}")
        filename = (
            f"section_{i+1}_{section_title.replace(' ', '_').replace(':', '')}.json"
        )
        filepath = os.path.join(output_dir, filename)

        # Save the content as JSON
        with open(filepath, "w") as f:
            json.dump(section_content, f, indent=2)

        logger.info(f"Saved section {i+1} to {filepath}")

    # Also save as markdown and notebook files
    try:
        # Convert the dictionary list to NotebookSectionContent objects
        section_objects = [
            NotebookSectionContent(**section) for section in content_list
        ]

        # Generate markdown from all sections
        markdown_content = notebook_content_to_markdown(section_objects)

        # Create a descriptive filename using the first section's title
        if content_list and "section_title" in content_list[0]:
            first_title = (
                content_list[0]["section_title"].replace(" ", "_").replace(":", "")
            )
            markdown_filename = f"notebook_{first_title}.md"
        else:
            markdown_filename = "full_notebook.md"

        # Save the markdown to a file
        markdown_filepath = os.path.join(output_dir, markdown_filename)
        save_markdown_to_file(markdown_content, markdown_filepath)
        logger.info(f"Saved complete notebook as markdown to {markdown_filepath}")

        # Save as Jupyter notebook (.ipynb)
        if is_single_section:
            # For single section, use a filename based on the section title
            section_title = content_list[0].get(
                "section_title",
                f"section_{section_index+1 if section_index is not None else 1}",
            )
            notebook_filename = f"section_{section_index+1 if section_index is not None else 1}_{section_title.replace(' ', '_')}.ipynb"
            section_notebook_title = f"Section {section_index+1 if section_index is not None else 1}: {section_title}"
            notebook_filepath = os.path.join(output_dir, notebook_filename)

            if writer_output_to_notebook(
                section_objects,
                notebook_filepath,
                notebook_title=section_notebook_title,
            ):
                logger.info(f"Saved section as Jupyter notebook to {notebook_filepath}")
        else:
            # For complete notebook, use the provided notebook title
            if notebook_title:
                notebook_filename = f"{notebook_title.replace(' ', '_')}.ipynb"
                notebook_filepath = os.path.join(output_dir, notebook_filename)

                if writer_output_to_notebook(
                    section_objects, notebook_filepath, notebook_title=notebook_title
                ):
                    logger.info(
                        f"Saved complete Jupyter notebook to {notebook_filepath}"
                    )
            else:
                # Fallback to a generic name if no title provided
                notebook_filename = "full_notebook.ipynb"
                notebook_filepath = os.path.join(output_dir, notebook_filename)

                if writer_output_to_notebook(
                    section_objects,
                    notebook_filepath,
                    notebook_title="Generated Notebook",
                ):
                    logger.info(
                        f"Saved complete Jupyter notebook to {notebook_filepath}"
                    )

    except Exception as e:
        logger.error(f"Error saving notebook versions: {e}")


def run_writer_test(
    plan_file: str,
    output_dir: str = "output",
    section_index: Optional[int] = None,
    model: str = "gpt-4o",
    max_retries: int = 3,
    parse_only: bool = False,
) -> None:
    """
    Run a test of the WriterAgent.

    Args:
        plan_file (str): Path to the markdown plan file.
        output_dir (str): Directory to save the output to.
        section_index (Optional[int]): Index of the section to generate content for.
            If None, generate content for all sections.
        model (str): The model to use for generation.
        max_retries (int): Maximum number of retries for content generation.
        parse_only (bool): Only parse the plan and save it as JSON, don't generate content.
    """
    logger.info(f"Loading plan from {plan_file}")
    plan_dict = parse_markdown_to_plan(plan_file)

    logger.info(f"Loaded plan: {plan_dict['title']}")
    logger.info(f"Plan has {len(plan_dict['sections'])} sections")

    # Save the plan as JSON for reference
    os.makedirs(output_dir, exist_ok=True)
    plan_json_path = os.path.join(output_dir, "notebook_plan.json")
    with open(plan_json_path, "w") as f:
        json.dump(plan_dict, f, indent=2)
    logger.info(f"Saved plan to {plan_json_path}")

    # If parse_only is True, we're done
    if parse_only:
        logger.info("Parse-only mode, skipping content generation")
        return

    # Import the WriterAgent here to avoid import issues
    try:
        # Add the parent directory to the path so we can import the src module
        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        )

        from src.writer import WriterAgent
        from src.models import NotebookPlanModel

        # Convert the plan dictionary to a NotebookPlanModel
        plan = NotebookPlanModel.model_validate(plan_dict)

        # Initialize the WriterAgent
        logger.info(f"Initializing WriterAgent with model {model}")
        writer = WriterAgent(model=model)

        # Additional requirements
        additional_requirements = []
        additional_requirements.append("Include detailed comments in code examples")
        additional_requirements.append(
            "Use clear markdown formatting with proper headers and code blocks"
        )

        # Generate content
        if section_index is not None:
            # Generate content for a specific section
            if section_index < 0 or section_index >= len(plan.sections):
                logger.error(
                    f"Invalid section index: {section_index}. Plan has {len(plan.sections)} sections."
                )
                return

            logger.info(
                f"Generating content for section {section_index+1}: {plan.sections[section_index].title}"
            )
            content = writer.generate_section_content(
                notebook_plan=plan,
                section_index=section_index,
                additional_requirements=additional_requirements,
            )

            # Save the content
            save_notebook_content(
                [content.model_dump()],
                output_dir,
                is_single_section=True,
                section_index=section_index,
            )

        else:
            # Generate content for all sections
            logger.info("Generating content for all sections")
            original_content, final_critique, revised_content = writer.generate_content(
                notebook_plan=plan,
                additional_requirements=additional_requirements,
            )

            save_notebook_versions(
                original_content,  # type: ignore
                revised_content,  # type: ignore
                final_critique,  # type: ignore
                output_dir,
                notebook_title=plan.title,
                formats=["ipynb", "md"],
            )

        logger.info("Content generation complete")
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        logger.info("Saving plan only, skipping content generation")
    except Exception as e:
        logger.error(f"Error generating content: {e}")
        logger.info("Plan was saved to JSON, but content generation failed")


def main():
    """
    Main function for running the WriterAgent test from the command line.

    example:
    python -m src.tests.test_writer --model gpt-4o --plan /Users/minhajul/personal/github/cookbook_agent/data/test_writer_notebook_plan.md
    python -m src.tests.test_writer --model gpt-4o --plan /Users/minhajul/personal/github/cookbook_agent/data/agents_sdk_synthetic_transcripts.md
    """
    parser = argparse.ArgumentParser(description="Test the WriterAgent")
    parser.add_argument(
        "--plan",
        type=str,
        default="./data/test_writer_notebook_plan.md",
        help="Path to the markdown plan file",
    )
    parser.add_argument(
        "--output", type=str, default="./output", help="Directory to save the output to"
    )
    parser.add_argument(
        "--section",
        type=int,
        default=None,
        help="Index of the section to generate content for (0-based)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="The model to use for generation"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for content generation",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Only parse the plan and save it as JSON, don't generate content",
    )

    args = parser.parse_args()

    run_writer_test(
        plan_file=args.plan,
        output_dir=args.output,
        section_index=args.section,
        model=args.model,
        max_retries=args.max_retries,
        parse_only=args.parse_only,
    )


if __name__ == "__main__":
    main()
