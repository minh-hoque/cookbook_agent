"""
Plan formatting utilities.

This module provides functions for formatting notebook plans.
"""

import logging
from typing import Optional
from src.models import NotebookPlanModel, Section, SubSection, SubSubSection

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


def parse_markdown_to_plan(markdown_file: str) -> NotebookPlanModel:
    """
    Parse a markdown file into a NotebookPlanModel.

    Args:
        markdown_file (str): Path to the markdown file containing the notebook plan.

    Returns:
        NotebookPlanModel: A NotebookPlanModel object parsed from the markdown.
    """
    logger.info(f"Parsing markdown plan file: {markdown_file}")
    try:
        with open(markdown_file, "r") as f:
            content = f.read()

        lines = content.split("\n")

        # Extract title (first line starting with #)
        title = ""
        for line in lines:
            if line.startswith("# "):
                title = line.replace("# ", "").strip()
                break

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
            # Section (starts with ###, format used in our markdown output)
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

            # Section description (not a header and after a section)
            elif (
                not line.startswith("#")
                and current_section is not None
                and current_subsection is None
            ):
                current_section["description"] += line + "\n"

            # Subsection (starts with ####)
            elif line.startswith("#### "):
                if current_section is None:
                    continue

                subsection_title = line.replace("#### ", "").strip()
                if (
                    "." in subsection_title
                    and subsection_title.split(".", 1)[0].isdigit()
                ):
                    # Handle format like "1.1. Title"
                    parts = subsection_title.split(".")
                    if (
                        len(parts) >= 2
                        and parts[0].isdigit()
                        and parts[1].strip().isdigit()
                    ):
                        subsection_title = ".".join(parts[2:]).strip()
                    # Handle format like "1. Title"
                    elif parts[0].isdigit():
                        subsection_title = ".".join(parts[1:]).strip()

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

                sub_subsection_title = line.replace("##### ", "").strip()
                if "." in sub_subsection_title:
                    # Handle format like "1.1.1. Title"
                    parts = sub_subsection_title.split(".")
                    if len(parts) >= 3 and all(
                        p.strip().isdigit() for p in parts[0:3] if p.strip()
                    ):
                        sub_subsection_title = ".".join(parts[3:]).strip()

                sub_subsection = {"title": sub_subsection_title, "description": ""}
                current_subsection["subsections"].append(sub_subsection)

            # Subsection or Sub-subsection description
            elif not line.startswith("#") and current_subsection is not None:
                # Check if we're in a sub-subsection
                if current_subsection["subsections"] and not any(
                    line.startswith(f"**{marker}:**")
                    for marker in ["Description", "Purpose", "Target Audience"]
                ):
                    current_subsection["subsections"][-1]["description"] += line + "\n"
                else:
                    current_subsection["description"] += line + "\n"

        # Create section objects
        section_objects = []
        for section_dict in sections:
            subsection_objects = []

            # Process subsections if they exist
            if "subsections" in section_dict and section_dict["subsections"]:
                for subsection_dict in section_dict["subsections"]:
                    sub_subsection_objects = []

                    # Process sub-subsections if they exist
                    if (
                        "subsections" in subsection_dict
                        and subsection_dict["subsections"]
                    ):
                        for sub_subsection_dict in subsection_dict["subsections"]:
                            sub_subsection_objects.append(
                                SubSubSection(
                                    title=sub_subsection_dict["title"],
                                    description=sub_subsection_dict[
                                        "description"
                                    ].strip(),
                                )
                            )

                    subsection_objects.append(
                        SubSection(
                            title=subsection_dict["title"],
                            description=subsection_dict["description"].strip(),
                            subsections=(
                                sub_subsection_objects
                                if sub_subsection_objects
                                else None
                            ),
                        )
                    )

            section_objects.append(
                Section(
                    title=section_dict["title"],
                    description=section_dict["description"].strip(),
                    subsections=subsection_objects if subsection_objects else None,
                )
            )

        # Create and return the plan
        return NotebookPlanModel(
            title=title,
            description=description,
            purpose=purpose,
            target_audience=target_audience,
            sections=section_objects,
        )

    except Exception as e:
        logger.error(f"Error parsing markdown plan: {str(e)}")
        raise
