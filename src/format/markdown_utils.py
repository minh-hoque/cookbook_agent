"""
Utility functions for working with markdown format.

This module provides functions for converting between markdown and other formats,
with functionality specifically for notebook content.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Union, Optional, Literal

from src.models import NotebookSectionContent, NotebookCell, NotebookPlanModel

# Set up logger
logger = logging.getLogger(__name__)


def notebook_cell_to_markdown(cell: NotebookCell) -> str:
    """
    Convert a single notebook cell to markdown format.

    Args:
        cell: NotebookCell object

    Returns:
        Markdown string representation of the cell
    """
    if cell.cell_type == "markdown":
        return cell.content
    elif cell.cell_type == "code":
        return f"```python\n{cell.content}\n```"
    else:
        return f"Unknown cell type: {cell.cell_type}"


def notebook_section_to_markdown(section: NotebookSectionContent) -> str:
    """
    Convert a notebook section to markdown format.

    Args:
        section: NotebookSectionContent object

    Returns:
        Markdown string representation of the section
    """
    # markdown = f"# {section.section_title}\n\n"
    markdown = ""
    for cell in section.cells:
        markdown += notebook_cell_to_markdown(cell) + "\n\n"

    return markdown


def notebook_content_to_markdown(sections: List[NotebookSectionContent]) -> str:
    """
    Convert a list of notebook sections to a complete markdown document.

    Args:
        sections: List of NotebookSectionContent objects

    Returns:
        Complete markdown document as a string
    """
    markdown = ""

    for section in sections:
        markdown += notebook_section_to_markdown(section) + "\n"

    return markdown


def notebook_plan_to_markdown(plan: NotebookPlanModel) -> str:
    """
    Convert a notebook plan to markdown format.

    Args:
        plan: NotebookPlanModel object

    Returns:
        Markdown string representation of the notebook plan
    """
    markdown = f"# {plan.title}\n\n"
    markdown += f"**Purpose:** {plan.purpose}\n\n"
    markdown += f"**Target Audience:** {plan.target_audience}\n\n"
    markdown += f"**Description:** {plan.description}\n\n"

    markdown += "## Sections\n\n"

    for i, section in enumerate(plan.sections, 1):
        markdown += f"### {i}. {section.title}\n\n"
        markdown += f"{section.description}\n\n"

        if section.subsections:
            for j, subsection in enumerate(section.subsections, 1):
                markdown += f"#### {i}.{j} {subsection.title}\n\n"
                markdown += f"{subsection.description}\n\n"

                if subsection.subsections:
                    for k, subsubsection in enumerate(subsection.subsections, 1):
                        markdown += f"##### {i}.{j}.{k} {subsubsection.title}\n\n"
                        markdown += f"{subsubsection.description}\n\n"

    return markdown


def save_markdown_to_file(markdown: str, filepath: str) -> None:
    """
    Save markdown content to a file.

    Args:
        markdown: Markdown content as a string
        filepath: Path to the output file

    Returns:
        None
    """
    with open(filepath, "w") as f:
        f.write(markdown)


def writer_output_to_markdown(
    writer_output: List[NotebookSectionContent], output_file: Optional[str] = None
) -> str:
    """
    Convert WriterAgent output to markdown and optionally save to a file.

    This is the main function to use for converting WriterAgent output to markdown
    for easy review.

    Args:
        writer_output: List of NotebookSectionContent objects from WriterAgent
        output_file: Optional path to save the markdown output

    Returns:
        Markdown string representation of the writer output
    """
    markdown = notebook_content_to_markdown(writer_output)

    if output_file:
        save_markdown_to_file(markdown, output_file)

    return markdown


def json_file_to_markdown(
    json_file_path: str, output_file: Optional[str] = None, is_section: bool = True
) -> str:
    """
    Convert a JSON file containing notebook content to markdown.

    This function reads a JSON file that contains either a single NotebookSectionContent
    or a list of NotebookSectionContent objects, converts it to markdown, and optionally
    saves the result to a file.

    Args:
        json_file_path: Path to the JSON file
        output_file: Optional path to save the markdown output
        is_section: Whether the JSON file contains a single section (True) or a list of sections (False)

    Returns:
        Markdown string representation of the notebook content
    """
    try:
        with open(json_file_path, "r") as f:
            json_data = json.load(f)

        if is_section:
            # Convert a single section
            section = NotebookSectionContent(**json_data)
            markdown = notebook_section_to_markdown(section)
        else:
            # Convert a list of sections
            sections = [
                NotebookSectionContent(**section_data) for section_data in json_data
            ]
            markdown = notebook_content_to_markdown(sections)

        if output_file:
            save_markdown_to_file(markdown, output_file)

        return markdown

    except Exception as e:
        print(f"Error converting JSON file to markdown: {e}")
        return f"Error: {e}"


def json_string_to_markdown(
    json_string: str, output_file: Optional[str] = None, is_section: bool = True
) -> str:
    """
    Convert a JSON string containing notebook content to markdown.

    This function takes a JSON string that contains either a single NotebookSectionContent
    or a list of NotebookSectionContent objects, converts it to markdown, and optionally
    saves the result to a file.

    Args:
        json_string: JSON string containing notebook content
        output_file: Optional path to save the markdown output
        is_section: Whether the JSON string contains a single section (True) or a list of sections (False)

    Returns:
        Markdown string representation of the notebook content
    """
    try:
        json_data = json.loads(json_string)

        if is_section:
            # Convert a single section
            section = NotebookSectionContent(**json_data)
            markdown = notebook_section_to_markdown(section)
        else:
            # Convert a list of sections
            sections = [
                NotebookSectionContent(**section_data) for section_data in json_data
            ]
            markdown = notebook_content_to_markdown(sections)

        if output_file:
            save_markdown_to_file(markdown, output_file)

        return markdown

    except Exception as e:
        print(f"Error converting JSON string to markdown: {e}")
        return f"Error: {e}"


def format_notebook_for_critique(
    notebook_plan: NotebookPlanModel,
    section_contents: List[NotebookSectionContent],
) -> str:
    """
    Format the notebook sections for the final critique.

    Args:
        notebook_plan (NotebookPlanModel): The notebook plan.
        section_contents (List[NotebookSectionContent]): The generated content for all sections.

    Returns:
        str: The formatted notebook.
    """
    logger.debug("Formatting notebook for final critique")

    # Start with notebook metadata
    formatted = f"# {notebook_plan.title}\n\n"
    formatted += f"**Description:** {notebook_plan.description}\n\n"
    formatted += f"**Purpose:** {notebook_plan.purpose}\n\n"
    formatted += f"**Target Audience:** {notebook_plan.target_audience}\n\n"
    formatted += "---\n\n"

    # Add section headers for each section before including content
    for i, (section, content) in enumerate(
        zip(notebook_plan.sections, section_contents)
    ):
        formatted += f"## Section {i+1}: {section.title}\n\n"
        formatted += notebook_section_to_markdown(content)
        formatted += "\n---\n\n"

    return formatted


def markdown_to_notebook_content(
    markdown_text: str, section_header_level: int = 2
) -> List[NotebookSectionContent]:
    """
    Convert a markdown string to a list of NotebookSectionContent objects.

    This function is the inverse of notebook_content_to_markdown, allowing for
    conversion of markdown back to the structured format used by the WriterAgent.

    Args:
        markdown_text: Markdown text to convert
        section_header_level: The markdown header level that defines sections (default: 2, meaning ## headers)

    Returns:
        List[NotebookSectionContent]: The markdown content in WriterAgent format
    """
    logger.info("Converting markdown to NotebookSectionContent objects")

    # Split the markdown text into lines
    lines = markdown_text.split("\n")

    # Initialize variables
    current_section = None
    current_section_cells = []
    sections = []
    # Use Union[None, Literal] for clarity
    current_cell_type: Union[None, Literal["markdown", "code"]] = None
    current_cell_content = []

    # Helper function to check for section headers
    def is_section_header(line, level):
        header_pattern = f"^{'#' * level} "
        return bool(re.match(header_pattern, line.strip()))

    # Helper function to extract header text
    def extract_header_text(line, level):
        header_pattern = f"^{'#' * level} (.*)"
        match = re.match(header_pattern, line.strip())
        if match:
            return match.group(1).strip()
        return "Untitled Section"

    # Helper function to add the current cell to current_section_cells
    def add_current_cell():
        nonlocal current_cell_type, current_cell_content
        if current_cell_type and current_cell_content:
            content = "\n".join(current_cell_content)
            current_section_cells.append(
                NotebookCell(cell_type=current_cell_type, content=content)
            )
            current_cell_type = None
            current_cell_content = []

    # Process each line in the markdown
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line marks a section header
        if is_section_header(line, section_header_level):
            # If we already have a section, save it
            if current_section is not None:
                add_current_cell()  # Add the last cell
                sections.append(
                    NotebookSectionContent(
                        section_title=current_section, cells=current_section_cells
                    )
                )

            # Start a new section
            current_section = extract_header_text(line, section_header_level)
            current_section_cells = []
            current_cell_type = "markdown"
            current_cell_content = [line]
            i += 1
        # Check if this line starts a code block
        elif line.strip().startswith("```python"):
            # Add the current markdown cell if it exists
            add_current_cell()

            # Start collecting code content
            current_cell_type = "code"
            current_cell_content = []
            i += 1

            # Collect all lines until the closing code block
            while i < len(lines) and not lines[i].strip().startswith("```"):
                current_cell_content.append(lines[i])
                i += 1

            # Skip the closing code block marker
            if i < len(lines):
                i += 1
        # Check if this line starts any other code block (treat as markdown)
        elif line.strip().startswith("```"):
            if current_cell_type != "markdown":
                add_current_cell()
                current_cell_type = "markdown"

            current_cell_content.append(line)
            i += 1

            # Include the code block content in the markdown cell
            while i < len(lines) and not lines[i].strip().startswith("```"):
                current_cell_content.append(lines[i])
                i += 1

            # Add the closing marker
            if i < len(lines):
                current_cell_content.append(lines[i])
                i += 1
        else:
            # Regular markdown content
            if current_cell_type != "markdown":
                add_current_cell()
                current_cell_type = "markdown"

            current_cell_content.append(line)
            i += 1

    # Add the last cell and section if they exist
    if current_section is not None:
        add_current_cell()
        sections.append(
            NotebookSectionContent(
                section_title=current_section, cells=current_section_cells
            )
        )

    # If no sections were found but there is content, create a default section
    if not sections and current_cell_content:
        add_current_cell()
        sections.append(
            NotebookSectionContent(
                section_title="Notebook Content", cells=current_section_cells
            )
        )

    logger.info(f"Converted markdown to {len(sections)} NotebookSectionContent objects")
    return sections
