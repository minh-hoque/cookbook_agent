"""
Utility functions for formatting data, particularly for converting WriterAgent output into markdown.

This module provides utilities to format JSON data and notebook content in various ways,
with a focus on converting notebook content from the WriterAgent into readable markdown for easy review.
"""

import json
from typing import Dict, List, Any, Union, Optional

from src.models import NotebookSectionContent, NotebookCell, NotebookPlanModel


def format_json(data: Any, indent: int = 2) -> str:
    """
    Format any JSON-serializable data with proper indentation.

    Args:
        data: Any JSON-serializable data
        indent: Number of spaces for indentation

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent)


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
