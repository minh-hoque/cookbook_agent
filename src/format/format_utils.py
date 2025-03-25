"""
Utility functions for formatting data, particularly for converting WriterAgent output into markdown.

This module provides utilities to format JSON data and notebook content in various ways,
with a focus on converting notebook content from the WriterAgent into readable markdown for easy review.
It also contains helper functions for formatting prompt elements.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Union, Optional, Literal

from src.models import NotebookSectionContent, NotebookCell, NotebookPlanModel

# Set up logger
logger = logging.getLogger(__name__)


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


# Functions moved from prompt_helpers.py


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


def save_notebook_content(
    content_list: List[Dict[str, Any]], output_dir: str
) -> Optional[str]:
    """
    Save the generated notebook content to files.

    Args:
        content_list (List[Dict[str, Any]]): List of notebook section contents.
        output_dir (str): Directory to save the content to.

    Returns:
        Optional[str]: The path to the markdown file if successful, None otherwise.
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

    # Also save a single markdown file with all content
    try:
        # Convert the dictionary list to NotebookSectionContent objects
        section_objects = [
            NotebookSectionContent.model_validate(section) for section in content_list
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
        return markdown_filepath
    except Exception as e:
        logger.error(f"Error saving markdown version: {e}")
        return None


def writer_output_to_notebook(
    writer_output: List[NotebookSectionContent],
    output_file: str,
    metadata: Optional[Dict[str, Any]] = None,
    notebook_title: Optional[str] = None,
) -> bool:
    """
    Convert WriterAgent output to Jupyter Notebook (.ipynb) format and save to a file.

    This function takes WriterAgent output (list of NotebookSectionContent objects),
    converts it to Jupyter Notebook format, and saves it to a file.

    Args:
        writer_output: List of NotebookSectionContent objects from WriterAgent
        output_file: Path to save the notebook output (should end with .ipynb)
        metadata: Optional metadata to include in the notebook
        notebook_title: Optional title for the notebook (used in markdown header)

    Returns:
        bool: True if the notebook was saved successfully, False otherwise
    """
    logger.info(f"Converting WriterAgent output to notebook format")

    try:
        # Initialize notebook structure
        notebook = {
            "metadata": metadata
            or {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 4,
            "cells": [],
        }

        # Add title cell if provided
        if notebook_title:
            notebook["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {notebook_title}"],
                }
            )

        # Process each section
        for section in writer_output:
            # Process each cell in the section
            for cell in section.cells:
                if cell.cell_type == "markdown":
                    notebook["cells"].append(
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": cell.content.split("\n"),
                        }
                    )
                elif cell.cell_type == "code":
                    notebook["cells"].append(
                        {
                            "cell_type": "code",
                            "metadata": {"execution_count": None, "outputs": []},
                            "source": cell.content.split("\n"),
                            "execution_count": None,
                            "outputs": [],
                        }
                    )
                else:
                    logger.warning(f"Unknown cell type: {cell.cell_type}")

        # Ensure .ipynb extension
        if not output_file.endswith(".ipynb"):
            output_file = output_file + ".ipynb"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Save the notebook
        with open(output_file, "w") as f:
            json.dump(notebook, f, indent=2)

        logger.info(f"Notebook saved successfully to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error converting WriterAgent output to notebook: {e}")
        return False


def writer_output_to_python_script(
    writer_output: List[NotebookSectionContent],
    output_file: str,
    include_markdown: bool = True,
) -> bool:
    """
    Convert WriterAgent output to a Python script (.py) file.

    This can be useful for users who want to run the code without using Jupyter.
    Markdown cells can be included as comments.

    Args:
        writer_output: List of NotebookSectionContent objects from WriterAgent
        output_file: Path to save the Python script
        include_markdown: Whether to include markdown cells as comments

    Returns:
        bool: True if the script was saved successfully, False otherwise
    """
    logger.info(f"Converting WriterAgent output to Python script")

    try:
        # Ensure .py extension
        if not output_file.endswith(".py"):
            output_file = output_file + ".py"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Convert to Python script
        script_content = []

        # Add a header
        script_content.append("#!/usr/bin/env python3")
        script_content.append('"""')
        script_content.append("This script was auto-generated from a notebook")
        script_content.append('"""')
        script_content.append("")

        # Process each section
        for section in writer_output:
            if include_markdown:
                script_content.append("#" * 80)
                script_content.append(f"# SECTION: {section.section_title}")
                script_content.append("#" * 80)
                script_content.append("")

            # Process each cell in the section
            for cell in section.cells:
                if cell.cell_type == "markdown" and include_markdown:
                    # Convert markdown to Python comments
                    for line in cell.content.split("\n"):
                        if line.strip():
                            script_content.append(f"# {line}")
                        else:
                            script_content.append("#")
                    script_content.append("")
                elif cell.cell_type == "code":
                    # Add code directly
                    script_content.append(cell.content)
                    script_content.append("")

        # Save the script
        with open(output_file, "w") as f:
            f.write("\n".join(script_content))

        logger.info(f"Python script saved successfully to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error converting WriterAgent output to Python script: {e}")
        return False


def writer_output_to_files(
    writer_output: List[NotebookSectionContent],
    output_dir: str,
    notebook_title: Optional[str] = None,
    formats: List[str] = ["ipynb", "py", "md"],
    original_content: Optional[List[NotebookSectionContent]] = None,
) -> Dict[str, str]:
    """
    Convert WriterAgent output to multiple file formats and save them.

    This is a convenience function that calls the appropriate conversion functions.
    If original_content is provided, it will save both the original and revised versions
    to allow for comparison.

    Args:
        writer_output: List of NotebookSectionContent objects from WriterAgent
        output_dir: Directory to save the output files
        notebook_title: Optional title for the notebook
        formats: List of formats to save as (supported: "ipynb", "py", "md")
        original_content: Optional original content before revisions for comparison

    Returns:
        Dict[str, str]: Dictionary mapping format to filepath
    """
    logger.info(f"Converting WriterAgent output to multiple formats: {formats}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate base filename from title or default
    base_filename = "notebook"
    if notebook_title:
        base_filename = notebook_title.lower().replace(" ", "_").replace("-", "_")

    results = {}

    # Convert to each requested format
    for fmt in formats:
        if fmt.lower() == "ipynb":
            output_file = os.path.join(output_dir, f"{base_filename}.ipynb")
            if writer_output_to_notebook(
                writer_output, output_file, notebook_title=notebook_title
            ):
                results["ipynb"] = output_file

                # If original content is provided, save it as well
                if original_content:
                    original_output_file = os.path.join(
                        output_dir, f"{base_filename}_original.ipynb"
                    )
                    if writer_output_to_notebook(
                        original_content,
                        original_output_file,
                        notebook_title=f"{notebook_title} (Original)",
                    ):
                        results["ipynb_original"] = original_output_file

        elif fmt.lower() == "py":
            output_file = os.path.join(output_dir, f"{base_filename}.py")
            if writer_output_to_python_script(writer_output, output_file):
                results["py"] = output_file

                # If original content is provided, save it as well
                if original_content:
                    original_output_file = os.path.join(
                        output_dir, f"{base_filename}_original.py"
                    )
                    if writer_output_to_python_script(
                        original_content, original_output_file
                    ):
                        results["py_original"] = original_output_file

        elif fmt.lower() == "md":
            output_file = os.path.join(output_dir, f"{base_filename}.md")
            markdown = writer_output_to_markdown(writer_output, output_file)
            if markdown:
                results["md"] = output_file

                # If original content is provided, save it as well
                if original_content:
                    original_output_file = os.path.join(
                        output_dir, f"{base_filename}_original.md"
                    )
                    original_markdown = writer_output_to_markdown(
                        original_content, original_output_file
                    )
                    if original_markdown:
                        results["md_original"] = original_output_file

        else:
            logger.warning(f"Unsupported format: {fmt}")

    return results


def notebook_to_writer_output(
    notebook_file: str, section_header_level: int = 2
) -> List[NotebookSectionContent]:
    """
    Convert a Jupyter notebook (.ipynb) file to WriterAgent output format.

    This function is useful for loading existing notebooks and converting them
    to the format expected by the WriterAgent for further processing.

    Args:
        notebook_file: Path to the Jupyter notebook file
        section_header_level: The markdown header level that defines sections (default: 2, meaning ## headers)

    Returns:
        List[NotebookSectionContent]: The notebook content in WriterAgent format
    """
    logger.info(
        f"Converting Jupyter notebook to WriterAgent output format: {notebook_file}"
    )

    try:
        # Load the notebook
        with open(notebook_file, "r") as f:
            notebook_data = json.load(f)

        # Process the notebook cells
        notebook_cells = notebook_data.get("cells", [])

        # Initialize variables
        current_section = None
        current_section_cells = []
        sections = []

        # Function to detect section headers in markdown cells
        def is_section_header(source, level):
            # Join the source lines and check for markdown header
            text = (
                "".join(source).strip() if isinstance(source, list) else source.strip()
            )
            header_pattern = f"^{'#' * level} "
            return bool(re.match(header_pattern, text))

        # Function to extract header text
        def extract_header_text(source, level):
            text = (
                "".join(source).strip() if isinstance(source, list) else source.strip()
            )
            header_pattern = f"^{'#' * level} (.*)"
            match = re.match(header_pattern, text)
            if match:
                return match.group(1).strip()
            return "Untitled Section"

        # Process each cell to find sections and their contents
        for cell in notebook_cells:
            cell_type = cell.get("cell_type")
            source = cell.get("source", "")

            # Check if this is a markdown cell that defines a new section
            if cell_type == "markdown" and is_section_header(
                source, section_header_level
            ):
                # If we already have a section, save it
                if current_section is not None:
                    sections.append(
                        NotebookSectionContent(
                            section_title=current_section, cells=current_section_cells
                        )
                    )

                # Start a new section
                current_section = extract_header_text(source, section_header_level)
                current_section_cells = []

                # Add this cell to the current section
                cell_content = "".join(source) if isinstance(source, list) else source
                current_section_cells.append(
                    NotebookCell(cell_type="markdown", content=cell_content)
                )
            else:
                # If we haven't encountered a section header yet, create a default section
                if current_section is None:
                    current_section = "Introduction"

                # Add this cell to the current section
                cell_content = "".join(source) if isinstance(source, list) else source
                current_section_cells.append(
                    NotebookCell(cell_type=cell_type, content=cell_content)
                )

        # Add the last section if it exists
        if current_section is not None and current_section_cells:
            sections.append(
                NotebookSectionContent(
                    section_title=current_section, cells=current_section_cells
                )
            )

        # If no sections were found, create a single section with all cells
        if not sections and notebook_cells:
            all_cells = []
            for cell in notebook_cells:
                cell_type = cell.get("cell_type")
                source = cell.get("source", "")
                cell_content = "".join(source) if isinstance(source, list) else source
                all_cells.append(
                    NotebookCell(cell_type=cell_type, content=cell_content)
                )

            sections.append(
                NotebookSectionContent(
                    section_title="Notebook Content", cells=all_cells
                )
            )

        logger.info(f"Converted notebook with {len(sections)} sections")
        return sections

    except Exception as e:
        logger.error(f"Error converting Jupyter notebook to WriterAgent output: {e}")
        # Return an empty list or raise an exception
        return []


def format_cells_for_evaluation(cells: List[NotebookCell]) -> str:
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


def save_notebook_versions(
    original_content: List[NotebookSectionContent],
    revised_content: List[NotebookSectionContent],
    critique: str,
    output_dir: str,
    notebook_title: Optional[str] = None,
    formats: List[str] = ["ipynb", "py", "md"],
) -> Dict[str, str]:
    """
    Save both original and revised versions of a notebook with the critique for comparison.

    This function wraps writer_output_to_files to save both versions of the notebook
    and also saves the critique as a separate markdown file.

    Args:
        original_content: Original NotebookSectionContent objects
        revised_content: Revised NotebookSectionContent objects after applying critique
        critique: The critique text that was used for revisions
        output_dir: Directory to save the output files
        notebook_title: Optional title for the notebook
        formats: List of formats to save as (supported: "ipynb", "py", "md")

    Returns:
        Dict[str, str]: Dictionary mapping format to filepath
    """
    logger.info(f"Saving original and revised notebook versions to {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate base filenames
    base_filename = "notebook"
    if notebook_title:
        base_filename = notebook_title.lower().replace(" ", "_").replace("-", "_")

    # Save both versions using writer_output_to_files
    results = writer_output_to_files(
        writer_output=revised_content,
        output_dir=output_dir,
        notebook_title=notebook_title,
        formats=formats,
        original_content=original_content,
    )

    # Save the critique as a separate markdown file
    critique_filename = f"{base_filename}_critique.md"
    critique_filepath = os.path.join(output_dir, critique_filename)

    try:
        with open(critique_filepath, "w") as f:
            # Add a title to the critique
            title = notebook_title or "Notebook"
            f.write(f"# Critique for {title}\n\n")
            f.write(critique)

        results["critique"] = critique_filepath
        logger.info(f"Saved critique to {critique_filepath}")
    except Exception as e:
        logger.error(f"Error saving critique to file: {e}")

    return results
