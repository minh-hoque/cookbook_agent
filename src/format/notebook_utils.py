"""
Utility functions for working with Jupyter notebooks and other file formats.

This module provides functions for converting between notebook formats and other file types,
including functions to save notebooks in various formats.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Union

from src.models import NotebookSectionContent, NotebookCell
from src.format.markdown_utils import (
    notebook_section_to_markdown,
    notebook_content_to_markdown,
    writer_output_to_markdown,
)

# Set up logger
logger = logging.getLogger(__name__)


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
        with open(markdown_filepath, "w") as f:
            f.write(markdown_content)

        logger.info(f"Saved complete notebook as markdown to {markdown_filepath}")
        return markdown_filepath
    except Exception as e:
        logger.error(f"Error saving markdown version: {e}")
        return None


def writer_output_to_notebook(
    writer_output: List[NotebookSectionContent],
    output_file: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    notebook_title: Optional[str] = None,
    return_notebook: bool = False,
) -> Union[bool, Dict[str, Any]]:
    """
    Convert WriterAgent output to Jupyter Notebook (.ipynb) format and save to a file.

    This function takes WriterAgent output (list of NotebookSectionContent objects),
    converts it to Jupyter Notebook format, and saves it to a file.

    Args:
        writer_output: List of NotebookSectionContent objects from WriterAgent
        output_file: Path to save the notebook output (should end with .ipynb). If None and return_notebook is True, no file will be saved.
        metadata: Optional metadata to include in the notebook
        notebook_title: Optional title for the notebook (used in markdown header)
        return_notebook: If True, return the notebook dictionary instead of a boolean success flag

    Returns:
        Union[bool, Dict[str, Any]]:
            - If return_notebook is False: True if the notebook was saved successfully, False otherwise
            - If return_notebook is True: The notebook dictionary
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
                # Split the content by newlines to create a list of lines
                # This ensures proper formatting in the notebook
                lines = cell.content.split("\n")

                # For empty lines, we need to ensure they're preserved as empty strings
                # rather than being converted to None or dropped
                source_lines = []
                for i, line in enumerate(lines):
                    if i < len(lines) - 1:
                        source_lines.append(line + "\n")
                    else:
                        # Don't add newline to the last line
                        source_lines.append(line)

                if cell.cell_type == "markdown":
                    # Fix section header formatting - ensure no extra newlines after hashtags
                    fixed_source_lines = []
                    for line in source_lines:
                        # If line starts with hashtags (MD header) followed by a newline
                        # Replace it with a properly formatted header
                        line = re.sub(r"^(#+)\s*\n", r"\1 ", line)
                        fixed_source_lines.append(line)

                    notebook["cells"].append(
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": fixed_source_lines,
                        }
                    )
                elif cell.cell_type == "code":
                    notebook["cells"].append(
                        {
                            "cell_type": "code",
                            "metadata": {"execution_count": None, "outputs": []},
                            "source": source_lines,
                            "execution_count": None,
                            "outputs": [],
                        }
                    )
                else:
                    logger.warning(f"Unknown cell type: {cell.cell_type}")

        # If we only want the notebook dictionary, return it now
        if return_notebook:
            return notebook

        # If output_file is None and we're not returning the notebook, that's an error
        if output_file is None:
            logger.error("No output file specified and return_notebook is False")
            return False

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
        if return_notebook:
            # If we're supposed to return the notebook but encountered an error,
            # return an empty notebook structure
            return {
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 4,
                "cells": [],
            }
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
            output_file = os.path.join(output_dir, f"{base_filename}_final.ipynb")
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
            output_file = os.path.join(output_dir, f"{base_filename}_final.py")
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
            output_file = os.path.join(output_dir, f"{base_filename}_final.md")
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

        # Process the notebook data using the shared function
        return notebook_dict_to_writer_output(notebook_data, section_header_level)

    except Exception as e:
        logger.error(f"Error converting Jupyter notebook to WriterAgent output: {e}")
        # Return an empty list or raise an exception
        return []


def notebook_dict_to_writer_output(
    notebook_data: Dict[str, Any], section_header_level: int = 2
) -> List[NotebookSectionContent]:
    """
    Convert a Jupyter notebook dictionary to WriterAgent output format.

    This function is useful for processing in-memory notebook data and converting it
    to the format expected by the WriterAgent without needing to read from a file.

    Args:
        notebook_data: Dictionary containing the notebook data
        section_header_level: The markdown header level that defines sections (default: 2, meaning ## headers)

    Returns:
        List[NotebookSectionContent]: The notebook content in WriterAgent format
    """
    logger.info(f"Converting notebook dictionary to WriterAgent output format")

    try:
        # Process the notebook cells
        notebook_cells = notebook_data.get("cells", [])

        # Initialize variables
        current_section = None
        current_section_cells = []
        sections = []

        # Function to process cell source to ensure consistent formatting
        def process_cell_source(source):
            if isinstance(source, list):
                # Join the lines with proper line breaks
                return "".join(source)
            return source

        # Function to detect section headers in markdown cells
        def is_section_header(source, level):
            # Process the source to ensure consistent format
            text = process_cell_source(source).strip()
            header_pattern = f"^{'#' * level} "
            return bool(re.match(header_pattern, text))

        # Function to extract header text
        def extract_header_text(source, level):
            text = process_cell_source(source).strip()
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
                cell_content = process_cell_source(source)
                current_section_cells.append(
                    NotebookCell(cell_type="markdown", content=cell_content)
                )
            else:
                # If we haven't encountered a section header yet, create a default section
                if current_section is None:
                    current_section = "Introduction"

                # Add this cell to the current section
                cell_content = process_cell_source(source)
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
                cell_content = process_cell_source(source)
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
        logger.error(f"Error converting notebook dictionary to WriterAgent output: {e}")
        # Return an empty list or raise an exception
        return []


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
