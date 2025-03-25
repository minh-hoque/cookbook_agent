#!/usr/bin/env python3
"""
Example script to demonstrate the notebook conversion functions.

This example shows how to convert the WriterAgent's output to various formats:
1. Jupyter Notebook (.ipynb)
2. Python script (.py)
3. Markdown (.md)
"""

import os
import logging
import sys
from typing import List

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import NotebookSectionContent, NotebookCell
from src.format import (
    writer_output_to_notebook,
    writer_output_to_python_script,
    writer_output_to_markdown,
    writer_output_to_files,
    notebook_to_writer_output,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_notebook_content() -> List[NotebookSectionContent]:
    """
    Create a sample notebook content for demonstration.

    Returns:
        List[NotebookSectionContent]: Sample notebook content
    """
    # Section 1: Introduction
    intro_section = NotebookSectionContent(
        section_title="Introduction",
        cells=[
            NotebookCell(
                cell_type="markdown",
                content="# Introduction to OpenAI API\n\nThis notebook demonstrates how to use the OpenAI API for text generation and other tasks.",
            ),
            NotebookCell(
                cell_type="code",
                content="import openai\nimport os\n\n# Set API key\nopenai.api_key = os.environ.get('OPENAI_API_KEY')",
            ),
            NotebookCell(
                cell_type="markdown",
                content="## Key Concepts\n\n- API Keys\n- Models\n- Tokens\n- Completions",
            ),
        ],
    )

    # Section 2: Text Generation
    text_gen_section = NotebookSectionContent(
        section_title="Text Generation",
        cells=[
            NotebookCell(
                cell_type="markdown",
                content="# Text Generation with GPT Models\n\nIn this section, we'll explore text generation capabilities.",
            ),
            NotebookCell(
                cell_type="code",
                content='from openai import OpenAI\n\nclient = OpenAI()\n\nresponse = client.chat.completions.create(\n    model="gpt-3.5-turbo",\n    messages=[\n        {"role": "system", "content": "You are a helpful assistant."},\n        {"role": "user", "content": "Tell me about quantum computing"}\n    ]\n)\n\nprint(response.choices[0].message.content)',
            ),
            NotebookCell(
                cell_type="markdown",
                content="## Parameters\n\n- `temperature`: Controls randomness\n- `max_tokens`: Controls length\n- `top_p`: Controls diversity",
            ),
            NotebookCell(
                cell_type="code",
                content='# Example with different parameters\nresponse = client.chat.completions.create(\n    model="gpt-3.5-turbo",\n    messages=[\n        {"role": "system", "content": "You are a helpful assistant."},\n        {"role": "user", "content": "Write a short poem about AI"}\n    ],\n    temperature=0.7,\n    max_tokens=100,\n    top_p=0.9\n)\n\nprint(response.choices[0].message.content)',
            ),
        ],
    )

    return [intro_section, text_gen_section]


def main():
    """Main function to demonstrate notebook conversion functions."""
    # Create sample content
    logger.info("Creating sample notebook content")
    notebook_content = create_sample_notebook_content()

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")

    # Example 1: Convert to Jupyter Notebook (.ipynb)
    ipynb_file = os.path.join(output_dir, "openai_api_example.ipynb")
    if writer_output_to_notebook(
        notebook_content, ipynb_file, notebook_title="OpenAI API Tutorial"
    ):
        logger.info(f"Jupyter Notebook saved to: {ipynb_file}")

    # Example 2: Convert to Python script (.py)
    py_file = os.path.join(output_dir, "openai_api_example.py")
    if writer_output_to_python_script(notebook_content, py_file, include_markdown=True):
        logger.info(f"Python script saved to: {py_file}")

    # Example 3: Convert to Markdown (.md)
    md_file = os.path.join(output_dir, "openai_api_example.md")
    md_content = writer_output_to_markdown(notebook_content, md_file)
    if md_content:
        logger.info(f"Markdown file saved to: {md_file}")

    # Example 4: Convert to all formats at once
    results = writer_output_to_files(
        notebook_content,
        output_dir=output_dir,
        notebook_title="OpenAI API Tutorial - All Formats",
        formats=["ipynb", "py", "md"],
    )

    for fmt, filepath in results.items():
        logger.info(f"{fmt.upper()} file saved to: {filepath}")

    # Example 5: Load a notebook and convert back to WriterAgent format
    # Note: This assumes the notebook was already created in Example 1
    if os.path.exists(ipynb_file):
        logger.info(f"Loading notebook from: {ipynb_file}")
        loaded_content = notebook_to_writer_output(ipynb_file)

        # Display information about the loaded content
        logger.info(f"Loaded {len(loaded_content)} sections from notebook")
        for i, section in enumerate(loaded_content):
            logger.info(
                f"  Section {i+1}: {section.section_title} ({len(section.cells)} cells)"
            )

        # Save the loaded content to a different file to verify it works
        converted_file = os.path.join(output_dir, "converted_notebook.ipynb")
        if writer_output_to_notebook(
            loaded_content, converted_file, notebook_title="Converted Notebook"
        ):
            logger.info(f"Converted notebook saved to: {converted_file}")
    else:
        logger.warning(f"Could not find notebook file to load: {ipynb_file}")


if __name__ == "__main__":
    main()
