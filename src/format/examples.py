"""
Examples demonstrating how to use the format_utils module.

This module contains examples of how to use the functions in the format_utils module
to convert WriterAgent output into markdown for easy review.
"""

from src.models import NotebookSectionContent, NotebookCell
from src.format.format_utils import writer_output_to_markdown


def create_sample_notebook_content() -> list[NotebookSectionContent]:
    """
    Create a sample notebook content for demonstration purposes.

    Returns:
        List of NotebookSectionContent objects
    """
    # Create a sample section with markdown and code cells
    section1 = NotebookSectionContent(
        section_title="Introduction to Data Analysis",
        cells=[
            NotebookCell(
                cell_type="markdown",
                content="# Introduction to Data Analysis\n\nIn this section, we'll explore the basics of data analysis using Python.",
            ),
            NotebookCell(
                cell_type="code",
                content="import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\n\n# Set up plotting\nplt.style.use('seaborn-whitegrid')",
            ),
            NotebookCell(
                cell_type="markdown",
                content="## Loading Data\n\nFirst, let's load some sample data to work with.",
            ),
            NotebookCell(
                cell_type="code",
                content="# Load sample data\ndf = pd.read_csv('sample_data.csv')\n\n# Display the first few rows\ndf.head()",
            ),
        ],
    )

    # Create another sample section
    section2 = NotebookSectionContent(
        section_title="Data Visualization",
        cells=[
            NotebookCell(
                cell_type="markdown",
                content="# Data Visualization\n\nNow let's create some visualizations to better understand our data.",
            ),
            NotebookCell(
                cell_type="code",
                content="# Create a histogram\nplt.figure(figsize=(10, 6))\ndf['value'].hist(bins=20)\nplt.title('Distribution of Values')\nplt.xlabel('Value')\nplt.ylabel('Frequency')\nplt.show()",
            ),
            NotebookCell(
                cell_type="markdown",
                content="## Scatter Plot\n\nLet's also create a scatter plot to see relationships between variables.",
            ),
            NotebookCell(
                cell_type="code",
                content="# Create a scatter plot\nplt.figure(figsize=(10, 6))\nplt.scatter(df['x'], df['y'], alpha=0.5)\nplt.title('Relationship between X and Y')\nplt.xlabel('X')\nplt.ylabel('Y')\nplt.show()",
            ),
        ],
    )

    return [section1, section2]


def example_writer_output_to_markdown():
    """
    Example of how to convert WriterAgent output to markdown.
    """
    # Create sample notebook content
    notebook_content = create_sample_notebook_content()

    # Convert to markdown
    markdown = writer_output_to_markdown(notebook_content)

    # Print the markdown
    print("=== Markdown Output ===")
    print(markdown)

    # Save to a file
    writer_output_to_markdown(notebook_content, "example_output.md")
    print("\nMarkdown has been saved to 'example_output.md'")


if __name__ == "__main__":
    example_writer_output_to_markdown()
