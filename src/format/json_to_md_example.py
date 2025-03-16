"""
Example script demonstrating how to convert JSON files to markdown.

This script shows how to use the json_file_to_markdown function to convert
JSON files containing notebook content to markdown format.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.format import json_file_to_markdown


def convert_json_file_to_markdown(json_file_path, output_file=None):
    """
    Convert a JSON file to markdown and print the result.

    Args:
        json_file_path: Path to the JSON file
        output_file: Optional path to save the markdown output
    """
    print(f"Converting {json_file_path} to markdown...")

    # If output_file is not provided, create one based on the input file name
    if output_file is None:
        output_file = os.path.splitext(json_file_path)[0] + ".md"

    # Convert the JSON file to markdown and save it
    markdown = json_file_to_markdown(json_file_path, output_file, is_section=True)

    print(f"Markdown saved to {output_file}")
    print("\nPreview of the markdown content:")
    print("=" * 50)
    # Print the first 500 characters of the markdown as a preview
    print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
    print("=" * 50)


def main():
    """
    Main function to run the example.
    """
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_json_file_to_markdown(json_file_path, output_file)
    else:
        # Use the example file from the output directory
        example_file = (
            "output/section_1_Using_Predicted_Outputs_Step-by-Step_Guide.json"
        )
        if os.path.exists(example_file):
            convert_json_file_to_markdown(example_file)
        else:
            print(f"Example file {example_file} not found.")
            print("Please provide a JSON file path as a command-line argument:")
            print(
                "python -m src.format.json_to_md_example <json_file_path> [output_file_path]"
            )


if __name__ == "__main__":
    main()
