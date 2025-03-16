"""
Script to convert multiple JSON files to markdown.

This script finds all JSON files in a specified directory and converts them to markdown
format, saving the output in the same or a different directory.
"""

import os
import sys
import json
from pathlib import Path
import argparse

# Add the parent directory to the path so we can import the src package
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.format import json_file_to_markdown, notebook_plan_to_markdown
from src.models import NotebookPlanModel


def convert_all_json_files(input_dir, output_dir=None, is_section=True):
    """
    Convert all JSON files in the input directory to markdown.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save markdown files (defaults to input_dir if None)
        is_section: Whether each JSON file contains a single section (True) or a list of sections (False)

    Returns:
        List of tuples containing (input_file, output_file) for each conversion
    """
    # If output_dir is not specified, use the input directory
    if output_dir is None:
        output_dir = input_dir

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Keep track of all conversions
    conversions = []

    # Find all JSON files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".json"):
            # Construct the full paths
            json_path = os.path.join(input_dir, filename)
            md_path = os.path.join(output_dir, filename.replace(".json", ".md"))

            try:
                # Convert the file
                print(f"Converting {json_path} to {md_path}...")
                json_file_to_markdown(json_path, md_path, is_section=is_section)
                conversions.append((json_path, md_path))
                print(f"✓ Successfully converted {filename}")
            except Exception as e:
                print(f"✗ Error converting {filename}: {e}")

    return conversions


def main():
    """
    Main function to parse arguments and run the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert multiple JSON files to markdown."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        default="output",
        help="Directory containing JSON files (default: output)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save markdown files (default: same as input directory)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_false",
        dest="is_section",
        help="Treat JSON files as containing lists of sections instead of single sections",
    )

    args = parser.parse_args()

    print(f"Looking for JSON files in {args.input_dir}...")
    conversions = convert_all_json_files(
        args.input_dir, args.output_dir, args.is_section
    )

    if conversions:
        print(f"\nSuccessfully converted {len(conversions)} files:")
        for json_path, md_path in conversions:
            print(f"  {os.path.basename(json_path)} → {os.path.basename(md_path)}")
    else:
        print(f"\nNo JSON files found in {args.input_dir}")


if __name__ == "__main__":
    main()
