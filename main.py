#!/usr/bin/env python3
"""
AI Agent for Generating OpenAI Demo Notebooks
Main entry point for the application.
"""

import argparse
import sys
import os
import logging
import json
from datetime import datetime
from typing import Dict, Optional

# Import core modules
from src.user_input import UserInputHandler
from src.planner import PlannerLLM
from src.models import NotebookPlanModel
from src.writer import WriterAgent
from src.tools.clarification_tools import get_clarifications
from src.tools.debug import DebugLevel

# Import utilities
from src.utils import configure_logging, test_logging
from src.format import (
    format_notebook_plan,
    save_plan_to_file,
    save_notebook_content,
    writer_output_to_notebook,
)

# Get a logger for this module
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenAI Demo Notebook Generator")
    parser.add_argument(
        "--debug",
        choices=["ERROR", "INFO", "DEBUG"],
        help="Set debug level (ERROR, INFO, DEBUG)",
        default="INFO",
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file. If not specified, logs will only go to console.",
        default=None,
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Create a new log file instead of appending to existing one",
        default=False,
    )
    parser.add_argument(
        "--test-logging",
        action="store_true",
        help="Run a test of the logging system and exit",
        default=False,
    )
    parser.add_argument(
        "--model",
        help="OpenAI model to use for content generation",
        default="gpt-4o",
    )
    return parser.parse_args()


def setup_logging_from_args(args):
    """Set up logging based on command line arguments."""
    try:
        # Convert string to DebugLevel enum
        debug_level = DebugLevel[args.debug]

        # Handle auto log file path
        log_file_path = args.log_file
        if log_file_path == "auto":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = os.path.join(os.getcwd(), "logs")
            log_file_path = os.path.join(log_dir, f"notebook_gen_{timestamp}.log")

        # Initialize logging
        configure_logging(
            debug_level=debug_level, log_file=log_file_path, append=not args.no_append
        )

        # Test logging if requested
        if args.test_logging:
            logger.info("Running logging test...")
            test_logging()
            logger.info("Logging test complete")
            return False

        return True
    except KeyError:
        # Fall back to INFO level if invalid level provided
        configure_logging(debug_level=DebugLevel.INFO)
        logger.info("Invalid debug level specified, defaulting to INFO")
        return True


def search_for_topic_info(notebook_description: str) -> Optional[str]:
    """
    Search for information about the notebook topic using OpenAI's web search capability.

    Args:
        notebook_description: Description of the notebook topic

    Returns:
        Formatted search results or None if search failed
    """
    logger.info("Searching for information about the notebook topic")

    if not notebook_description:
        logger.info("No notebook description provided for search")
        print(
            "No notebook description provided for search. Continuing without search results."
        )
        return None

    try:
        from src.searcher import search_with_openai, format_openai_search_results

        print(f"\nSearching for information about: {notebook_description}")
        search_results = search_with_openai(notebook_description)

        if "error" in search_results and search_results["error"]:
            logger.warning(f"Search error: {search_results['error']}")
            print(f"Search warning: {search_results['error']}")
            return None

        formatted_results = format_openai_search_results(search_results)
        print("Search completed successfully")

        # Log a preview of the search results
        logger.debug(f"Search results preview: {formatted_results}")

        return formatted_results
    except ImportError:
        logger.warning("OpenAI package not available. Install with: pip install openai")
        print("Search functionality not available. Continuing without search results.")
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        print(f"Error searching for information: {str(e)}")

    return None


def create_notebook_plan(
    user_requirements: Dict, search_results: Optional[str]
) -> Optional[NotebookPlanModel]:
    """
    Create a notebook plan based on user requirements and search results.

    Args:
        user_requirements: Dictionary of user requirements
        search_results: Optional formatted search results

    Returns:
        Notebook plan model or None if planning failed
    """
    logger.info("Planning notebook with clarifications")
    try:
        # Initialize the planner LLM
        logger.debug("Initializing PlannerLLM")
        planner = PlannerLLM()

        # Plan the notebook with interactive clarifications
        notebook_plan = planner.plan_notebook(
            user_requirements,
            clarification_callback=get_clarifications,
            search_results=search_results,
        )
        logger.debug("Notebook plan created successfully")
        return notebook_plan
    except Exception as e:
        logger.error(f"Error planning notebook: {str(e)}")
        print(f"An error occurred while planning the notebook: {str(e)}")
        return None


def generate_notebook_content(
    notebook_plan: NotebookPlanModel,
    additional_requirements: list,
    model: str,
    output_dir: str,
) -> bool:
    """
    Generate notebook content based on the plan.

    Args:
        notebook_plan: The notebook plan model
        additional_requirements: List of additional requirements
        model: Model to use for generation
        output_dir: Directory to save output to

    Returns:
        True if content generation was successful, False otherwise
    """
    logger.info(f"Initializing WriterAgent with model: {model}")
    try:
        # Initialize the writer agent
        writer = WriterAgent(
            model=model, max_retries=3, search_enabled=True, final_critique_enabled=True
        )
        logger.debug("WriterAgent initialized successfully")

        # Generate the notebook content
        print("\nGenerating notebook content based on the plan...")
        logger.info("Generating notebook content")

        # Generate the content
        original_content, final_critique, revised_content = writer.generate_content(
            notebook_plan=notebook_plan,
            additional_requirements=additional_requirements,
        )

        # Save the generated content in various formats

        # 1. Save as JSON files and markdown
        content_list = [content.model_dump() for content in revised_content]
        markdown_filepath = save_notebook_content(content_list, output_dir)

        # 2. Save as Jupyter notebook (.ipynb)
        notebook_title = notebook_plan.title
        notebook_filepath = os.path.join(
            output_dir, f"{notebook_title.replace(' ', '_')}.ipynb"
        )
        jupyter_success = writer_output_to_notebook(
            revised_content, notebook_filepath, notebook_title=notebook_title  # type: ignore
        )

        if markdown_filepath:
            logger.info(
                f"Notebook content generated successfully and saved to {output_dir} directory"
            )
            print(
                f"\nNotebook content generated successfully and saved to {output_dir} directory"
            )
            print(f"Markdown file: {markdown_filepath}")

            if jupyter_success:
                print(f"Jupyter Notebook: {notebook_filepath}")

            return True
        else:
            logger.warning("Failed to save markdown content")
            return jupyter_success  # Return True if at least Jupyter was successful
    except ImportError:
        logger.error("WriterAgent or required dependencies not available")
        print(
            "Error: WriterAgent or required dependencies are not available. Please check your installation."
        )
    except Exception as e:
        logger.error(f"Error generating notebook content: {str(e)}")
        print(f"Error generating notebook content: {str(e)}")

    return False


def main():
    """Main entry point for the AI Agent application."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    if not setup_logging_from_args(args):
        sys.exit(0)  # Exit if just testing logging

    logger.info("Starting OpenAI Demo Notebook Generator")

    # Print welcome message
    print("Welcome to the OpenAI Demo Notebook Generator!")
    print(
        "This tool will help you create Python notebooks showcasing OpenAI API capabilities."
    )

    # Initialize the user input handler and collect requirements
    logger.debug("Initializing UserInputHandler")
    input_handler = UserInputHandler()

    logger.info("Collecting user requirements")
    user_requirements = input_handler.collect_requirements()
    logger.debug(f"User requirements collected: {user_requirements}")

    # Search for information about the notebook topic
    notebook_description = user_requirements.get("notebook_description", "")
    search_results = search_for_topic_info(notebook_description)

    # Create notebook plan
    notebook_plan = create_notebook_plan(user_requirements, search_results)
    if not notebook_plan:
        sys.exit(1)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")

    # Format and display the notebook plan
    print("\nNotebook Plan:")
    print(format_notebook_plan(notebook_plan))

    # Save the plan to a file
    notebook_title = notebook_plan.title
    output_file = os.path.join(output_dir, f"{notebook_title}.md")
    try:
        save_plan_to_file(notebook_plan, output_file)
        print(f"\nNotebook plan saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving notebook plan: {str(e)}")
        print(f"Warning: Could not save notebook plan to file: {str(e)}")

    # Ask user if they want to proceed with generating the notebook
    user_input = input("\nDo you want to generate the notebook content now? (y/n): ")
    if user_input.lower() != "y":
        print(
            "\nNotebook generation skipped. You can run the tool again later to generate content."
        )
        logger.info("User chose to skip notebook content generation")
        sys.exit(0)

    # Generate notebook content
    additional_requirements = user_requirements.get("additional_requirements", [])
    generate_notebook_content(
        notebook_plan=notebook_plan,
        additional_requirements=additional_requirements,
        model=args.model,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
