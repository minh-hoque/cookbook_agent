#!/usr/bin/env python3
"""
AI Agent for Generating OpenAI Demo Notebooks
Main entry point for the application.
"""

import argparse
import sys
import os
import logging
import json  # Add json import
from datetime import datetime
from typing import Dict, List, Any
from src.user_input import UserInputHandler
from src.planner import PlannerLLM
from src.models import NotebookPlanModel
from src.tools.clarification_tools import get_clarifications
from src.tools.debug import DebugLevel, setup_logging
from src.writer import WriterAgent  # Import the WriterAgent
from src.format import (
    notebook_content_to_markdown,
    save_markdown_to_file,
)  # Import format utilities

# Get a logger for this module
logger = logging.getLogger(__name__)


def test_logging():
    """Test function to verify that the logging system captures caller information correctly."""
    logger.debug("Test debug message from test_logging function")
    logger.info("Test info message from test_logging function")
    logger.error("Test error message from test_logging function")


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


def main():
    """Main entry point for the AI Agent application."""
    # Parse command line arguments first
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
        default="gpt-4.5-preview-2025-02-27",
    )
    args = parser.parse_args()

    # Set up logging based on command line arguments
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
        setup_logging(
            level=debug_level, log_file=log_file_path, append=not args.no_append
        )

        # Test logging if requested
        if args.test_logging:
            logger.info("Running logging test...")
            test_logging()
            logger.info("Logging test complete")
            sys.exit(0)

        # ---------------------------------------------------------------------
        # UPDATED CODE: More comprehensive reduction of external library logging
        # Reduce logging for external libraries to WARNING or higher:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

        # Silence other noisy libraries
        for logger_name in logging.root.manager.loggerDict:
            if any(
                name in logger_name.lower()
                for name in ["http", "urllib", "requests", "openai", "aiohttp"]
            ):
                logging.getLogger(logger_name).setLevel(logging.WARNING)
        # ---------------------------------------------------------------------

    except KeyError:
        # Fall back to INFO level if invalid level provided
        setup_logging(level=DebugLevel.INFO)
        logger.info("Invalid debug level specified, defaulting to INFO")

    logger.info("Starting OpenAI Demo Notebook Generator")

    print("Welcome to the OpenAI Demo Notebook Generator!")
    print(
        "This tool will help you create Python notebooks showcasing OpenAI API capabilities."
    )

    # Initialize the user input handler
    logger.debug("Initializing UserInputHandler")
    input_handler = UserInputHandler()

    # Collect user requirements
    logger.info("Collecting user requirements")
    user_requirements = input_handler.collect_requirements()
    logger.debug(f"User requirements collected: {user_requirements}")

    # Initialize the planner LLM
    logger.debug("Initializing PlannerLLM")
    planner = PlannerLLM()

    # Search for information about the notebook topic
    logger.info("Searching for information about the notebook topic")
    formatted_search_results = None
    notebook_description = user_requirements.get("notebook_description", "")

    if notebook_description:
        try:
            from src.searcher import search_topic, format_search_results

            print(f"\nSearching for information about: {notebook_description}")
            search_results = search_topic(notebook_description)

            if "error" in search_results:
                logger.warning(f"Search error: {search_results['error']}")
                print(f"Search warning: {search_results['error']}")
            else:
                result_count = len(search_results.get("results", []))
                formatted_search_results = format_search_results(search_results)
                print(f"Found {result_count} search results")

                # Log a preview of the search results
                if formatted_search_results:
                    preview = (
                        formatted_search_results[:200] + "..."
                        if len(formatted_search_results) > 200
                        else formatted_search_results
                    )
                    logger.debug(f"Search results preview: {preview}")

        except ImportError:
            logger.warning(
                "Tavily search module not available. Install with: pip install tavily-python"
            )
            print(
                "Search functionality not available. Continuing without search results."
            )
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            print(f"Error searching for information: {str(e)}")
    else:
        logger.info("No notebook description provided for search")
        print(
            "No notebook description provided for search. Continuing without search results."
        )

    # Plan the notebook with interactive clarifications
    logger.info("Planning notebook with clarifications")
    try:
        notebook_plan = planner.plan_notebook(
            user_requirements,
            clarification_callback=get_clarifications,
            search_results=formatted_search_results,
        )
        logger.debug("Notebook plan created successfully")
    except Exception as e:
        logger.error(f"Error planning notebook: {str(e)}")
        print(f"An error occurred while planning the notebook: {str(e)}")
        sys.exit(1)

    # Format and display the notebook plan
    formatted_plan = format_notebook_plan(notebook_plan)
    logger.info("Notebook plan formatted")
    print("\nNotebook Plan:")
    print(formatted_plan)

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    logger.debug(f"Created output directory: {output_dir}")

    # Save the plan to a file
    output_file = os.path.join(output_dir, "notebook_plan.md")
    logger.debug(f"Saving notebook plan to file: {output_file}")
    try:
        with open(output_file, "w") as f:
            f.write(formatted_plan)
        logger.info(f"Notebook plan saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving notebook plan: {str(e)}")
        print(f"Warning: Could not save notebook plan to file: {str(e)}")

    print(f"\nNotebook plan saved to {output_file}")

    # Initialize the writer agent
    logger.info(f"Initializing WriterAgent with model: {args.model}")
    try:
        writer = WriterAgent(model=args.model)
        logger.debug("WriterAgent initialized successfully")

        # Ask user if they want to proceed with generating the notebook
        user_input = input(
            "\nDo you want to generate the notebook content now? (y/n): "
        )
        if user_input.lower() != "y":
            print(
                "\nNotebook generation skipped. You can run the tool again later to generate content."
            )
            logger.info("User chose to skip notebook content generation")
            sys.exit(0)

        # Generate the notebook content
        print("\nGenerating notebook content based on the plan...")
        logger.info("Generating notebook content")

        # Use existing requirements directly from user input
        additional_requirements = user_requirements.get("additional_requirements", [])

        # Generate the content
        section_contents = writer.generate_content(
            notebook_plan=notebook_plan,
            additional_requirements=additional_requirements,
            max_retries=3,  # Reduce retries for faster generation
        )

        # Save the generated content to a file
        notebook_filename = f"{notebook_plan.title.replace(' ', '_').lower()}"

        try:
            # Save each section content as JSON
            for i, section_content in enumerate(section_contents):
                section_json_file = os.path.join(
                    output_dir,
                    f"section_{i+1}_{section_content.section_title.replace(' ', '_').replace(':', '')}.json",
                )
                with open(section_json_file, "w") as f:
                    json.dump(section_content.model_dump(), f, indent=2)
                logger.info(f"Saved section {i+1} to {section_json_file}")

            # Generate markdown from all sections
            markdown_content = notebook_content_to_markdown(section_contents)

            # Save the markdown to a file
            markdown_filepath = os.path.join(output_dir, f"{notebook_filename}.md")
            save_markdown_to_file(markdown_content, markdown_filepath)

            logger.info(
                f"Notebook content generated successfully and saved to {output_dir} directory"
            )
            print(
                f"\nNotebook content generated successfully and saved to {output_dir} directory"
            )
            print(f"Markdown file: {markdown_filepath}")
        except Exception as e:
            logger.error(f"Error generating notebook file: {str(e)}")
            print(f"Error generating notebook file: {str(e)}")

    except ImportError:
        logger.error("WriterAgent or required dependencies not available")
        print(
            "Error: WriterAgent or required dependencies are not available. Please check your installation."
        )
    except Exception as e:
        logger.error(f"Error initializing WriterAgent: {str(e)}")
        print(f"Error initializing WriterAgent: {str(e)}")


if __name__ == "__main__":
    main()
