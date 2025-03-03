#!/usr/bin/env python3
"""
AI Agent for Generating OpenAI Demo Notebooks
Main entry point for the application.
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any
from src.user_input import UserInputHandler
from src.planner import PlannerLLM
from src.models import NotebookPlanModel
from src.tools.clarification_tools import get_clarifications
from src.tools.debug import DebugLevel, setup_logging

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

    # Plan the notebook with interactive clarifications
    logger.info("Planning notebook with clarifications")
    try:
        notebook_plan = planner.plan_notebook(
            user_requirements, clarification_callback=get_clarifications
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

    # Save the plan to a file
    output_file = "notebook_plan.md"
    logger.debug(f"Saving notebook plan to file: {output_file}")
    try:
        with open(output_file, "w") as f:
            f.write(formatted_plan)
        logger.info(f"Notebook plan saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving notebook plan: {str(e)}")
        print(f"Warning: Could not save notebook plan to file: {str(e)}")

    print(f"\nNotebook plan saved to {output_file}")

    # TODO: Pass the plan to the Writer LLM
    logger.info("Notebook planning complete")
    print(
        "\nNext steps would be to generate the notebook content using the Writer LLM."
    )


if __name__ == "__main__":
    main()
