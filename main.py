#!/usr/bin/env python3
"""
AI Agent for Generating OpenAI Demo Notebooks
Main entry point for the application.
"""

import argparse
import sys
from typing import Dict, List, Any
from src.user_input import UserInputHandler
from src.planner import PlannerLLM
from src.models import NotebookPlanModel
from src.tools.clarification_tools import get_clarifications
from src.tools.debug import info, debug, error, set_level, DebugLevel


def format_notebook_plan(plan: NotebookPlanModel) -> str:
    """
    Format the notebook plan as a markdown string.

    Args:
        plan: The notebook plan to format

    Returns:
        A markdown string representation of the plan
    """
    debug("Formatting notebook plan")
    result = f"# {plan.title}\n\n"
    result += f"**Description:** {plan.description}\n\n"
    result += f"**Purpose:** {plan.purpose}\n\n"
    result += f"**Target Audience:** {plan.target_audience}\n\n"
    result += "## Outline\n\n"

    for section in plan.sections:
        result += f"### {section.title}\n\n"
        result += f"{section.description}\n\n"

        if section.subsections:
            for subsection in section.subsections:
                result += f"#### {subsection.title}\n\n"
                result += f"{subsection.description}\n\n"

                if subsection.subsections:
                    for sub_subsection in subsection.subsections:
                        result += f"##### {sub_subsection.title}\n\n"
                        result += f"{sub_subsection.description}\n\n"

    debug("Notebook plan formatting complete")
    return result


def main():
    """Main entry point for the AI Agent application."""
    info("Starting OpenAI Demo Notebook Generator")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OpenAI Demo Notebook Generator")
    parser.add_argument(
        "--debug",
        choices=["OFF", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"],
        help="Set debug level",
        default="OFF",
    )
    args = parser.parse_args()

    # Set debug level if specified
    if args.debug:
        set_level(args.debug)
        info(f"Debug level set to {args.debug}")

    print("Welcome to the OpenAI Demo Notebook Generator!")
    print(
        "This tool will help you create Python notebooks showcasing OpenAI API capabilities."
    )

    # Initialize the user input handler
    debug("Initializing UserInputHandler")
    input_handler = UserInputHandler()

    # Collect user requirements
    info("Collecting user requirements")
    user_requirements = input_handler.collect_requirements()
    debug("User requirements collected", user_requirements)

    # Initialize the planner LLM
    debug("Initializing PlannerLLM")
    planner = PlannerLLM()

    # Plan the notebook with interactive clarifications
    info("Planning notebook with clarifications")
    try:
        notebook_plan = planner.plan_notebook(
            user_requirements, clarification_callback=get_clarifications
        )
        debug("Notebook plan created successfully")
    except Exception as e:
        error(f"Error planning notebook: {str(e)}")
        sys.exit(1)

    # Format and display the notebook plan
    formatted_plan = format_notebook_plan(notebook_plan)
    info("Notebook plan formatted")
    print("\nNotebook Plan:")
    print(formatted_plan)

    # Save the plan to a file
    debug("Saving notebook plan to file")
    try:
        with open("notebook_plan.md", "w") as f:
            f.write(formatted_plan)
        info("Notebook plan saved to notebook_plan.md")
    except Exception as e:
        error(f"Error saving notebook plan: {str(e)}")

    print("\nNotebook plan saved to notebook_plan.md")

    # TODO: Pass the plan to the Writer LLM
    info("Notebook planning complete")
    print(
        "\nNext steps would be to generate the notebook content using the Writer LLM."
    )


if __name__ == "__main__":
    main()
