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


def get_interactive_clarifications(questions: List[str]) -> Dict[str, str]:
    """
    Interactive callback function to get answers to clarification questions.

    Args:
        questions: List of clarification questions

    Returns:
        Dictionary of questions and answers
    """
    print("\nI need some clarifications to better understand your requirements:")
    clarifications = {}

    print(f"Received {len(questions)} questions to clarify.")
    for question in questions:
        print(f"\n----------------------------------")
        print(f"Question: {question}")
        answer = input("Your answer: ").strip()
        clarifications[question] = answer

    return clarifications


def format_notebook_plan(plan: NotebookPlanModel) -> str:
    """
    Format the notebook plan as a markdown string.

    Args:
        plan: The notebook plan to format

    Returns:
        A markdown string representation of the plan
    """
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

    return result


def main():
    """Main entry point for the AI Agent application."""
    print("Welcome to the OpenAI Demo Notebook Generator!")
    print(
        "This tool will help you create Python notebooks showcasing OpenAI API capabilities."
    )

    # Initialize the user input handler
    input_handler = UserInputHandler()

    # Collect user requirements
    user_requirements = input_handler.collect_requirements()

    # Initialize the planner LLM
    planner = PlannerLLM()

    # Plan the notebook with interactive clarifications
    notebook_plan = planner.plan_notebook(
        user_requirements, clarification_callback=get_interactive_clarifications
    )

    # Format and display the notebook plan
    formatted_plan = format_notebook_plan(notebook_plan)
    print("\nNotebook Plan:")
    print(formatted_plan)

    # Save the plan to a file
    with open("notebook_plan.md", "w") as f:
        f.write(formatted_plan)

    print("\nNotebook plan saved to notebook_plan.md")

    # TODO: Pass the plan to the Writer LLM
    print(
        "\nNext steps would be to generate the notebook content using the Writer LLM."
    )


if __name__ == "__main__":
    main()
