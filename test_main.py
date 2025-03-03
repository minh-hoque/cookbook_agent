#!/usr/bin/env python3
"""
Test version of the OpenAI Demo Notebook Generator.
This file has hardcoded inputs for testing purposes.
"""

import sys
from typing import Dict, List, Any
from src.planner import PlannerLLM
from src.models import NotebookPlanModel
from src.tools.clarification_tools import get_clarifications
from src.tools.debug import info, set_level
import argparse


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


# Custom implementation of get_clarifications for testing
def test_get_clarifications(questions: List[str]) -> Dict[str, str]:
    """
    Automated test version of get_clarifications that returns pre-defined answers.

    Args:
        questions: List of clarification questions

    Returns:
        Dictionary of questions and pre-defined answers
    """
    print("\n=== TEST CLARIFICATIONS ===")
    print(f"Received {len(questions)} questions to clarify.")

    # Define pre-determined answers
    predefined_answers = {
        "What specific OpenAI models would you like to focus on?": "Focus on GPT-4o model",
        "Would you like to include error handling examples?": "Yes, basic error handling would be useful",
        "Are there any specific applications of predicted outputs you'd like to cover?": "Code refactoring and transformation examples",
        "What level of detail do you want for the code explanations?": "Medium detail - enough for ML engineers to understand the concepts",
        "Would you like to include examples of different prediction types?": "Yes, focus on content predictions",
    }

    # Build responses for questions
    clarifications = {}
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = predefined_answers.get(
            question, "Yes, please include this in the notebook"
        )
        print(f"Answer: {answer}")
        clarifications[question] = answer

    print("=== END TEST CLARIFICATIONS ===\n")
    return clarifications


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

    """Test entry point with hardcoded input values."""
    print("Welcome to the TEST OpenAI Demo Notebook Generator!")
    print("This test will run with predefined inputs.")

    # Hardcoded user requirements (no user input needed)
    user_requirements = {
        "notebook_description": "This notebook will help you learn how to use Predicted Outputs.",
        "notebook_purpose": "Tutorial",
        "target_audience": "Machine Learning Engineers",
        "additional_requirements": ["Include examples for predicted outputs"],
        "code_snippets": [
            """
from openai import OpenAI

code = \"\"\"
class User {
  firstName: string = "";
  lastName: string = "";
  username: string = "";
}

export default User;
\"\"\"

refactor_prompt = \"\"\"
Replace the "username" property with an "email" property. Respond only 
with code, and with no markdown formatting.
\"\"\"

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": refactor_prompt
        },
        {
            "role": "user",
            "content": code
        }
    ],
    prediction={
        "type": "content",
        "content": code
    }
)

print(completion)
print(completion.choices[0].message.content)
"""
        ],
    }

    # Initialize the planner LLM
    print("Initializing the planner...")
    planner = PlannerLLM()

    # Plan the notebook with our test clarifications function
    print("Planning the notebook...")
    notebook_plan = planner.plan_notebook(
        user_requirements, clarification_callback=get_clarifications
    )

    # Format and display the notebook plan
    formatted_plan = format_notebook_plan(notebook_plan)
    print("\nTest Notebook Plan:")
    print(formatted_plan)

    # Save the plan to a test file
    output_file = "test_notebook_plan.md"
    with open(output_file, "w") as f:
        f.write(formatted_plan)

    print(f"\nTest notebook plan saved to {output_file}")


if __name__ == "__main__":
    main()
