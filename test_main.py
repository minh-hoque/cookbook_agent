#!/usr/bin/env python3
"""
Test version of the OpenAI Demo Notebook Generator.
This file has hardcoded inputs for testing purposes.
"""

import sys
import os
import logging
from typing import Dict, List, Any
from src.planner import PlannerLLM
from src.models import NotebookPlanModel
from src.tools.clarification_tools import get_clarifications
from src.tools.debug import DebugLevel, setup_logging

# Get a logger for this module
logger = logging.getLogger(__name__)


def format_notebook_plan(plan: NotebookPlanModel) -> str:
    """
    Format the notebook plan as a markdown string.

    Args:
        plan: The notebook plan to format

    Returns:
        A markdown string representation of the plan
    """
    logger.debug("Formatting notebook plan")
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

    logger.debug("Notebook plan formatting complete")
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
    logger.info(f"Processing {len(questions)} test clarification questions")
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
        logger.debug(f"Test clarification - Q: {question} A: {answer}")

    print("=== END TEST CLARIFICATIONS ===\n")
    logger.info("Test clarifications completed")
    return clarifications


def main():
    """Test entry point with hardcoded input values."""
    # Set up logging
    setup_logging(level=DebugLevel.INFO)

    logger.info("Starting TEST OpenAI Demo Notebook Generator")
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

    # Search for information about the notebook topic
    logger.info("Searching for information about the notebook topic")
    formatted_search_results = None
    notebook_description = user_requirements.get("notebook_description", "")

    if notebook_description:
        try:
            from src.searcher import search_topic, format_search_results

            print(f"\nSearching for information about: {notebook_description}")
            logger.debug(f"Searching for topic: {notebook_description}")
            search_results = search_topic(notebook_description)

            if "error" in search_results:
                logger.warning(f"Search error: {search_results['error']}")
                print(f"Search warning: {search_results['error']}")
            else:
                result_count = len(search_results.get("results", []))
                formatted_search_results = format_search_results(search_results)
                logger.info(f"Found {result_count} search results")
                print(f"Found {result_count} search results")

                # Log a preview of the search results
                if formatted_search_results:
                    preview = (
                        formatted_search_results[:200] + "..."
                        if len(formatted_search_results) > 200
                        else formatted_search_results
                    )
                    logger.debug(f"Search results preview: {preview}")
        except ImportError as e:
            logger.error(f"Failed to import search modules: {str(e)}")
            print(f"Search functionality not available: {str(e)}")
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            print(f"Error during search: {str(e)}")

    # Initialize the planner LLM
    logger.info("Initializing the planner...")
    print("Initializing the planner...")
    planner = PlannerLLM()

    # Plan the notebook with our test clarifications function
    logger.info("Planning the notebook...")
    print("Planning the notebook...")
    notebook_plan = planner.plan_notebook(
        user_requirements,
        clarification_callback=test_get_clarifications,
        search_results=formatted_search_results,
    )

    # Format and display the notebook plan
    logger.info("Formatting notebook plan")
    formatted_plan = format_notebook_plan(notebook_plan)
    print("\nTest Notebook Plan:")
    print(formatted_plan)

    # Save the plan to a test file
    output_file = "test_notebook_plan.md"
    logger.info(f"Saving notebook plan to {output_file}")
    with open(output_file, "w") as f:
        f.write(formatted_plan)

    logger.info(f"Test notebook plan saved to {output_file}")
    print(f"\nTest notebook plan saved to {output_file}")


if __name__ == "__main__":
    main()
