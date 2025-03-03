"""
Test script for PlannerLLM.
"""

import sys
import os

# Add the project root directory to the path so we can use absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.planner import PlannerLLM
from pprint import pprint


def callback(questions):
    """
    Interactive callback function that asks for input for each question individually.

    Args:
        questions: List of clarification questions

    Returns:
        Dictionary of questions and answers
    """
    print("\n=== CLARIFICATION CALLBACK TRIGGERED ===")
    print(f"Received {len(questions)} questions to clarify.")
    clarifications = {}

    for i, question in enumerate(questions, 1):
        print(f"\n----------------------------------")
        print(f"Question {i}/{len(questions)}: {question}")
        print(f"----------------------------------")

        # In test mode, you can uncomment this to use actual user input
        # answer = input("Your answer: ").strip()

        # For automated testing, use a predefined answer
        if i == 1:
            answer = "OpenAI Chat API, Vision API, and Assistants API"
        elif i == 2:
            answer = "Yes, include Python code examples"
        elif i == 3:
            answer = "Focus on authentication, basic calls, and error handling"
        else:
            answer = "This is a predefined answer for testing purposes"

        print(f"Answer: {answer}")  # Echo the answer for clarity
        clarifications[question] = answer

    print("\n=== CLARIFICATION ANSWERS COMPLETE ===")
    print(f"Returning {len(clarifications)} answers.")
    return clarifications


def main():
    """
    Test the PlannerLLM class.
    """
    print("Initializing PlannerLLM...")
    planner = PlannerLLM()

    print("Planning notebook...")
    try:
        print("Calling plan_notebook with clarification_callback...")
        result = planner.plan_notebook(
            {
                "notebook_description": "Guide to OpenAI APIs",
                "notebook_purpose": "Education",
                "target_audience": "Beginners",
                "additional_requirements": [
                    "Include code examples",
                    "Cover authentication",
                ],
            },
            callback,
        )

        print(f"\nGenerated notebook plan:")
        print(f"Title: {result.title}")
        print(f"Description: {result.description}")
        print(f"Target Audience: {result.target_audience}")

        print("\nSections:")
        for i, section in enumerate(result.sections, 1):
            print(f"\n{i}. {section.title}")
            print(f"   Description: {section.description}")

            if section.subsections:
                print("   Subsections:")
                for j, subsection in enumerate(section.subsections, 1):
                    print(f"   {i}.{j} {subsection.title}")
                    print(f"      Description: {subsection.description}")

                    if subsection.subsections:
                        print("      Sub-subsections:")
                        for k, subsubsection in enumerate(subsection.subsections, 1):
                            print(f"      {i}.{j}.{k} {subsubsection.title}")
                            print(f"         Description: {subsubsection.description}")
    except Exception as e:
        print(f"Error occurred during testing: {e}")


if __name__ == "__main__":
    main()
