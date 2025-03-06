"""
Test script for PlannerLLM.

This module provides simple tests for the PlannerLLM class.
"""

import sys
import os
import unittest

# Add the project root directory to the path so we can use absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.planner import PlannerLLM
from src.models import NotebookPlanModel


def simple_clarification_callback(questions):
    """
    Simple callback function that returns predefined answers for clarification questions.

    Args:
        questions: List of clarification questions

    Returns:
        Dictionary of questions and answers
    """
    print(f"\nReceived {len(questions)} clarification questions")
    answers = {}

    for question in questions:
        # Return a simple answer for all questions
        answers[question] = "Yes, please include that in the notebook."
        print(f"Q: {question}\nA: {answers[question]}")

    return answers


class TestPlannerFunctions(unittest.TestCase):
    """Simple tests for the planner functions."""

    def test_format_notebook_plan(self):
        """Test that a NotebookPlanModel can be created and formatted."""
        # Create a simple notebook plan
        plan = NotebookPlanModel(
            title="Test Notebook",
            description="A test notebook",
            purpose="Testing",
            target_audience="Testers",
            sections=[],
        )

        # Verify the plan has the expected attributes
        self.assertEqual(plan.title, "Test Notebook")
        self.assertEqual(plan.description, "A test notebook")
        self.assertEqual(plan.purpose, "Testing")
        self.assertEqual(plan.target_audience, "Testers")
        self.assertEqual(len(plan.sections), 0)


def run_interactive_test():
    """
    Run an interactive test of the PlannerLLM.
    This is useful for manual testing and debugging.
    """
    print("Running interactive test of PlannerLLM...")

    # Create a simple requirements dictionary
    requirements = {
        "notebook_description": "Guide to OpenAI APIs",
        "notebook_purpose": "Education",
        "target_audience": "Beginners",
        "additional_requirements": [
            "Include code examples",
            "Cover authentication",
        ],
        "code_snippets": [],
    }

    # Create a search results string
    search_results = """
    ### Search Summary:
    OpenAI offers several APIs including GPT-4, DALL-E, and Whisper. 
    Authentication is done via API keys.
    
    ### Key Search Results:
    #### 1. OpenAI API Documentation
    Source: https://platform.openai.com/docs/api-reference
    The OpenAI API provides a simple interface for accessing AI models.
    """

    try:
        # Initialize the planner
        planner = PlannerLLM()

        # Plan the notebook
        print("\nPlanning notebook...")
        plan = planner.plan_notebook(
            requirements=requirements,
            clarification_callback=simple_clarification_callback,
            search_results=search_results,
        )

        # Print the plan
        print("\nGenerated notebook plan:")
        print(f"Title: {plan.title}")
        print(f"Description: {plan.description}")
        print(f"Purpose: {plan.purpose}")
        print(f"Target Audience: {plan.target_audience}")

        print("\nSections:")
        for i, section in enumerate(plan.sections, 1):
            print(f"\n{i}. {section.title}")
            print(f"   Description: {section.description}")

            if section.subsections:
                for j, subsection in enumerate(section.subsections, 1):
                    print(f"   {i}.{j} {subsection.title}")
                    print(f"      Description: {subsection.description}")

    except Exception as e:
        print(f"Error during interactive test: {e}")


if __name__ == "__main__":
    # Run unit tests by default
    # unittest.main()

    # Uncomment to run interactive test instead
    run_interactive_test()
