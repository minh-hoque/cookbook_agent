"""
Test script for the Searcher module.

This module provides simple tests for the searcher.py module.
"""

import sys
import os
import unittest

# Add the project root directory to the path so we can use absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.searcher import format_search_results


class TestSearcher(unittest.TestCase):
    """Simple tests for the searcher functions."""

    def test_format_search_results_success(self):
        """Test formatting search results with valid data."""
        # Create a sample search result
        search_results = {
            "query": "OpenAI API tutorial",
            "results": [
                {
                    "title": "OpenAI API Documentation",
                    "url": "https://platform.openai.com/docs/api-reference",
                    "content": "The OpenAI API provides a simple interface for accessing AI models.",
                },
                {
                    "title": "Getting Started with OpenAI",
                    "url": "https://platform.openai.com/docs/quickstart",
                    "content": "This guide will help you start building with the OpenAI API.",
                },
            ],
            "summary": "OpenAI offers APIs for various AI models including GPT-4.",
            "metadata": {
                "result_count": 2,
                "search_depth": "advanced",
                "max_results": 5,
            },
        }

        # Format the results
        formatted_results = format_search_results(search_results)

        # Verify the formatting
        self.assertIsInstance(formatted_results, str)
        self.assertIn("### Search Summary:", formatted_results)
        self.assertIn("OpenAI offers APIs", formatted_results)
        self.assertIn("### Key Search Results:", formatted_results)
        self.assertIn("#### 1. OpenAI API Documentation", formatted_results)
        self.assertIn("#### 2. Getting Started with OpenAI", formatted_results)

    def test_format_search_results_error(self):
        """Test formatting search results with an error."""
        # Create a sample error result
        search_results = {
            "error": "Tavily API key not found",
            "results": [],
            "summary": "",
        }

        # Format the results
        formatted_results = format_search_results(search_results)

        # Verify the formatting
        self.assertIsInstance(formatted_results, str)
        self.assertIn("### Error:", formatted_results)
        self.assertIn("Tavily API key not found", formatted_results)

    def test_format_search_results_empty(self):
        """Test formatting empty search results."""
        # Create a sample empty result
        search_results = {
            "query": "NonexistentTopic12345",
            "results": [],
            "summary": "",
            "metadata": {
                "result_count": 0,
                "search_depth": "advanced",
                "max_results": 5,
            },
        }

        # Format the results
        formatted_results = format_search_results(search_results)

        # Verify the formatting
        self.assertIsInstance(formatted_results, str)
        self.assertIn("No search results found", formatted_results)


def run_interactive_test():
    """
    Run an interactive test of the searcher functions.
    This is useful for manual testing and debugging.
    """
    from src.searcher import search_with_taviley

    print("Running interactive test of searcher functions...")

    # Test search_topic function if API key is available
    try:
        print("\nSearching for 'OpenAI API tutorial'...")
        results = search_with_taviley("OpenAI API tutorial", max_results=3)

        print(f"\nSearch query: {results.get('query')}")
        print(
            f"Number of results: {results.get('metadata', {}).get('result_count', 0)}"
        )
        print(f"Summary: {results.get('summary')}")

        print("\nResults:")
        for i, result in enumerate(results.get("results", []), 1):
            print(f"\n{i}. {result.get('title')}")
            print(f"   URL: {result.get('url')}")
            print(f"   Content: {result.get('content')[:100]}...")

        # Test format_search_results function
        print("\nFormatted search results:")
        formatted = format_search_results(results)
        print(formatted)

    except Exception as e:
        print(f"Error during interactive test: {e}")


if __name__ == "__main__":
    # Run unit tests by default
    # unittest.main()

    # Uncomment to run interactive test instead
    run_interactive_test()
