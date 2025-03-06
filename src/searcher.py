"""
Searcher Module

This module is responsible for searching the internet for information about notebook topics
using the Tavily API. It provides functions to search for information and format the results
for use in the notebook planning process.
"""

import os
import logging
import json
from typing import Dict, Any, Optional

# Try to import tavily, but make it optional
try:
    from tavily import TavilyClient
except ImportError:
    # If tavily is not installed, just print a warning
    print(
        "Warning: tavily-python not installed. Please install it with 'pip install tavily-python'."
    )

# Get a logger for this module
logger = logging.getLogger(__name__)


# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except ImportError:
    # If dotenv is not installed, just print a warning
    print("Warning: python-dotenv not installed. Using environment variables directly.")


def get_tavily_api_key() -> Optional[str]:
    """
    Get the Tavily API key from environment variables.

    Returns:
        Optional[str]: The Tavily API key if available, None otherwise.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not found in environment variables")
        return None
    return api_key


def search_topic(topic: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search for information about a topic using the Tavily API.

    Args:
        topic (str): The topic to search for.
        max_results (int, optional): Maximum number of search results to return. Defaults to 5.

    Returns:
        Dict[str, Any]: A dictionary containing the search results and metadata.
            If the search fails, returns a dictionary with an error message.
    """
    api_key = get_tavily_api_key()
    if not api_key:
        return {"error": "Tavily API key not found", "results": [], "summary": ""}

    try:
        # Initialize the Tavily client
        client = TavilyClient(api_key=api_key)

        # Log the search query
        logger.info(f"Searching for information about: {topic}")

        # Perform the search
        response = client.search(
            query=topic,
            search_depth="advanced",  # Use advanced search for more comprehensive results
            max_results=max_results,
            include_answer="advanced",  # Use boolean instead of string # type: ignore
            include_domains=[],  # No domain restrictions
            exclude_domains=[],  # No domain exclusions
        )

        # Extract and format the results
        results = response.get("results", [])
        summary = response.get("answer", "")

        # Log the number of results found
        logger.info(f"Found {len(results)} search results for topic: {topic}")

        return {
            "query": topic,
            "results": results,
            "summary": summary,
            "metadata": {
                "result_count": len(results),
                "search_depth": "advanced",
                "max_results": max_results,
            },
        }

    except Exception as e:
        # Log the error and return an empty result
        logger.error(f"Error searching for topic '{topic}': {str(e)}")
        return {
            "error": str(e),
            "query": topic,
            "results": [],
            "summary": f"Error searching for information: {str(e)}",
        }


def format_search_results(search_results: Dict[str, Any]) -> str:
    """
    Format search results into a string that can be included in the planning prompt.

    Args:
        search_results (Dict[str, Any]): The search results from the search_topic function.

    Returns:
        str: A formatted string containing the search results.
    """
    if not search_results:
        return "No search results found"

    if "error" in search_results and search_results["error"]:
        return f"### Error: {search_results['error']}"

    # Check if there are no results
    if not search_results.get("results", []):
        return "No search results found"

    # Start with the summary
    formatted_results = f"### Search Summary:\n{search_results.get('summary', 'No summary available.')}\n\n"

    # Add individual search results (limited to top 3 for simplicity)
    formatted_results += "### Key Search Results:\n"

    for i, result in enumerate(search_results.get("results", [])[:3], 1):
        title = result.get("title", "No title")
        content = result.get("content", "No content")
        url = result.get("url", "No URL")

        formatted_results += f"#### {i}. {title}\n"
        formatted_results += f"Source: {url}\n"
        formatted_results += f"{content}\n\n"

    return formatted_results
