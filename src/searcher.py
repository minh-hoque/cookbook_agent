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

# Import the web search prompt
from src.prompts.writer_prompts import WEB_SEARCH_PROMPT


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


def get_openai_api_key() -> Optional[str]:
    """
    Get the OpenAI API key from environment variables.

    Returns:
        Optional[str]: The OpenAI API key if available, None otherwise.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        return None
    return api_key


def search_with_openai(
    topic: str,
    model: str = "gpt-4o",
    search_context_size: str = "high",
    client=None,
    tool_choice: str = "required",
) -> Dict[str, Any]:
    """
    Search for information about a topic using OpenAI's web search capability.

    Args:
        topic (str): The topic to search for.
        model (str, optional): The OpenAI model to use. Defaults to "gpt-4o".
        search_context_size (str, optional): The search context size. Defaults to "high".
        client (OpenAI, optional): An existing OpenAI client. If not provided, a new one will be created.
        tool_choice (str, optional): Controls which (if any) tool is called by the model.
            Options:
            - "none": The model will not call any tool and instead generates a message.
            - "auto": The model can pick between generating a message or calling tools.
            - "required": The model must call one or more tools.
            Defaults to "auto".

    Returns:
        Dict[str, Any]: A dictionary containing the search query and results.
            If the search fails, returns a dictionary with an error message.
    """
    try:
        # Import OpenAI
        try:
            from openai import OpenAI
        except ImportError:
            return {
                "error": "openai package not installed. Please install it with 'pip install openai'.",
                "results": [],
                "text": "",
            }

        # Use provided client or create a new one
        if client is None:
            api_key = get_openai_api_key()
            if not api_key:
                return {"error": "OpenAI API key not found", "results": [], "text": ""}

            # Initialize the OpenAI client
            client = OpenAI(api_key=api_key)

        # Log the search query
        logger.info(f"Searching for information about: {topic} using OpenAI")

        # Validate tool_choice
        valid_tool_choices = ["none", "auto", "required"]
        if tool_choice not in valid_tool_choices:
            logger.warning(f"Invalid tool_choice: {tool_choice}. Using default: 'auto'")
            tool_choice = "auto"

        response = client.responses.create(
            model=model,
            instructions=WEB_SEARCH_PROMPT,
            tools=[{"type": "web_search_preview", "search_context_size": search_context_size}],  # type: ignore
            input=topic,
            tool_choice=tool_choice,  # type: ignore  # Add tool_choice parameter
        )

        # Get the result
        result_text = response.output_text

        return {
            "query": topic,
            "text": result_text,
        }

    except Exception as e:
        # Log the error and return an error result
        logger.error(f"Error searching for topic '{topic}' with OpenAI: {str(e)}")
        return {
            "error": str(e),
            "query": topic,
            "text": f"Error searching for information: {str(e)}",
        }


def format_openai_search_results(search_results: Dict[str, Any]) -> str:
    """
    Format OpenAI search results into a string that can be included in the planning prompt.

    Args:
        search_results (Dict[str, Any]): The search results from the search_with_openai function.

    Returns:
        str: A formatted string containing the search results.
    """
    if not search_results:
        return "No search results found"

    if "error" in search_results and search_results["error"]:
        return f"### Error: {search_results['error']}"

    # Check if text is available
    if "text" not in search_results or not search_results["text"]:
        return "No search results found"

    # Add query
    formatted_results = (
        f"### Query:\n{search_results.get('query', 'No query available')}\n\n"
    )

    # Format the text results
    formatted_results += f"### Search Results:\n\n{search_results['text']}\n\n"

    return formatted_results
