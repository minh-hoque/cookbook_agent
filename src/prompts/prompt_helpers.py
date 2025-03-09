"""
Helper functions for prompt formatting.

This module contains helper functions used to format various elements in prompts.
"""

from typing import Dict, List, Any, Optional, Union


def format_subsections_details(subsections):
    """
    Format subsections details for the prompt.

    Args:
        subsections: List of subsections to format.

    Returns:
        str: Formatted subsections details.
    """
    if not subsections:
        return "No subsections specified"

    formatted = ""
    for i, subsection in enumerate(subsections, 1):
        formatted += f"### Subsection {i}: {subsection.title}\n"
        formatted += f"Description: {subsection.description}\n\n"

        if subsection.subsections:
            formatted += "Sub-subsections:\n"
            for j, subsubsection in enumerate(subsection.subsections, 1):
                formatted += (
                    f"- {j}. {subsubsection.title}: {subsubsection.description}\n"
                )
            formatted += "\n"

    return formatted.strip() or "No subsections specified"


def format_additional_requirements(requirements):
    """
    Format additional requirements for the prompt.

    Args:
        requirements: Additional requirements to format. Can be a list, dict, or string.

    Returns:
        str: Formatted additional requirements.
    """
    if not requirements:
        return "None specified"

    formatted = ""

    # Handle both list and dict inputs
    if isinstance(requirements, list):
        for req in requirements:
            formatted += f"- {req}\n"
    elif isinstance(requirements, dict):
        for key, value in requirements.items():
            if key not in [
                "description",
                "purpose",
                "target_audience",
                "code_snippets",
            ]:
                formatted += f"- {key}: {value}\n"
    else:
        formatted = f"- {str(requirements)}\n"

    return formatted.strip() or "None specified"


def format_previous_content(previous_content):
    """
    Format previously generated content for the prompt.

    Args:
        previous_content: Dictionary mapping section titles to content.

    Returns:
        str: Formatted previous content.
    """
    if not previous_content:
        return "No previously generated content available"

    formatted = "### Previously Generated Sections:\n\n"

    for section_title, content in previous_content.items():
        formatted += f"#### {section_title}\n"
        formatted += f"{content}\n\n"

    return formatted.strip() or "No previously generated content available"


def format_code_snippets(snippets):
    """
    Format code snippets for the prompt.

    Args:
        snippets: List of code snippets to format.

    Returns:
        str: Formatted code snippets.
    """
    if not snippets:
        return "None provided"

    formatted = ""
    for i, snippet in enumerate(snippets, 1):
        formatted += f"Snippet {i}:\n```python\n{snippet}\n```\n\n"

    return formatted.strip() or "None provided"


def format_clarifications(clarifications):
    """
    Format clarifications for the prompt.

    Args:
        clarifications: Dictionary mapping questions to answers.

    Returns:
        str: Formatted clarifications.
    """
    if not clarifications:
        return "No clarifications provided"

    formatted = ""
    for question, answer in clarifications.items():
        formatted += f"Q: {question}\nA: {answer}\n\n"

    return formatted.strip() or "No clarifications provided"
