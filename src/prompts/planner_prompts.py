"""
Prompt templates for the Planner LLM.

This module contains the prompt templates used by the Planner LLM to generate
clarification questions and notebook plans.
"""

PLANNING_SYSTEM_PROMPT = """
You are an AI assistant that helps plan educational notebooks about OpenAI APIs. 
Your task is to create a detailed outline for a Jupyter notebook based on the user's requirements.

The notebook should be well-structured with clear sections and subsections, each with descriptive titles and explanations.
Focus on creating a comprehensive plan that covers all aspects of the requested OpenAI API functionality.
"""

PLANNING_PROMPT = """
I need to create an educational Jupyter notebook about OpenAI APIs. Here are my requirements:

Description: {description}
Purpose: {purpose}
Target Audience: {target_audience}

{additional_requirements}

{code_snippets}

Please create a detailed outline for this Jupyter notebook, including sections, subsections, and descriptions of what each should cover.
"""

PLANNING_WITH_CLARIFICATION_PROMPT = """
I need to create an educational Jupyter notebook about OpenAI APIs. Here are my requirements:

Description: {description}
Purpose: {purpose}
Target Audience: {target_audience}

{additional_requirements}

{code_snippets}

Based on these requirements, I've provided the following clarifications:

{clarifications}

Please create a detailed outline for this Jupyter notebook, including sections, subsections, and descriptions of what each should cover.
"""

CLARIFICATION_SYSTEM_PROMPT = """
You are an AI assistant that helps plan educational notebooks about OpenAI APIs. 
Your task is to identify any missing or unclear information in the user's requirements and generate specific clarification questions.

Only ask questions if there is genuinely ambiguous or missing information that would significantly impact the notebook's structure or content.
If the requirements are clear enough to proceed with planning, don't ask any questions.
"""


def format_additional_requirements(requirements):
    """Format additional requirements for inclusion in prompts."""
    if not requirements:
        return ""

    formatted = "Additional Requirements:\n"
    for req in requirements:
        formatted += f"- {req}\n"

    return formatted


def format_code_snippets(snippets):
    """Format code snippets for inclusion in prompts."""
    if not snippets:
        return ""

    formatted = "Code Snippets to Include:\n"
    for i, snippet in enumerate(snippets, 1):
        formatted += f"Snippet {i}:\n```python\n{snippet}\n```\n\n"

    return formatted


def format_clarifications(clarifications):
    """Format clarifications for inclusion in prompts."""
    if not clarifications:
        return ""

    formatted = "Clarifications:\n"
    for question, answer in clarifications.items():
        formatted += f"Q: {question}\nA: {answer}\n\n"

    return formatted
