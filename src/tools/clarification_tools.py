"""
Clarification tools for the Cookbook Agent.

This module contains tools used for clarifying requirements and gathering more information from users.
"""

from typing import Dict, List
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


def get_clarifications_tool() -> ChatCompletionToolParam:
    """
    Returns the tool definition for getting clarifications from users.

    This tool allows the PlannerLLM to ask clarification questions when
    requirements are ambiguous or insufficient.

    Returns:
        A ChatCompletionToolParam for get_clarifications
    """
    return {
        "type": "function",
        "function": {
            "name": "get_clarifications",
            "description": "Get clarification answers for ambiguous requirements.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of clarification questions to ask the user",
                    }
                },
                "required": ["questions"],
            },
        },
    }
