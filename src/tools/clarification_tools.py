"""
Clarification tools for the Cookbook Agent.

This module contains tools used for clarifying requirements and gathering more information from users.
"""

import logging
from typing import Dict, List
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

# Get a logger for this module
logger = logging.getLogger(__name__)


def get_clarifications_tool() -> ChatCompletionToolParam:
    """
    Returns the tool definition for getting clarifications from users.

    This tool allows the PlannerLLM to ask clarification questions when
    requirements are ambiguous or insufficient.

    Returns:
        A ChatCompletionToolParam for get_clarifications
    """
    logger.debug("Creating get_clarifications tool definition")
    tool_definition: ChatCompletionToolParam = {
        "type": "function",
        "function": {
            "name": "get_clarifications",
            "description": "Use this tool when requirements are ambiguous, incomplete, or have multiple possible interpretations. Ask specific questions to gather critical missing details, understand user preferences, or confirm technical specifications before proceeding with a solution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific, focused clarification questions. Each question should address a distinct ambiguity or missing information. Prioritize questions that are blocking progress and phrase them neutrally to avoid biasing the user's response.",
                    }
                },
                "required": ["questions"],
            },
        },
    }
    logger.debug(
        f"Created tool definition with name: {tool_definition['function']['name']}"
    )
    return tool_definition


def get_clarifications(questions: List[str]) -> Dict[str, str]:
    """
    Interactive callback function to get answers to clarification questions.

    Args:
        questions: List of clarification questions

    Returns:
        Dictionary of questions and answers
    """
    logger.info("Getting clarifications from user")
    logger.debug(f"Number of clarification questions: {len(questions)}")
    print("\nI need some clarifications to better understand your requirements:")
    clarifications = {}

    logger.info(f"Processing {len(questions)} clarification questions")
    for i, question in enumerate(questions, 1):
        logger.debug(f"Asking clarification question {i}/{len(questions)}: {question}")
        print(f"\n----------------------------------")
        print(f"Question: {question}")
        answer = input("Your answer: ").strip()
        logger.debug(
            f"Received answer for question {i}: {answer[:30]}..."
            if len(answer) > 30
            else f"Received answer for question {i}: {answer}"
        )
        clarifications[question] = answer

    logger.info("Clarification gathering complete")
    logger.debug(f"Collected {len(clarifications)} clarifications")
    return clarifications
