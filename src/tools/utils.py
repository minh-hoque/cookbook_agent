"""
Utility functions for working with tools.

This module contains helper functions for working with OpenAI tools.
"""

from typing import Dict, List, Any, Optional, Callable
import json
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)


def extract_tool_arguments(tool_call: ChatCompletionMessageToolCall) -> Dict[str, Any]:
    """
    Extract and parse arguments from a tool call.

    Args:
        tool_call: The tool call object from OpenAI

    Returns:
        Dictionary of parsed arguments
    """
    try:
        return json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        # Return empty dict on JSON parse error
        return {}


def process_tool_call(
    tool_call: ChatCompletionMessageToolCall,
    handlers: Dict[str, Callable[[Dict[str, Any]], Any]],
) -> Optional[Any]:
    """
    Process a tool call with the appropriate handler.

    Args:
        tool_call: The tool call object from OpenAI
        handlers: A dictionary mapping tool names to handler functions

    Returns:
        The result of the handler function or None if no handler found
    """
    function_name = tool_call.function.name
    if function_name in handlers:
        args = extract_tool_arguments(tool_call)
        return handlers[function_name](args)
    return None


def has_tool_call(message: ChatCompletionMessage, tool_name: str) -> bool:
    """
    Check if a message has a call to a specific tool.

    Args:
        message: The message to check
        tool_name: The name of the tool to check for

    Returns:
        True if the message has a call to the specified tool, False otherwise
    """
    # Safety check: If message is None, return False
    if message is None:
        return False

    # If there are no tool_calls attribute or it's None, return False
    if not hasattr(message, "tool_calls") or message.tool_calls is None:
        return False

    # If tool_calls is empty, return False
    if len(message.tool_calls) == 0:
        return False

    # Check if any tool call matches the tool name
    for tool_call in message.tool_calls:
        if tool_call.function.name == tool_name:
            return True

    return False
