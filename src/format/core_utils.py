"""
Core utility functions for formatting data.

This module provides basic utility functions used across various formatting operations.
"""

import json
import logging
from typing import Any

# Set up logger
logger = logging.getLogger(__name__)


def format_json(data: Any, indent: int = 2) -> str:
    """
    Format any JSON-serializable data with proper indentation.

    Args:
        data: Any JSON-serializable data
        indent: Number of spaces for indentation

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent)
