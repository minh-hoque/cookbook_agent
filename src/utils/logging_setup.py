"""
Logging setup utilities.

This module provides functions for setting up logging configuration.
"""

import logging
from typing import Optional
from src.tools.debug import DebugLevel, setup_logging


def configure_logging(debug_level: DebugLevel = DebugLevel.INFO) -> None:
    """
    Configure the logging system.

    Args:
        debug_level: The debug level (ERROR, INFO, DEBUG)
    """
    # Initialize logging
    setup_logging(level=debug_level)

    # Reduce logging for external libraries to WARNING or higher
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Silence other noisy libraries
    for logger_name in logging.root.manager.loggerDict:
        if any(
            name in logger_name.lower()
            for name in ["http", "urllib", "requests", "openai", "aiohttp"]
        ):
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def test_logging():
    """Test function to verify that the logging system captures caller information correctly."""
    logger = logging.getLogger(__name__)
    logger.debug("Test debug message from test_logging function")
    logger.info("Test info message from test_logging function")
    logger.error("Test error message from test_logging function")
