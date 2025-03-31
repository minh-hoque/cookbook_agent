"""
Debugging and logging utilities for the Cookbook Agent.

This module provides standardized logging functions with different severity levels.
"""

import logging
import os
import sys
from enum import Enum
from typing import Optional


class DebugLevel(Enum):
    """Debug level enum for controlling logging verbosity."""

    ERROR = logging.ERROR
    INFO = logging.INFO
    DEBUG = logging.DEBUG


# ANSI color codes for colored console output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[37m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to the output based on log level."""

    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        colored_record = logging.makeLogRecord(record.__dict__)

        # Add color based on level
        level_color = ""
        if record.levelno >= logging.ERROR:
            level_color = Colors.RED + Colors.BOLD
        elif record.levelno >= logging.WARNING:
            level_color = Colors.YELLOW
        elif record.levelno >= logging.INFO:
            level_color = Colors.GREEN
        elif record.levelno >= logging.DEBUG:
            level_color = Colors.BLUE

        # Add the colored levelname
        colored_record.levelname_colored = (
            f"{level_color}{record.levelname}{Colors.RESET}"
        )

        # Add colored pathname (if it's a file path)
        if hasattr(record, "pathname") and os.path.exists(record.pathname):
            # Get relative path if possible
            try:
                rel_path = os.path.relpath(record.pathname)
                colored_record.pathname_colored = (
                    f"{Colors.CYAN}{rel_path}{Colors.RESET}"
                )
            except ValueError:
                # If relpath fails, use the full path
                colored_record.pathname_colored = (
                    f"{Colors.CYAN}{record.pathname}{Colors.RESET}"
                )
        else:
            colored_record.pathname_colored = record.pathname

        # Add colored function name
        colored_record.funcName_colored = (
            f"{Colors.MAGENTA}{record.funcName}{Colors.RESET}"
        )

        # Format with the colored attributes
        return super().format(colored_record)


def setup_logging(level=DebugLevel.INFO):
    """
    Set up application-wide logging configuration.

    Args:
        level: The DebugLevel to use
    """
    # Convert our enum to logging level
    log_level = level.value

    # Remove all existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set root logger level
    root_logger.setLevel(log_level)

    # Set third-party loggers to a higher level to filter out unwanted logs
    # First set all existing loggers to ERROR level
    for logger_name in logging.Logger.manager.loggerDict:
        # Skip our own application loggers
        if not logger_name.startswith("src") and not logger_name == "__main__":
            logger_obj = logging.getLogger(logger_name)
            logger_obj.setLevel(logging.ERROR)

    # Create console formatter with colors
    console_format = "[%(asctime)s] %(levelname_colored)s | %(pathname_colored)s:%(lineno)d | %(funcName_colored)s() | %(message)s"
    console_formatter = ColoredFormatter(console_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log initial setup
    logging.info(f"Logging initialized with level: {level.name}")


# Get module-level logger - each module should use logging.getLogger(__name__)
logger = logging.getLogger(__name__)


# For backward compatibility with existing code, provide direct functions
def debug(message):
    """Log a debug message using the module's logger."""
    logger.debug(message)


def info(message):
    """Log an info message using the module's logger."""
    logger.info(message)


def error(message):
    """Log an error message using the module's logger."""
    logger.error(message)


def set_level(level):
    """Set the logging level for the root logger."""
    logging.root.setLevel(level.value)
    logging.info(f"Log level changed to {level.name}")
