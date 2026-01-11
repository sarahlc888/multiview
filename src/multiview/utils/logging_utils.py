"""Simple logging setup for benchmark."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter that adds colors to console output based on log level."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[36m",  # Cyan
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"
    DIM = "\033[2m"  # Dim text for logger names

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color codes."""
        # Color only the level name, not the message
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # For DEBUG, also gray the entire message
        if record.levelname.startswith("\033[90m"):
            record.msg = f"\033[90m{record.msg}{self.RESET}"

        # Add subtle dim color to logger name
        record.name = f"{self.DIM}{record.name}{self.RESET}"

        # Format the record
        result = super().format(record)
        return result


def setup_logging(level: str = "INFO", output_file: str | None = None) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        output_file: Optional file path to write logs to
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Format string
    format_string = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredConsoleFormatter(format_string, datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if output_file:
        # Create parent directories if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party packages
    for package in [
        "urllib3",
        "filelock",
        "datasets",
        "huggingface_hub",
        "httpx",
        "httpcore",
        "fsspec",
        "huggingface_hub.repocard",
        "google_genai",
        "google_genai._api_client",
    ]:
        logging.getLogger(package).setLevel(logging.ERROR)


def setup_logging_from_config(cfg) -> None:
    """Set up logging from Hydra config.

    Args:
        cfg: Hydra config with logging settings
    """
    level = getattr(cfg.logging, "level", "INFO")
    output_file = getattr(cfg.logging, "output_file", None)
    setup_logging(level=level, output_file=output_file)
