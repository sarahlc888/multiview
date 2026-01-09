"""Simple logging setup for benchmark."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


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

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")
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
    ]:
        logging.getLogger(package).setLevel(logging.WARNING)


def setup_logging_from_config(cfg) -> None:
    """Set up logging from Hydra config.

    Args:
        cfg: Hydra config with logging settings
    """
    level = getattr(cfg.logging, "level", "INFO")
    output_file = getattr(cfg.logging, "output_file", None)
    setup_logging(level=level, output_file=output_file)
