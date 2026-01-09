"""Logging for all LM prompts and responses.

Provides detailed logging of prompts and responses for debugging and auditing.
Enable with environment variable: MULTIVIEW_LOG_PROMPTS=1
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if prompt logging is enabled via environment variable
PROMPT_LOGGING_ENABLED = os.getenv("MULTIVIEW_LOG_PROMPTS", "0") == "1"
PROMPT_LOG_DIR = Path(os.getenv("MULTIVIEW_PROMPT_LOG_DIR", "outputs/prompt_logs"))


def log_prompt_response(
    prompt: str,
    response: Any,
    metadata: dict | None = None,
) -> None:
    """Log a prompt and its response to a file.

    Args:
        prompt: The prompt sent to the LM
        response: The response from the LM
        metadata: Optional metadata (model, config, cache_alias, etc.)
    """
    if not PROMPT_LOGGING_ENABLED:
        return

    try:
        # Create log directory if it doesn't exist
        PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = PROMPT_LOG_DIR / f"prompt_{timestamp}.jsonl"

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
        }

        # Write to file (append mode)
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        logger.debug(f"Logged prompt/response to {log_file}")

    except Exception as e:
        logger.warning(f"Failed to log prompt/response: {e}")


def log_batch_prompts_responses(
    prompts: list[str],
    responses: list[Any],
    metadata: dict | None = None,
) -> None:
    """Log a batch of prompts and responses to a single file.

    Args:
        prompts: List of prompts sent to the LM
        responses: List of responses from the LM
        metadata: Optional metadata (model, config, cache_alias, etc.)
    """
    if not PROMPT_LOGGING_ENABLED:
        return

    try:
        # Create log directory if it doesn't exist
        PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Use cache_alias or metadata to create descriptive filename
        descriptor = ""
        if metadata and metadata.get("cache_alias"):
            descriptor = f"_{metadata['cache_alias']}"
        elif metadata and metadata.get("config"):
            descriptor = f"_{metadata['config']}"

        log_file = PROMPT_LOG_DIR / f"batch{descriptor}_{timestamp}.jsonl"

        # Prepare log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(prompts),
            "metadata": metadata or {},
            "prompts_responses": [
                {
                    "index": i,
                    "prompt": prompt,
                    "response": response,
                }
                for i, (prompt, response) in enumerate(
                    zip(prompts, responses, strict=False)
                )
            ],
        }

        # Write to file
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)

        logger.info(f"Logged {len(prompts)} prompts/responses to {log_file}")

    except Exception as e:
        logger.warning(f"Failed to log batch prompts/responses: {e}")


def is_prompt_logging_enabled() -> bool:
    """Check if prompt logging is enabled."""
    return PROMPT_LOGGING_ENABLED


def get_prompt_log_dir() -> Path:
    """Get the directory where prompt logs are stored."""
    return PROMPT_LOG_DIR
