"""Utility functions for inference module.

Includes helpers for prompt loading and text processing.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_or_return(text_or_path: str, base_dir: str | Path | None = None) -> str:
    """Load text from file if it's a path, otherwise return as-is.

    This utility enables backward compatibility by accepting both:
    - File paths (e.g., "prompts/lm_judge/triplet.txt")
    - Inline strings (e.g., "You are a helpful assistant...")

    Args:
        text_or_path: Either a file path or inline text
        base_dir: Base directory for relative paths. If None, uses the
            multiview package root (src/multiview/)

    Returns:
        Text content (loaded from file if path, otherwise returned as-is)

    Examples:
        >>> # File path - loads from file
        >>> read_or_return("prompts/lm_judge/triplet.txt")
        "You are a helpful assistant..."

        >>> # Inline string - returns as-is
        >>> read_or_return("You are a helpful assistant...")
        "You are a helpful assistant..."
    """
    # Determine base directory
    if base_dir is None:
        # Default to the multiview package root (parent of inference/)
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)

    # Check if this looks like a file path.
    #
    # Important: prompts are often stored as inline strings that may contain
    # escape sequences like "\\n" or "\\'". Treating *any* backslash as a path
    # causes us to try to stat() a filename equal to the entire prompt text,
    # which can raise OSError: [Errno 63] File name too long on macOS.
    stripped = text_or_path.strip()
    windows_path = (
        len(stripped) >= 3 and stripped[1] == ":" and stripped[2] in ("\\", "/")
    ) or stripped.startswith("\\\\")
    has_newline = ("\n" in text_or_path) or ("\r" in text_or_path)
    has_escaped_newline = (
        ("\\n" in text_or_path) or ("\\r" in text_or_path) or ("\\t" in text_or_path)
    )
    looks_like_path = (
        (
            "/" in stripped
            or windows_path
            or stripped.endswith((".txt", ".md", ".prompt"))
        )
        and not has_newline
        and not has_escaped_newline
        and len(stripped) <= 300
    )

    if looks_like_path:
        try:
            # Try to resolve as a path
            if os.path.isabs(stripped):
                file_path = Path(stripped)
            else:
                file_path = base_dir / stripped

            # If file exists, load it
            if file_path.exists() and file_path.is_file():
                return file_path.read_text(encoding="utf-8")
        except OSError:
            # Fall back to returning as-is for pathological cases (e.g., huge strings).
            pass

    # Otherwise, return as-is (inline string)
    return text_or_path
