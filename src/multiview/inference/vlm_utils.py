"""Vision-Language Model utilities for image handling.

This module provides utilities for loading and encoding images
for use with vision-language models (VLMs) like Gemini Vision.

Supported image sources:
- URLs (http:// or https://)
- Local file paths
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def load_image_from_source(source: str, timeout: int = 10) -> bytes:
    """Load image from URL or local file path.

    Args:
        source: Image source - either a URL (http:// or https://) or local file path
        timeout: Timeout in seconds for URL requests (default: 10)

    Returns:
        Image data as bytes

    Raises:
        ValueError: If source is invalid or image cannot be loaded
        FileNotFoundError: If local file path doesn't exist
        requests.exceptions.RequestException: If URL request fails
    """
    if not source:
        raise ValueError("Image source cannot be empty")

    # Check if source is a URL
    if source.startswith("http://") or source.startswith("https://"):
        logger.debug(f"Loading image from URL: {source}")
        try:
            # Add User-Agent header to avoid 403 errors from some sites
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(source, timeout=timeout, headers=headers)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            raise ValueError(f"Failed to fetch image from URL {source}: {e}") from e
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error loading image from URL {source}: {e}") from e

    # Assume source is a local file path
    logger.debug(f"Loading image from local path: {source}")
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {source}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {source}")

    try:
        return path.read_bytes()
    except Exception as e:
        raise ValueError(f"Error reading image file {source}: {e}") from e


def detect_mime_type(image_bytes: bytes, source_path: str | None = None) -> str:
    """Detect MIME type of image.

    Args:
        image_bytes: Image data as bytes
        source_path: Optional source path for extension-based detection

    Returns:
        MIME type string (e.g., "image/jpeg", "image/png")
        Defaults to "image/jpeg" if detection fails
    """
    # Try to detect from file extension if path provided
    if source_path:
        mime_type, _ = mimetypes.guess_type(source_path)
        if mime_type and mime_type.startswith("image/"):
            logger.debug(f"Detected MIME type from path: {mime_type}")
            return mime_type

    # Try to detect from magic bytes
    if image_bytes[:2] == b"\xff\xd8":  # JPEG
        return "image/jpeg"
    elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
        return "image/png"
    elif image_bytes[:4] == b"GIF8":  # GIF
        return "image/gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":  # WebP
        return "image/webp"

    # Default to JPEG
    logger.warning("Could not detect MIME type, defaulting to image/jpeg")
    return "image/jpeg"


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.

    Args:
        image_bytes: Image data as bytes

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def prepare_image_for_gemini(
    image_source: str, timeout: int = 10
) -> dict[str, str | dict]:
    """Prepare image in Gemini's expected format.

    Args:
        image_source: Image URL or local file path
        timeout: Timeout in seconds for URL requests

    Returns:
        Dict in Gemini's inline_data format:
        {"inline_data": {"mime_type": "image/jpeg", "data": "base64_string"}}

    Raises:
        ValueError: If image cannot be loaded or encoded
        FileNotFoundError: If local file doesn't exist
    """
    # Load image
    image_bytes = load_image_from_source(image_source, timeout=timeout)

    # Detect MIME type
    mime_type = detect_mime_type(image_bytes, source_path=image_source)

    # Encode to base64
    base64_data = encode_image_base64(image_bytes)

    logger.debug(
        f"Prepared image: {len(image_bytes)} bytes, {mime_type}, "
        f"{len(base64_data)} base64 chars"
    )

    return {"inline_data": {"mime_type": mime_type, "data": base64_data}}


def validate_image_list(images: list[str | None]) -> list[str | None]:
    """Validate and log information about image list.

    Args:
        images: List of image sources (URLs or paths), may contain None

    Returns:
        Same list (validation only)

    Raises:
        ValueError: If images is not a list
    """
    if not isinstance(images, list):
        raise ValueError(f"Images must be a list, got {type(images)}")

    valid_count = sum(1 for img in images if img is not None)
    none_count = len(images) - valid_count

    logger.debug(
        f"Image list validation: {len(images)} total, "
        f"{valid_count} valid, {none_count} None"
    )

    return images
