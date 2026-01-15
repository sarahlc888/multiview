"""Gemini provider for completions.

Uses Google's Generative AI API.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from multiview.constants import GEMINI_API_KEY
from multiview.inference.cost_tracker import record_usage

logger = logging.getLogger(__name__)


def _gemini_single_completion(
    client,
    prompt: str,
    prefill: str | None,
    image: str | None,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    **kwargs,
) -> dict:
    """Process a single Gemini completion with retry logic.

    Args:
        client: Gemini client instance
        prompt: Prompt text
        prefill: Optional prefill string to force response start
        image: Optional image source (URL or file path) for vision tasks
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay in seconds for exponential backoff
        retry_backoff_factor: Multiplier for backoff delay
        **kwargs: Additional Gemini API parameters

    Returns:
        Dict with "text" key containing the completion
    """
    try:
        from google import genai  # noqa: F401, F811
        from google.genai.types import GenerateContentConfig
    except ImportError:
        raise ImportError(
            "google-genai package required. Install with: pip install google-genai"
        ) from None

    # Build contents with optional image
    if image:
        # Import VLM utilities for image handling
        from multiview.inference.vlm_utils import prepare_image_for_gemini

        try:
            # Prepare image in Gemini format
            image_part = prepare_image_for_gemini(image)
            logger.debug(f"Prepared image for Gemini: {image}")

            # Build multimodal contents
            if prefill:
                contents = [
                    {"role": "user", "parts": [image_part, {"text": prompt}]},
                    {"role": "model", "parts": [{"text": prefill}]},
                ]
            else:
                # For simple case, can use parts directly
                contents = [{"role": "user", "parts": [image_part, {"text": prompt}]}]

        except Exception as e:
            logger.warning(
                f"Failed to load image {image}, falling back to text-only: {e}"
            )
            # Fall back to text-only
            if prefill:
                contents = [
                    {"role": "user", "parts": [{"text": prompt}]},
                    {"role": "model", "parts": [{"text": prefill}]},
                ]
            else:
                contents = prompt
    else:
        # Text-only (original behavior)
        if prefill:
            contents = [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "model", "parts": [{"text": prefill}]},
            ]
        else:
            contents = prompt

    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        **kwargs,
    )

    # Retry with exponential backoff
    delay = initial_retry_delay

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            completion_text = response.text
            if prefill:
                completion_text = prefill + completion_text

            # Record usage
            if hasattr(response, "usage_metadata"):
                record_usage(
                    model_name=model_name,
                    input_tokens=getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ),
                    output_tokens=getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ),
                )

            return {"text": completion_text}

        except Exception as e:
            error_str = str(e).lower()

            # Check if this is a quota exhaustion (not retriable)
            if "quota exceeded" in error_str or "free_tier" in error_str:
                logger.error(
                    f"Gemini quota exhausted: {e}\n\n"
                    "You appear to be using the FREE TIER Google AI Studio API.\n"
                    "To use paid tier:\n"
                    "  1. Use Vertex AI authentication instead of API key, OR\n"
                    "  2. Wait for quota to reset (usually daily), OR\n"
                    "  3. Use a different model or reduce request volume\n"
                    "See: https://ai.google.dev/gemini-api/docs/rate-limits"
                )
                return {"text": ""}

            # Check if this is a transient rate limit (retriable)
            if "rate" in error_str or "429" in error_str:
                if attempt < max_retries:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= retry_backoff_factor
                else:
                    logger.error(
                        f"Rate limit error after {max_retries + 1} attempts: {e}"
                    )
                    return {"text": ""}
            else:
                # Other errors - don't retry
                logger.error(f"Error getting completion: {e}")
                return {"text": ""}

    # Should not reach here, but just in case
    return {"text": ""}


def gemini_completions(
    prompts: list[str],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    force_prefills: list[str] | None = None,
    images: list[str | None] | None = None,
    max_workers: int = 5,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    **kwargs,
) -> dict:
    """Get Gemini completions with optional vision support.

    Args:
        prompts: List of prompts
        model_name: Model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        force_prefills: Optional list of prefill strings to force response start
        images: Optional list of image sources (URLs or file paths) for vision tasks
                Each image corresponds to the prompt at the same index
                Use None for text-only prompts in a mixed batch
        max_workers: Maximum concurrent API requests (default 5)
        max_retries: Maximum retry attempts per request (default 5)
        initial_retry_delay: Initial delay for exponential backoff (default 1.0s)
        retry_backoff_factor: Backoff multiplier (default 2.0)
        **kwargs: Additional Gemini API parameters

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "text" key with the generated text
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package required. Install with: pip install google-genai"
        ) from None

    # Initialize client (shared across threads - Gemini client is thread-safe)
    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set"
        )

    client = genai.Client(api_key=api_key)

    # Validate and log image list if provided
    if images is not None:
        from multiview.inference.vlm_utils import validate_image_list

        validate_image_list(images)
        logger.info(
            f"Processing {len(prompts)} prompts with {sum(1 for img in images if img is not None)} images"
        )

    # Pair prompts with their corresponding prefills and images
    prompt_prefill_image_tuples = []
    for i, prompt in enumerate(prompts):
        prefill = None
        if force_prefills and i < len(force_prefills):
            prefill = force_prefills[i]

        image = None
        if images and i < len(images):
            image = images[i]

        prompt_prefill_image_tuples.append((prompt, prefill, image))

    # Create partial function with fixed parameters
    completion_fn = partial(
        _gemini_single_completion,
        client=client,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        initial_retry_delay=initial_retry_delay,
        retry_backoff_factor=retry_backoff_factor,
        **kwargs,
    )

    # Execute concurrently (max_workers=1 makes it sequential)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        completions = list(
            executor.map(
                lambda t: completion_fn(prompt=t[0], prefill=t[1], image=t[2]),
                prompt_prefill_image_tuples,
            )
        )

    return {"completions": completions}
