"""Gemini provider for completions.

Uses Google's Generative AI API.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

from multiview.constants import GEMINI_API_KEY
from multiview.inference.cost_tracker import record_usage

logger = logging.getLogger(__name__)


def prettify_parts(contents):
    pretty_version = []
    for msg in contents:
        pretty_msg = {"role": msg["role"], "parts": []}
        # copy over the 'parts', but for inline data, truncate the data field to a max of like 20 chars
        for part in msg["parts"]:
            if "inline_data" in part:
                pretty_msg["parts"].append(
                    {
                        "inline_data": {
                            "mime_type": part["inline_data"]["mime_type"],
                            "data": part["inline_data"]["data"][:20] + "...",
                        }
                    }
                )
            else:
                pretty_msg["parts"].append(part)
        pretty_version.append(pretty_msg)
    return pretty_version


def _gemini_single_completion(
    client,
    prompt: str,
    prefill: str | None,
    image: str | list[str] | None,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    request_timeout: float = 120.0,
    **kwargs,
) -> dict:
    """Process a single Gemini completion with retry logic and timeout.

    Args:
        client: Gemini client instance
        prompt: Prompt text
        prefill: Optional prefill string to force response start
        image: Optional image source(s) for vision tasks. Can be:
               - Single image: str (URL or file path)
               - Multiple images: list[str] (for multi-image prompts)
               - None for text-only prompts
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay in seconds for exponential backoff
        retry_backoff_factor: Multiplier for backoff delay
        request_timeout: Timeout in seconds for each API request (default 120s)
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

    # Build contents with optional image(s)
    if image:
        # Import VLM utilities for image handling
        from multiview.inference.vlm_utils import prepare_image_for_gemini

        # Support both single image (str) and multiple images (list)
        images_to_process = [image] if isinstance(image, str) else image

        try:
            # Prepare all images in Gemini format
            # Handle individual image failures gracefully
            image_parts = []
            failed_images = []

            for img in images_to_process:
                if not img:  # Skip None values
                    continue

                try:
                    image_part = prepare_image_for_gemini(img)
                    image_parts.append(image_part)
                    # logger.debug(f"Prepared image for Gemini: {img}")
                except Exception as img_error:
                    logger.warning(f"Failed to load image {img}: {img_error}")
                    failed_images.append(img)
                    # Continue with other images

            if not image_parts:
                # All images failed, fall back to text-only
                raise ValueError(
                    f"No valid images to process. Failed: {len(failed_images)}/{len(images_to_process)}"
                )

            # Build multimodal contents
            # If prompt contains <image> markers, interleave images at those locations
            # Otherwise, put all images at the start
            if "<image>" in prompt:
                # Split text by <image> markers and interleave with images
                text_parts = prompt.split("<image>")
                parts = []

                # Interleave text and images
                for i, text_part in enumerate(text_parts):
                    if text_part:  # Add non-empty text
                        parts.append({"text": text_part})
                    if i < len(image_parts):  # Add image if available
                        parts.append(image_parts[i])

                # If there are more images than markers, add remaining images at end
                if len(image_parts) > len(text_parts) - 1:
                    for remaining_img in image_parts[len(text_parts) - 1 :]:
                        parts.append(remaining_img)
            else:
                # Default: all images at start, then text
                parts = image_parts + [{"text": prompt}]

            if prefill:
                contents = [
                    {"role": "user", "parts": parts},
                    {"role": "model", "parts": [{"text": prefill}]},
                ]
            else:
                contents = [{"role": "user", "parts": parts}]
            logger.debug(f"Contents: {prettify_parts(contents)=}")
        except Exception as e:
            logger.warning(
                f"Failed to load image(s) {image}, falling back to text-only: {e}"
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
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                output_tokens = getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )

                # Apply tiered pricing for Gemini 2.5 Pro based on context length
                # Prompts >200K tokens are charged at higher rates
                cost_model_name = model_name
                if model_name == "gemini-2.5-pro" and input_tokens > 200_000:
                    cost_model_name = "gemini-2.5-pro-long"
                    logger.debug(
                        f"Long context detected ({input_tokens:,} tokens > 200K), "
                        f"using higher pricing tier"
                    )

                record_usage(
                    model_name=cost_model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            return {"text": completion_text}

        except Exception as e:
            error_str = str(e).lower()

            # Check if this is a timeout (retriable)
            if "timeout" in error_str or "timed out" in error_str:
                if attempt < max_retries:
                    logger.warning(
                        f"Request timeout (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= retry_backoff_factor
                else:
                    logger.error(f"Timeout after {max_retries + 1} attempts: {e}")
                    return {"text": ""}

            # Check if this is a quota exhaustion (not retriable)
            elif "quota exceeded" in error_str or "free_tier" in error_str:
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
    images: list[str | list[str] | None] | None = None,
    max_workers: int = 20,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    request_timeout: float = 120.0,
    **kwargs,
) -> dict:
    """Get Gemini completions with optional vision support.

    Args:
        prompts: List of prompts
        model_name: Model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        force_prefills: Optional list of prefill strings to force response start
        images: Optional list of image sources for vision tasks. Each element can be:
                - str: Single image (URL or file path)
                - list[str]: Multiple images for multi-image prompts
                - None: No images for text-only prompts
                Each element corresponds to the prompt at the same index
        max_workers: Maximum concurrent API requests (default 20)
        max_retries: Maximum retry attempts per request (default 5)
        initial_retry_delay: Initial delay for exponential backoff (default 1.0s)
        retry_backoff_factor: Backoff multiplier (default 2.0)
        request_timeout: Timeout in seconds for each API request (default 120s)
        **kwargs: Additional Gemini API parameters

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "text" key with the generated text
    """
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError:
        raise ImportError(
            "google-genai package required. Install with: pip install google-genai"
        ) from None

    # Initialize client with timeout (shared across threads - Gemini client is thread-safe)
    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set"
        )

    client = genai.Client(
        api_key=api_key,
        http_options=HttpOptions(timeout=int(request_timeout * 1000)),
    )

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
        request_timeout=request_timeout,
        **kwargs,
    )

    # Log concurrency info
    logger.info(
        f"Processing {len(prompts)} prompts with {max_workers} concurrent workers "
        f"(timeout: {request_timeout}s per request)"
    )

    # Execute concurrently with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        completions = []

        # Submit all tasks
        futures = [
            executor.submit(completion_fn, prompt=t[0], prefill=t[1], image=t[2])
            for t in prompt_prefill_image_tuples
        ]

        # Collect results with progress bar
        for future in tqdm(futures, desc="Gemini completions", unit="req"):
            try:
                result = future.result()
                completions.append(result)
            except Exception as e:
                logger.error(f"Request failed with exception: {e}")
                completions.append({"text": ""})

    return {"completions": completions}


def _extract_embedding_values(embedding_obj) -> list[float] | None:
    """Extract embedding values from Gemini embed_content response objects."""
    if embedding_obj is None:
        return None
    if isinstance(embedding_obj, dict):
        values = embedding_obj.get("values")
        if values is not None:
            return [float(x) for x in values]
    if hasattr(embedding_obj, "values"):
        return [float(x) for x in embedding_obj.values]
    if isinstance(embedding_obj, list | tuple):
        return [float(x) for x in embedding_obj]
    return None


def gemini_embedding_completions(
    prompts: list[str],
    model_name: str,
    **kwargs,
) -> dict:
    """Get text embeddings from Gemini Embedding API.

    Args:
        prompts: List of texts to embed
        model_name: Gemini embedding model name (e.g., "gemini-embedding-001")
        **kwargs: Additional parameters including:
            - instructions: Optional list of instructions to prepend to prompts
            - output_dimensionality: Optional reduced embedding dimension
            - task_type: Optional task type string (e.g., "RETRIEVAL_DOCUMENT")
            - title: Optional title when task_type is RETRIEVAL_DOCUMENT
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package required. Install with: pip install google-genai"
        ) from None

    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set"
        )

    client = genai.Client(api_key=api_key)

    # Handle embedding instructions by prepending them to prompts
    instructions = kwargs.pop("instructions", None)

    # Strip internal/provider-level arguments not used by embed_content
    for key in [
        "max_retries",
        "initial_retry_delay",
        "retry_backoff_factor",
        "max_workers",
        "temperature",
        "max_tokens",
    ]:
        kwargs.pop(key, None)

    final_prompts = []
    for i, prompt in enumerate(prompts):
        final_prompt = prompt
        if instructions and i < len(instructions):
            final_prompt = instructions[i] + final_prompt
        final_prompts.append(final_prompt)

    # Build optional config
    config_kwargs = {}
    if "output_dimensionality" in kwargs:
        config_kwargs["output_dimensionality"] = kwargs.pop("output_dimensionality")
    if "task_type" in kwargs:
        config_kwargs["task_type"] = kwargs.pop("task_type")
    if "title" in kwargs:
        config_kwargs["title"] = kwargs.pop("title")

    config = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

    # Try batch embedding first
    try:
        result = client.models.embed_content(
            model=model_name,
            contents=final_prompts,
            config=config,
        )
        embeddings = result.embeddings
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        vectors = [_extract_embedding_values(e) for e in embeddings]
        if any(v is None for v in vectors) or len(vectors) != len(final_prompts):
            raise ValueError("Unexpected embeddings response shape")
    except Exception as e:
        logger.warning(f"Batch embedding failed, falling back to single: {e}")
        vectors = []
        for prompt in final_prompts:
            single = client.models.embed_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            values = _extract_embedding_values(
                single.embeddings if hasattr(single, "embeddings") else None
            )
            vectors.append(values or [])

    completions = [{"vector": v} for v in vectors]
    return {"completions": completions}
