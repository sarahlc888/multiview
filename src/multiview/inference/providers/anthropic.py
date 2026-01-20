"""Anthropic provider for completions.

Simplified implementation focusing on core functionality.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from tqdm import tqdm

from multiview.constants import ANTHROPIC_API_KEY
from multiview.inference.cost_tracker import record_usage

logger = logging.getLogger(__name__)


def _anthropic_single_completion(
    client,
    prompt: str,
    prefill: str | None,
    model_name: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    **kwargs,
) -> dict:
    """Process a single Anthropic completion with retry logic.

    Args:
        client: Anthropic client instance
        prompt: Prompt text
        prefill: Optional prefill string to force response start
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay in seconds for exponential backoff
        retry_backoff_factor: Multiplier for backoff delay
        **kwargs: Additional Anthropic API parameters

    Returns:
        Dict with "text" key containing the completion
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        ) from None

    # Build messages
    messages = [{"role": "user", "content": prompt}]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})

    # Retry with exponential backoff
    delay = initial_retry_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )
            completion_text = response.content[0].text
            if prefill:
                completion_text = prefill + completion_text

            # Record usage
            if hasattr(response, "usage"):
                record_usage(
                    model_name=model_name,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            return {"text": completion_text}

        except anthropic.RateLimitError as e:
            last_exception = e  # noqa: F841
            if attempt < max_retries:
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                delay *= retry_backoff_factor
            else:
                logger.error(f"Rate limit error after {max_retries + 1} attempts: {e}")
                return {"text": ""}

        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            return {"text": ""}

    # Should not reach here, but just in case
    return {"text": ""}


def anthropic_completions(
    prompts: list[str],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    force_prefills: list[str] | None = None,
    max_workers: int = 5,
    max_retries: int = 5,
    initial_retry_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    **kwargs,
) -> dict:
    """Get Anthropic completions.

    Args:
        prompts: List of prompts (will be formatted as user messages)
        model_name: Model name (e.g., "claude-3-5-sonnet-20241022")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        force_prefills: Optional list of prefill strings to force response start
        max_workers: Maximum concurrent API requests (default 5)
        max_retries: Maximum retry attempts per request (default 5)
        initial_retry_delay: Initial delay for exponential backoff (default 1.0s)
        retry_backoff_factor: Backoff multiplier (default 2.0)
        **kwargs: Additional Anthropic API parameters

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "text" key with the generated text
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required. Install with: pip install anthropic"
        ) from None

    # Initialize client (shared across threads - Anthropic client is thread-safe)
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Pair prompts with their corresponding prefills
    prompt_prefill_pairs = []
    for i, prompt in enumerate(prompts):
        prefill = None
        if force_prefills and i < len(force_prefills):
            prefill = force_prefills[i]
        prompt_prefill_pairs.append((prompt, prefill))

    # Create partial function with fixed parameters
    completion_fn = partial(
        _anthropic_single_completion,
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
            tqdm(
                executor.map(
                    lambda pair: completion_fn(prompt=pair[0], prefill=pair[1]),
                    prompt_prefill_pairs,
                ),
                desc="Anthropic completions",
                total=len(prompt_prefill_pairs),
                unit="req",
            )
        )

    return {"completions": completions}
