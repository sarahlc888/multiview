"""Anthropic provider for completions.

Simplified implementation focusing on core functionality.
"""

import logging
import time

from multiview.constants import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)


def anthropic_completions(
    prompts: list[str],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    force_prefills: list[str] | None = None,
    **kwargs,
) -> dict:
    """Get Anthropic completions.

    Args:
        prompts: List of prompts (will be formatted as user messages)
        model_name: Model name (e.g., "claude-3-5-sonnet-20241022")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        force_prefills: Optional list of prefill strings to force response start
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

    # Initialize client
    api_key = ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Process each prompt
    completions = []
    for i, prompt in enumerate(prompts):
        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Add prefill as assistant message if provided
        prefill = None
        if force_prefills and i < len(force_prefills):
            prefill = force_prefills[i]
            if prefill:
                messages.append({"role": "assistant", "content": prefill})

        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )

            completion_text = response.content[0].text
            # Prepend prefill to completion if it was used
            if prefill:
                completion_text = prefill + completion_text
            completions.append({"text": completion_text})

        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}. Sleeping 10s and retrying...")
            time.sleep(10)
            # Retry once
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
            completions.append({"text": completion_text})

        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            # Return empty completion on error
            completions.append({"text": ""})

    return {"completions": completions}
