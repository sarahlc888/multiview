"""Gemini provider for completions.

Uses Google's Generative AI API.
"""

import logging
import time

from multiview.constants import GEMINI_API_KEY

logger = logging.getLogger(__name__)


def gemini_completions(
    prompts: list[str],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    force_prefills: list[str] | None = None,
    **kwargs,
) -> dict:
    """Get Gemini completions.

    Args:
        prompts: List of prompts
        model_name: Model name (e.g., "gemini-2.0-flash-exp", "gemini-1.5-pro")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        force_prefills: Optional list of prefill strings to force response start
        **kwargs: Additional Gemini API parameters

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "text" key with the generated text
    """
    try:
        from google import genai
        from google.genai.types import GenerateContentConfig
    except ImportError:
        raise ImportError(
            "google-genai package required. Install with: pip install google-genai"
        ) from None

    # Initialize client
    api_key = GEMINI_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set"
        )

    client = genai.Client(api_key=api_key)

    # Process each prompt
    completions = []
    for i, prompt in enumerate(prompts):
        # Get prefill if provided
        prefill = None
        if force_prefills and i < len(force_prefills):
            prefill = force_prefills[i]

        # Build contents (can be string or list of messages)
        if prefill:
            # Use multi-turn conversation format with prefill
            contents = [
                {"role": "user", "parts": [{"text": prompt}]},
                {"role": "model", "parts": [{"text": prefill}]},
            ]
        else:
            # Simple string format
            contents = prompt

        try:
            config = GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

            completion_text = response.text
            # Prepend prefill to completion if it was used
            if prefill:
                completion_text = prefill + completion_text
            completions.append({"text": completion_text})

        except Exception as e:
            error_str = str(e).lower()

            # Check if this is a quota exhaustion (not a transient rate limit)
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
                completions.append({"text": ""})
            elif "rate" in error_str or "429" in error_str:
                # Transient rate limit - retry once
                logger.warning(f"Rate limit hit: {e}. Sleeping 10s and retrying...")
                time.sleep(10)
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                    completion_text = response.text
                    if prefill:
                        completion_text = prefill + completion_text
                    completions.append({"text": completion_text})
                except Exception as e2:
                    logger.error(f"Error after retry: {e2}")
                    completions.append({"text": ""})
            else:
                logger.error(f"Error getting completion: {e}")
                completions.append({"text": ""})

    return {"completions": completions}
