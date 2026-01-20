"""OpenAI provider for completions and embeddings.

Simplified implementation focusing on core functionality.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from tqdm import tqdm

from multiview.constants import OPENAI_API_KEYS
from multiview.inference.cost_tracker import record_usage

logger = logging.getLogger(__name__)


def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install openai"
        ) from None

    api_key = (
        OPENAI_API_KEYS[0] if isinstance(OPENAI_API_KEYS, list) else OPENAI_API_KEYS
    )
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY or OPENAI_API_KEYS environment variable not set"
        )
    return OpenAI(api_key=api_key)


def _build_responses_input(prompt: str, prefill: str | None) -> Any:
    """Build Responses API input.

    - If no prefill is provided, we pass a plain string.
    - If a prefill is provided, we pass a conversation-style list of messages so the
      model can continue from that assistant text.
    """
    if not prefill:
        return prompt
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prefill},
    ]


def _extract_usage_tokens(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if not usage:
        return None, None

    # Responses API uses (input_tokens, output_tokens); other endpoints may use
    # (prompt_tokens, completion_tokens).
    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", None)

    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", None)

    return input_tokens, output_tokens


def _extract_output_text(response: Any) -> str:
    """Best-effort extraction of assistant text from a Responses API response."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    output_items = getattr(response, "output", None)
    if not isinstance(output_items, Iterable):
        return ""

    chunks: list[str] = []
    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None)
        if not isinstance(content, Iterable):
            continue
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type not in {"output_text", "text"}:
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
    return "".join(chunks)


def _extract_first_token_top_logprobs(response: Any) -> list[dict[str, Any]] | None:
    """Best-effort: return top_logprobs for the first output token.

    Returns a list of dicts like: [{"token": "...", "logprob": -1.23}, ...]
    """
    output_items = getattr(response, "output", None)
    if not isinstance(output_items, Iterable):
        return None

    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None)
        if not isinstance(content, Iterable):
            continue

        for block in content:
            block_type = getattr(block, "type", None)
            if block_type not in {"output_text", "text"}:
                continue

            logprobs = getattr(block, "logprobs", None)
            if not logprobs:
                continue

            # SDKs may represent logprobs as a list of per-token objects/dicts.
            first = logprobs[0] if isinstance(logprobs, list) and logprobs else None
            if not first:
                continue

            top = getattr(first, "top_logprobs", None)
            if top is None and isinstance(first, dict):
                top = first.get("top_logprobs")

            if isinstance(top, list) and top:
                out: list[dict[str, Any]] = []
                for t in top:
                    if isinstance(t, dict):
                        token = t.get("token")
                        logprob = t.get("logprob")
                    else:
                        token = getattr(t, "token", None)
                        logprob = getattr(t, "logprob", None)
                    if token is None or logprob is None:
                        continue
                    out.append({"token": token, "logprob": logprob})
                return out or None

            # Fallback: if only the chosen token logprob is present
            if isinstance(first, dict) and "token" in first and "logprob" in first:
                return [{"token": first["token"], "logprob": first["logprob"]}]
            token = getattr(first, "token", None)
            logprob = getattr(first, "logprob", None)
            if token is not None and logprob is not None:
                return [{"token": token, "logprob": logprob}]

    return None


def _openai_single_completion(
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
    """Process a single OpenAI completion with retry logic.

    Args:
        client: OpenAI client instance
        prompt: Prompt text
        prefill: Optional prefill string to force response start
        model_name: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay in seconds for exponential backoff
        retry_backoff_factor: Multiplier for backoff delay
        **kwargs: Additional OpenAI API parameters

    Returns:
        Dict with "text" key containing the completion
    """
    try:
        from openai import RateLimitError
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install openai"
        ) from None

    responses_input = _build_responses_input(prompt=prompt, prefill=prefill)

    # Retry with exponential backoff
    delay = initial_retry_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model_name,
                input=responses_input,
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs,
            )
            completion_text = _extract_output_text(response)
            if prefill:
                completion_text = prefill + completion_text

            # Record usage
            input_tokens, output_tokens = _extract_usage_tokens(response)
            if input_tokens is not None and output_tokens is not None:
                record_usage(
                    model_name=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            return {"text": completion_text}

        except RateLimitError as e:
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


def openai_completions(
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
    """Get OpenAI text responses via the Responses API.

    Args:
        prompts: List of prompts
        model_name: Model name (e.g., "gpt-4.1", "gpt-4.1-mini")
        temperature: Sampling temperature
        max_tokens: Maximum output tokens to generate
        force_prefills: Optional list of prefill strings to force response start
        max_workers: Maximum concurrent API requests (default 5)
        max_retries: Maximum retry attempts per request (default 5)
        initial_retry_delay: Initial delay for exponential backoff (default 1.0s)
        retry_backoff_factor: Backoff multiplier (default 2.0)
        **kwargs: Additional OpenAI API parameters

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "text" key with the generated text
    """
    # Initialize client (shared across threads - OpenAI client is thread-safe)
    client = _get_openai_client()

    # Pair prompts with their corresponding prefills
    prompt_prefill_pairs = []
    for i, prompt in enumerate(prompts):
        prefill = None
        if force_prefills and i < len(force_prefills):
            prefill = force_prefills[i]
        prompt_prefill_pairs.append((prompt, prefill))

    # Create partial function with fixed parameters
    completion_fn = partial(
        _openai_single_completion,
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
                desc="OpenAI completions",
                total=len(prompt_prefill_pairs),
                unit="req",
            )
        )

    return {"completions": completions}


def openai_embedding_completions(
    prompts: list[str],
    model_name: str,
    **kwargs,
) -> dict:
    """Get OpenAI embeddings.

    Args:
        prompts: List of texts to embed (without instructions)
        model_name: Model name (e.g., "text-embedding-3-large")
        **kwargs: Additional parameters including:
            - instructions: Optional list of instructions to prepend

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "vector" key with the embedding
    """
    try:
        from openai import RateLimitError
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install openai"
        ) from None

    # Initialize client
    client = _get_openai_client()

    # Handle embedding instructions by prepending them to prompts
    instructions = kwargs.pop("instructions", None)

    # Strip internal/provider-level arguments that must not be forwarded to the OpenAI SDK.
    kwargs.pop("max_retries", None)
    kwargs.pop("initial_retry_delay", None)
    kwargs.pop("retry_backoff_factor", None)
    kwargs.pop("max_workers", None)

    final_prompts = []
    for i, prompt in enumerate(prompts):
        final_prompt = prompt
        if instructions and i < len(instructions):
            final_prompt = instructions[i] + final_prompt
        final_prompts.append(final_prompt)

    # Filter out parameters not supported by embeddings API
    embedding_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]
    }

    try:
        # OpenAI allows batch embedding requests
        response = client.embeddings.create(
            model=model_name,
            input=final_prompts,
            **embedding_kwargs,
        )

        # Record usage
        if hasattr(response, "usage") and response.usage:
            record_usage(
                model_name=model_name,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0,  # Embeddings don't have output tokens
            )

        # Extract embeddings
        completions = [{"vector": item.embedding} for item in response.data]

        return {"completions": completions}

    except RateLimitError as e:
        # Check if it's actually a quota issue (insufficient_quota) vs rate limit
        error_message = str(e)
        if "insufficient_quota" in error_message.lower():
            logger.error(
                f"OpenAI quota exceeded: {e}. Please check your billing details."
            )
            return {"completions": [{"vector": []} for _ in final_prompts]}

        logger.warning(f"Rate limit hit: {e}. Sleeping 5s and retrying...")
        time.sleep(5)
        # Retry once
        response = client.embeddings.create(
            model=model_name,
            input=final_prompts,
            **embedding_kwargs,
        )

        # Record usage
        if hasattr(response, "usage") and response.usage:
            record_usage(
                model_name=model_name,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0,
            )

        completions = [{"vector": item.embedding} for item in response.data]
        return {"completions": completions}

    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        # Return empty vectors on error
        return {"completions": [{"vector": []} for _ in final_prompts]}
