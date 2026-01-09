"""OpenAI provider for completions and embeddings.

Simplified implementation focusing on core functionality.
"""

import logging
import time

from multiview.constants import OPENAI_API_KEYS

logger = logging.getLogger(__name__)


def openai_completions(
    prompts: list[str],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    force_prefills: list[str] | None = None,
    **kwargs,
) -> dict:
    """Get OpenAI chat completions.

    Args:
        prompts: List of prompts (will be formatted as user messages)
        model_name: Model name (e.g., "gpt-4.1", "gpt-4.1-mini")
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        force_prefills: Optional list of prefill strings to force response start
        **kwargs: Additional OpenAI API parameters

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "text" key with the generated text
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install openai"
        ) from None

    # Initialize client
    api_key = (
        OPENAI_API_KEYS[0] if isinstance(OPENAI_API_KEYS, list) else OPENAI_API_KEYS
    )
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY or OPENAI_API_KEYS environment variable not set"
        )

    client = openai.OpenAI(api_key=api_key)

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
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            completion_text = response.choices[0].message.content
            # Prepend prefill to completion if it was used
            if prefill:
                completion_text = prefill + completion_text
            completions.append({"text": completion_text})

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}. Sleeping 5s and retrying...")
            time.sleep(5)
            # Retry once
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            completion_text = response.choices[0].message.content
            if prefill:
                completion_text = prefill + completion_text
            completions.append({"text": completion_text})

        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            # Return empty completion on error
            completions.append({"text": ""})

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
            - embed_query_instrs: Optional list of query instructions to prepend
            - embed_doc_instrs: Optional list of document instructions to prepend

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "vector" key with the embedding
    """
    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package required. Install with: pip install openai"
        ) from None

    # Initialize client
    api_key = (
        OPENAI_API_KEYS[0] if isinstance(OPENAI_API_KEYS, list) else OPENAI_API_KEYS
    )
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY or OPENAI_API_KEYS environment variable not set"
        )

    client = openai.OpenAI(api_key=api_key)

    # Handle embedding instructions by prepending them to prompts
    embed_query_instrs = kwargs.pop("embed_query_instrs", None)
    embed_doc_instrs = kwargs.pop("embed_doc_instrs", None)

    final_prompts = []
    for i, prompt in enumerate(prompts):
        final_prompt = prompt
        if embed_query_instrs and i < len(embed_query_instrs):
            final_prompt = embed_query_instrs[i] + final_prompt
        if embed_doc_instrs and i < len(embed_doc_instrs):
            final_prompt = embed_doc_instrs[i] + final_prompt
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

        # Extract embeddings
        completions = [{"vector": item.embedding} for item in response.data]

        return {"completions": completions}

    except openai.RateLimitError as e:
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
        completions = [{"vector": item.embedding} for item in response.data]
        return {"completions": completions}

    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        # Return empty vectors on error
        return {"completions": [{"vector": []} for _ in final_prompts]}
