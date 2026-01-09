"""Provider implementations for different LM/embedding APIs.

Each provider module implements completion functions that:
- Take a list of prompts and model parameters
- Return a dict with "completions" key containing list of results
- Handle retries and rate limiting internally
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def get_completion_fn(provider: str, is_embedding: bool = False) -> Callable:
    """Get the completion function for a given provider.

    Args:
        provider: Provider name ("openai", "anthropic", "hf_api")
        is_embedding: If True, return embedding function; else text completion

    Returns:
        Completion function that takes prompts and returns completions

    Raises:
        ValueError: If provider is unknown
        ImportError: If required packages are not installed
    """
    if provider == "openai":
        if is_embedding:
            try:
                from .openai import openai_embedding_completions

                return openai_embedding_completions
            except ImportError as e:
                logger.error(
                    "openai package not installed. Install with: pip install openai"
                )
                raise e
        else:
            try:
                from .openai import openai_completions

                return openai_completions
            except ImportError as e:
                logger.error(
                    "openai package not installed. Install with: pip install openai"
                )
                raise e

    elif provider == "anthropic":
        if is_embedding:
            raise ValueError("Anthropic does not provide embedding models")
        try:
            from .anthropic import anthropic_completions

            return anthropic_completions
        except ImportError as e:
            logger.error(
                "anthropic package not installed. Install with: pip install anthropic"
            )
            raise e

    elif provider == "hf_api":
        if not is_embedding:
            raise ValueError("hf_api provider currently only supports embeddings")
        try:
            from .hf_api import hf_embedding_completions

            return hf_embedding_completions
        except ImportError as e:
            logger.error(
                "huggingface_hub package not installed. Install with: pip install huggingface_hub"
            )
            raise e

    elif provider == "gemini":
        if is_embedding:
            raise ValueError(
                "Gemini does not provide embedding models (use OpenAI or HF API)"
            )
        try:
            from .gemini import gemini_completions

            return gemini_completions
        except ImportError as e:
            logger.error(
                "google-genai package not installed. Install with: pip install google-genai"
            )
            raise e

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: openai, anthropic, hf_api, gemini"
        )
