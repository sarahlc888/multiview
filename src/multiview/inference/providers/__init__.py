"""Provider implementations for different LM/embedding APIs.

Each provider module implements completion functions that:
- Take a list of prompts and model parameters
- Return a dict with "completions" key containing list of results
- Handle retries and rate limiting internally
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def get_completion_fn(provider: str) -> Callable:
    """Get the completion function for a given provider.

    Args:
        provider: Provider name (e.g., "openai", "openai_embedding", "anthropic",
                  "hf_embedding", "hf_chat", "gemini", "hf_local_reranker",
                  "hf_local_contextual_reranker", "voyage_reranker",
                  "voyage_embedding", "hf_local_hidden_state")

    Returns:
        Completion function that takes prompts and returns completions

    Raises:
        ValueError: If provider is unknown
        ImportError: If required packages are not installed
    """
    # Registry of provider functions with lazy imports
    provider_registry = {
        "openai": (
            lambda: __import__(
                "multiview.inference.providers.openai", fromlist=["openai_completions"]
            ).openai_completions,
            "openai package not installed. Install with: pip install openai",
        ),
        "openai_embedding": (
            lambda: __import__(
                "multiview.inference.providers.openai",
                fromlist=["openai_embedding_completions"],
            ).openai_embedding_completions,
            "openai package not installed. Install with: pip install openai",
        ),
        "anthropic": (
            lambda: __import__(
                "multiview.inference.providers.anthropic",
                fromlist=["anthropic_completions"],
            ).anthropic_completions,
            "anthropic package not installed. Install with: pip install anthropic",
        ),
        "hf_embedding": (
            lambda: __import__(
                "multiview.inference.providers.hf_api",
                fromlist=["hf_embedding_completions"],
            ).hf_embedding_completions,
            "huggingface_hub package not installed. Install with: pip install huggingface_hub",
        ),
        "hf_chat": (
            lambda: __import__(
                "multiview.inference.providers.hf_api",
                fromlist=["hf_chat_completions"],
            ).hf_chat_completions,
            "openai package not installed. Install with: pip install openai",
        ),
        "gemini": (
            lambda: __import__(
                "multiview.inference.providers.gemini", fromlist=["gemini_completions"]
            ).gemini_completions,
            "google-genai package not installed. Install with: pip install google-genai",
        ),
        "gemini_embedding": (
            lambda: __import__(
                "multiview.inference.providers.gemini",
                fromlist=["gemini_embedding_completions"],
            ).gemini_embedding_completions,
            "google-genai package not installed. Install with: pip install google-genai",
        ),
        "hf_local_hidden_state": (
            lambda: __import__(
                "multiview.inference.providers.hf_local",
                fromlist=["hf_local_hidden_state_completions"],
            ).hf_local_hidden_state_completions,
            "transformers, torch, and numpy packages required. Install with: pip install transformers torch numpy",
        ),
        "hf_local_reranker": (
            lambda: __import__(
                "multiview.inference.providers.hf_local",
                fromlist=["hf_local_reranker_completions"],
            ).hf_local_reranker_completions,
            "transformers and torch packages required. Install with: pip install transformers torch",
        ),
        "hf_local_contextual_reranker": (
            lambda: __import__(
                "multiview.inference.providers.hf_local",
                fromlist=["hf_local_contextual_reranker_completions"],
            ).hf_local_contextual_reranker_completions,
            "transformers and torch packages required. Install with: pip install transformers torch",
        ),
        "hf_local_colbert": (
            lambda: __import__(
                "multiview.inference.providers.hf_local_colbert",
                fromlist=["hf_local_colbert_completions"],
            ).hf_local_colbert_completions,
            "pylate, transformers, and torch packages required. Install with: pip install pylate transformers torch",
        ),
        "voyage_reranker": (
            lambda: __import__(
                "multiview.inference.providers.voyage",
                fromlist=["voyage_reranker_completions"],
            ).voyage_reranker_completions,
            "voyageai package not installed. Install with: pip install voyageai",
        ),
        "voyage_embedding": (
            lambda: __import__(
                "multiview.inference.providers.voyage",
                fromlist=["voyage_embedding_completions"],
            ).voyage_embedding_completions,
            "voyageai package not installed. Install with: pip install voyageai",
        ),
        "voyage_multimodal_embedding": (
            lambda: __import__(
                "multiview.inference.providers.voyage",
                fromlist=["voyage_multimodal_embedding_completions"],
            ).voyage_multimodal_embedding_completions,
            "voyageai and Pillow packages required. Install with: pip install voyageai Pillow",
        ),
        "pseudologit": (
            lambda: __import__(
                "multiview.inference.providers.pseudologit",
                fromlist=["pseudologit_completions"],
            ).pseudologit_completions,
            "pseudologit provider requires API provider dependencies (openai, google-genai, etc.)",
        ),
    }

    if provider not in provider_registry:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available providers: {', '.join(sorted(provider_registry.keys()))}"
        )

    loader, error_msg = provider_registry[provider]
    try:
        return loader()
    except ImportError as e:
        logger.error(error_msg)
        raise e
