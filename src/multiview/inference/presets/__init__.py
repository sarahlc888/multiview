"""Inference configuration and presets.

This module defines InferenceConfig and provides preset configurations
for common models/providers via the PRESET_REGISTRY.

All preset configurations are organized by category in separate modules:
- bm25: BM25 lexical retrieval
- embeddings: Embedding models (OpenAI, Voyage AI, HuggingFace, ColBERT)
- language_models: Basic LM models (Claude, GPT, Gemini)
- lm_judge: LM judge configurations (pairwise, triplet, quality)
- annotation: Annotation schema generation and application
- evaluation: Evaluation presets (document summarization, query relevance, triplet selection)
- rerankers: Reranker models
- specialized: Specialized methods (in-one-word, pseudologit)

Usage:
    from multiview.inference.presets import get_preset

    # Get a preset by name
    config = get_preset("openai_embedding_large")

    # List all available presets
    from multiview.inference.presets import list_presets
    presets = list_presets()
"""

from __future__ import annotations

import logging

from ._base import InferenceConfig
from .annotation import ANNOTATION_PRESETS
from .bm25 import BM25_PRESETS
from .embeddings import CRITERION_AWARE_EMBED_INSTR, EMBEDDING_PRESETS
from .evaluation import EVALUATION_PRESETS
from .language_models import LANGUAGE_MODEL_PRESETS
from .lm_judge import LM_JUDGE_PRESETS
from .rerankers import RERANKER_PRESETS
from .specialized import SPECIALIZED_PRESETS

logger = logging.getLogger(__name__)

# GPU availability cache (check once per session)
_GPU_AVAILABLE: bool | None = None

# Providers that require local GPU/hardware
GPU_REQUIRED_PROVIDERS = {
    "hf_local_hidden_state",
    "hf_local_reranker",
    "hf_local_colbert",
}

# ============================================================================
# PRESET REGISTRY
# ============================================================================

# Combine all preset dictionaries into a single registry
PRESET_REGISTRY = {
    **BM25_PRESETS,
    **EMBEDDING_PRESETS,
    **LANGUAGE_MODEL_PRESETS,
    **LM_JUDGE_PRESETS,
    **ANNOTATION_PRESETS,
    **EVALUATION_PRESETS,
    **RERANKER_PRESETS,
    **SPECIALIZED_PRESETS,
}


def get_preset(name: str) -> InferenceConfig:
    """Get a preset configuration by name.

    Args:
        name: Preset name (e.g., "openai_embedding_large", "lmjudge_pair_plaintext_likerthard_gemini")

    Returns:
        InferenceConfig for the preset

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESET_REGISTRY:
        raise ValueError(
            f"Unknown preset: {name}. "
            f"Available presets: {sorted(PRESET_REGISTRY.keys())}"
        )
    return PRESET_REGISTRY[name]


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        Sorted list of preset names
    """
    return sorted(PRESET_REGISTRY.keys())


def is_gpu_available() -> bool:
    """Check if GPU is available for local model inference.

    This function caches the result for the session to avoid repeated checks.

    Returns:
        True if GPU (CUDA) is available, False otherwise
    """
    global _GPU_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    try:
        import torch

        _GPU_AVAILABLE = torch.cuda.is_available()
        if _GPU_AVAILABLE:
            logger.info("GPU detected: CUDA is available")
        else:
            logger.info("No GPU detected: CUDA is not available")
    except ImportError:
        logger.warning(
            "PyTorch not installed - assuming no GPU available. "
            "Install with: pip install torch"
        )
        _GPU_AVAILABLE = False

    return _GPU_AVAILABLE


def preset_requires_gpu(preset_name: str) -> bool:
    """Check if a preset requires GPU for inference.

    Args:
        preset_name: Name of the preset to check

    Returns:
        True if the preset requires GPU, False otherwise

    Raises:
        ValueError: If preset name is not found
    """
    config = get_preset(preset_name)
    return config.provider in GPU_REQUIRED_PROVIDERS


# Export all public symbols
__all__ = [
    "InferenceConfig",
    "get_preset",
    "list_presets",
    "is_gpu_available",
    "preset_requires_gpu",
    "PRESET_REGISTRY",
    "GPU_REQUIRED_PROVIDERS",
    "CRITERION_AWARE_EMBED_INSTR",
]
