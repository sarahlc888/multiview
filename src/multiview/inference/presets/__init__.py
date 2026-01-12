"""Inference configuration and presets.

This package defines InferenceConfig and provides preset configurations
for common models/providers. Presets are organized by category for easy
browsing and maintenance.
"""

# Base configuration class
from ._base import InferenceConfig

# Annotation (schema generation + application)
from .annotation import (
    CATEGORY_CLASSIFY_GEMINI,
    CATEGORY_SCHEMA_GENERATION_GEMINI,
    SPURIOUS_TAG_SCHEMA_GENERATION_GEMINI,
    SUMMARY_GENERATE_GEMINI,
    SUMMARY_GUIDANCE_GENERATION_GEMINI,
    TAG_APPLY_GEMINI,
    TAG_SCHEMA_GENERATION_GEMINI,
)

# Basic LM model presets
from .basic_lm_models import (
    CLAUDE_HAIKU,
    CLAUDE_SONNET,
    GEMINI_FLASH,
    GEMINI_PRO,
    GPT41,
    GPT41_MINI,
)

# Embedding presets
from .embeddings import (
    EMBED_PLAINTEXT_HFAPI,
    HF_QWEN3_EMBEDDING_4B,
    HF_QWEN3_EMBEDDING_8B,
    OPENAI_EMBEDDING_LARGE,
    OPENAI_EMBEDDING_SMALL,
)

# LM judge presets - pairwise
from .lm_judge_pair import (
    LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI,
    LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI,
    LMJUDGE_PAIR_WITH_ANNOTATION_BINARYHARD_GEMINI,
    LMJUDGE_PAIR_WITH_ANNOTATION_LIKERTHARD_GEMINI,
)

# LM judge presets - triplet
from .lm_judge_triplet import (
    LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI,
    LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_FLASH,
    LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_PRO,
    LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI,
    LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI_FLASH,
    LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI_PRO,
    LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI,
    LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_FLASH,
    LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_PRO,
)

# Pairwise similarity hint generation
from .pairwise_sim_hint import PAIRWISE_SIM_HINT_GENERATION_GEMINI

# Quality rating presets
from .quality_rating import (
    LMJUDGE_QUALITY_RATING_GEMINI,
    LMJUDGE_QUALITY_RATING_WITH_ANNOTATION_GEMINI,
)

# Document rewriting presets
from .rewrite import REWRITE_PLAINTEXT_FREEFORM_GEMINI

# Triplet selection
from .triplet_selection import (
    TRIPLET_SELECT_NEGATIVE_GEMINI,
    TRIPLET_SELECT_POSITIVE_GEMINI,
)

# Query expansion (for evaluation)
from .query_expansion import (
    QUERY_EXPANSION_SUMMARY_CLAUDE_HAIKU,
    QUERY_EXPANSION_SUMMARY_CLAUDE_SONNET,
    QUERY_EXPANSION_SUMMARY_GEMINI,
    QUERY_EXPANSION_SUMMARY_GEMINI_FLASH,
    QUERY_EXPANSION_SUMMARY_GEMINI_PRO,
)

# ============================================================================
# PRESET REGISTRY
# ============================================================================

PRESET_REGISTRY = {
    # Basic embeddings
    "openai_embedding_large": OPENAI_EMBEDDING_LARGE,
    "openai_embedding_small": OPENAI_EMBEDDING_SMALL,
    "hf_qwen3_embedding_8b": HF_QWEN3_EMBEDDING_8B,
    "hf_qwen3_embedding_4b": HF_QWEN3_EMBEDDING_4B,
    # Basic LM models
    "claude_sonnet": CLAUDE_SONNET,
    "claude_haiku": CLAUDE_HAIKU,
    "gpt41": GPT41,
    "gpt41_mini": GPT41_MINI,
    "gemini_flash": GEMINI_FLASH,
    "gemini_pro": GEMINI_PRO,
    # Specialized task presets
    "embed_plaintext_hfapi": EMBED_PLAINTEXT_HFAPI,
    "rewrite_plaintext_freeform_gemini": REWRITE_PLAINTEXT_FREEFORM_GEMINI,
    "lmjudge_pair_plaintext_likerthard_gemini": LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI,
    "lmjudge_triplet_plaintext_binaryhard_gemini": LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI,
    "lmjudge_pair_norewrite_binaryhard_gemini": LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI,
    # LM judge presets with annotations
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini": LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI,
    "lmjudge_triplet_with_annotation_gemini": LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI,
    "lmjudge_pair_with_annotation_likerthard_gemini": LMJUDGE_PAIR_WITH_ANNOTATION_LIKERTHARD_GEMINI,
    "lmjudge_pair_with_annotation_binaryhard_gemini": LMJUDGE_PAIR_WITH_ANNOTATION_BINARYHARD_GEMINI,
    # LM judge triplet presets - Gemini 2.5 Flash (full)
    "lmjudge_triplet_plaintext_binaryhard_gemini_flash": LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_FLASH,
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini_flash": LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI_FLASH,
    "lmjudge_triplet_with_annotation_gemini_flash": LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_FLASH,
    # LM judge triplet presets - Gemini 2.5 Pro
    "lmjudge_triplet_plaintext_binaryhard_gemini_pro": LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_PRO,
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini_pro": LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI_PRO,
    "lmjudge_triplet_with_annotation_gemini_pro": LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_PRO,
    # Quality rating presets
    "lmjudge_quality_rating_gemini": LMJUDGE_QUALITY_RATING_GEMINI,
    "lmjudge_quality_rating_with_annotation_gemini": LMJUDGE_QUALITY_RATING_WITH_ANNOTATION_GEMINI,
    # Multi-faceted annotation presets
    "category_schema_generation_gemini": CATEGORY_SCHEMA_GENERATION_GEMINI,
    "tag_schema_generation_gemini": TAG_SCHEMA_GENERATION_GEMINI,
    "spurious_tag_schema_generation_gemini": SPURIOUS_TAG_SCHEMA_GENERATION_GEMINI,
    "summary_guidance_generation_gemini": SUMMARY_GUIDANCE_GENERATION_GEMINI,
    "pairwise_sim_hint_generation_gemini": PAIRWISE_SIM_HINT_GENERATION_GEMINI,
    "category_classify_gemini": CATEGORY_CLASSIFY_GEMINI,
    "tag_apply_gemini": TAG_APPLY_GEMINI,
    "summary_generate_gemini": SUMMARY_GENERATE_GEMINI,
    "triplet_select_positive_gemini": TRIPLET_SELECT_POSITIVE_GEMINI,
    "triplet_select_negative_gemini": TRIPLET_SELECT_NEGATIVE_GEMINI,
    # Query expansion presets
    "query_expansion_summary_gemini": QUERY_EXPANSION_SUMMARY_GEMINI,
    "query_expansion_summary_gemini_flash": QUERY_EXPANSION_SUMMARY_GEMINI_FLASH,
    "query_expansion_summary_gemini_pro": QUERY_EXPANSION_SUMMARY_GEMINI_PRO,
    "query_expansion_summary_claude_haiku": QUERY_EXPANSION_SUMMARY_CLAUDE_HAIKU,
    "query_expansion_summary_claude_sonnet": QUERY_EXPANSION_SUMMARY_CLAUDE_SONNET,
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


# Export all public symbols
__all__ = [
    # Base class
    "InferenceConfig",
    # Registry functions
    "get_preset",
    "list_presets",
    "PRESET_REGISTRY",
    # All preset constants (for direct imports)
    "OPENAI_EMBEDDING_LARGE",
    "OPENAI_EMBEDDING_SMALL",
    "HF_QWEN3_EMBEDDING_8B",
    "HF_QWEN3_EMBEDDING_4B",
    "CLAUDE_SONNET",
    "CLAUDE_HAIKU",
    "GPT41",
    "GPT41_MINI",
    "GEMINI_FLASH",
    "GEMINI_PRO",
    "EMBED_PLAINTEXT_HFAPI",
    "REWRITE_PLAINTEXT_FREEFORM_GEMINI",
    "LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI",
    "LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI",
    "LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI",
    "LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI",
    "LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI",
    "LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_FLASH",
    "LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI_FLASH",
    "LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_FLASH",
    "LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_PRO",
    "LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI_PRO",
    "LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_PRO",
    "LMJUDGE_PAIR_WITH_ANNOTATION_LIKERTHARD_GEMINI",
    "LMJUDGE_PAIR_WITH_ANNOTATION_BINARYHARD_GEMINI",
    "LMJUDGE_QUALITY_RATING_GEMINI",
    "LMJUDGE_QUALITY_RATING_WITH_ANNOTATION_GEMINI",
    "CATEGORY_SCHEMA_GENERATION_GEMINI",
    "TAG_SCHEMA_GENERATION_GEMINI",
    "SPURIOUS_TAG_SCHEMA_GENERATION_GEMINI",
    "SUMMARY_GUIDANCE_GENERATION_GEMINI",
    "PAIRWISE_SIM_HINT_GENERATION_GEMINI",
    "CATEGORY_CLASSIFY_GEMINI",
    "TAG_APPLY_GEMINI",
    "SUMMARY_GENERATE_GEMINI",
    "TRIPLET_SELECT_POSITIVE_GEMINI",
    "TRIPLET_SELECT_NEGATIVE_GEMINI",
    "QUERY_EXPANSION_SUMMARY_GEMINI",
    "QUERY_EXPANSION_SUMMARY_GEMINI_FLASH",
    "QUERY_EXPANSION_SUMMARY_GEMINI_PRO",
    "QUERY_EXPANSION_SUMMARY_CLAUDE_HAIKU",
    "QUERY_EXPANSION_SUMMARY_CLAUDE_SONNET",
]
