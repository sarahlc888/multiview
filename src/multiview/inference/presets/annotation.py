"""Annotation presets for documents.

This module contains presets for the full annotation workflow:
1. Schema Generation: Create annotation schemas from sample documents
2. Schema Application: Apply schemas to annotate individual documents
"""

from ._base import InferenceConfig

# ============================================================================
# SCHEMA GENERATION - Create annotation schemas from samples
# ============================================================================

# ----------------------------------------------------------------------------
# Category Schema Generation
# ----------------------------------------------------------------------------

CATEGORY_SCHEMA_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="prompts/schema/category_schema_generation_gemini.txt",
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)

# ----------------------------------------------------------------------------
# Tag Schema Generation
# ----------------------------------------------------------------------------

TAG_SCHEMA_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="prompts/schema/tag_schema_generation_gemini.txt",
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)

# ----------------------------------------------------------------------------
# Spurious Tag Schema Generation
# ----------------------------------------------------------------------------

SPURIOUS_TAG_SCHEMA_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="prompts/schema/spurious_tag_schema_generation_gemini.txt",
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)

# ----------------------------------------------------------------------------
# Summary Guidance Generation
# ----------------------------------------------------------------------------

SUMMARY_GUIDANCE_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="prompts/schema/summary_guidance_generation_gemini.txt",
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)

# ============================================================================
# SCHEMA APPLICATION - Annotate documents using schemas
# ============================================================================

# ----------------------------------------------------------------------------
# Category Classification
# ----------------------------------------------------------------------------

CATEGORY_CLASSIFY_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/annotation/category_classify_gemini.txt",
    parser="json",
    temperature=1.0,  # Use Gemini API default
    max_tokens=2048,
    extra_kwargs={"top_p": 0.95, "top_k": 64},  # Gemini API defaults
)

# ----------------------------------------------------------------------------
# Tag Application
# ----------------------------------------------------------------------------

TAG_APPLY_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/annotation/tag_apply_gemini.txt",
    parser="json",
    temperature=1.0,  # Use Gemini API default
    max_tokens=2048,
    extra_kwargs={"top_p": 0.95, "top_k": 64},  # Gemini API defaults
)

# ----------------------------------------------------------------------------
# Summary Generation
# ----------------------------------------------------------------------------

SUMMARY_GENERATE_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/annotation/summary_generate_gemini.txt",
    parser="json",
    temperature=1.0,  # Use Gemini API default
    max_tokens=8192,
    extra_kwargs={"top_p": 0.95, "top_k": 64},  # Gemini API defaults
)
