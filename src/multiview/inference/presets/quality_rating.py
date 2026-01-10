"""Triplet quality rating presets.

This module defines presets for LM judges that rate triplet quality on a 1-4 scale.
"""

from ._base import InferenceConfig

# ============================================================================
# QUALITY RATING JUDGE
# ============================================================================

LMJUDGE_QUALITY_RATING_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/quality/quality_rating_gemini.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"Final\s+[Rr]ating:\s*1": 1,
            r"Final\s+[Rr]ating:\s*2": 2,
            r"Final\s+[Rr]ating:\s*3": 3,
            r"Final\s+[Rr]ating:\s*4": 4,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# QUALITY RATING JUDGE WITH ANNOTATIONS
# ============================================================================

LMJUDGE_QUALITY_RATING_WITH_ANNOTATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/quality/quality_rating_with_annotation_gemini.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"Final\s+[Rr]ating:\s*1": 1,
            r"Final\s+[Rr]ating:\s*2": 2,
            r"Final\s+[Rr]ating:\s*3": 3,
            r"Final\s+[Rr]ating:\s*4": 4,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)
