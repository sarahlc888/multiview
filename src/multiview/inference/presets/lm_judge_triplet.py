"""Triplet similarity judge presets.

This module defines presets for LM judges that compare three documents and determine
which of two documents is more similar to an anchor document.
"""

from ._base import InferenceConfig

# ============================================================================
# TRIPLET COMPARISON JUDGE (a vs b vs c)
# ============================================================================

LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"Final\s+[Aa]nswer:\s*\(b\)": 1,
            r"Final\s+[Aa]nswer:\s*\(c\)": -1,
            r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# TRIPLET COMPARISON JUDGE WITH ANNOTATIONS (a vs b vs c)
# ============================================================================

LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"Final\s+[Aa]nswer:\s*\(b\)": 1,
            r"Final\s+[Aa]nswer:\s*\(c\)": -1,
            r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# Alias with consistent naming
LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_WITH_ANNOTATION_GEMINI = (
    LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI
)
