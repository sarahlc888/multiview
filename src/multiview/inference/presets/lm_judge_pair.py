"""Pairwise similarity judge presets.

This module defines presets for LM judges that compare two documents and output
similarity scores.
"""

from ._base import InferenceConfig

# ============================================================================
# LIKERT SCALE JUDGE (1-5)
# ============================================================================

LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/lm_judge/pair_plaintext_likerthard.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# BINARY JUDGE (0=same, 1=different)
# ============================================================================

LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/lm_judge/pair_norewrite_binaryhard.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            # Model outputs 0 for "same", return 0; outputs 1 for "different", return 1
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
        }
    },
    temperature=0.0,
    max_tokens=2048,
)

# ============================================================================
# LIKERT SCALE JUDGE WITH ANNOTATIONS (1-5)
# ============================================================================

LMJUDGE_PAIR_WITH_ANNOTATION_LIKERTHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/lm_judge/pair_with_annotation_likerthard.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# BINARY JUDGE WITH ANNOTATIONS (0=same, 1=different)
# ============================================================================

LMJUDGE_PAIR_WITH_ANNOTATION_BINARYHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/lm_judge/pair_with_annotation_binaryhard.txt",
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            # Model outputs 0 for "same", return 0; outputs 1 for "different", return 1
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
        }
    },
    temperature=0.0,
    max_tokens=2048,
)
