"""Document rewriting presets.

This module defines presets for rewriting/summarizing documents based on criteria.
"""

from ._base import InferenceConfig

REWRITE_PLAINTEXT_FREEFORM_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/rewrite/rewrite_plaintext_freeform.txt",
    parser="json",
    parser_kwargs={"annotation_key": "summary"},
    temperature=0.0,
    max_tokens=2048,
)
