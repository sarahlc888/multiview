"""Triplet selection presets.

This module defines presets for LM judges that select positive and negative examples
from a pool of candidates during triplet creation. This is used to build high-quality
training data for similarity models.
"""

from ._base import InferenceConfig

TRIPLET_SELECT_POSITIVE_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",  # Changed from Flash to Pro for JSON mode support
    prompt_template="prompts/triplet/triplet_select_positive.txt",
    parser="json",
    parser_kwargs={"annotation_key": "chosen_positive"},
    temperature=0.5,  # Lowered from 1.0 for more consistent JSON output
    max_tokens=8192,
    extra_kwargs={"response_mime_type": "application/json"},  # Force JSON mode
)

TRIPLET_SELECT_NEGATIVE_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",  # harder than positive selection
    prompt_template="prompts/triplet/triplet_select_negative.txt",
    parser="json",
    parser_kwargs={"annotation_key": "chosen_negative"},
    temperature=0.5,  # Lowered from 1.0 for more consistent JSON output
    max_tokens=8192,
    extra_kwargs={"response_mime_type": "application/json"},  # Force JSON mode
)
