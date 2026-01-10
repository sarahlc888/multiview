"""Triplet selection presets.

This module defines presets for LM judges that select positive and negative examples
from a pool of candidates during triplet creation. This is used to build high-quality
training data for similarity models.
"""

from ._base import InferenceConfig

TRIPLET_SELECTION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash",
    prompt_template="prompts/triplet/triplet_selection_gemini.txt",
    parser="text",
    temperature=1.0,
    max_tokens=8192,
)
