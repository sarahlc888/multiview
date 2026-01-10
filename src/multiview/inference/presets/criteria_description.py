"""Criteria description generation presets.

This module defines presets for generating detailed descriptions of similarity criteria
from sample documents.
"""

from ._base import InferenceConfig

CRITERIA_DESCRIPTION_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="prompts/criteria_description_generation_gemini.txt",
    # Prompt uses a "---FINAL OUTPUT---" delimiter; parse that centrally instead of
    # ad-hoc post-processing at call sites.
    parser="delimiter",
    parser_kwargs={"delimiter": "---FINAL OUTPUT---"},
    temperature=0.0,
    max_tokens=8192,
)
