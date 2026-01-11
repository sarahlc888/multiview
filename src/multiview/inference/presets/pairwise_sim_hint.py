"""Pairwise similarity hint generation presets.

This module defines presets for generating pairwise similarity hints (detailed descriptions
of similarity criteria) from sample documents.
"""

from ._base import InferenceConfig

PAIRWISE_SIM_HINT_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="prompts/pairwise_sim_hint_generation.txt",
    # Prompt uses a "---FINAL OUTPUT---" delimiter; parse that centrally instead of
    # ad-hoc post-processing at call sites.
    parser="delimiter",
    parser_kwargs={"delimiter": "---FINAL OUTPUT---"},
    temperature=0.0,
    max_tokens=8192,
)
