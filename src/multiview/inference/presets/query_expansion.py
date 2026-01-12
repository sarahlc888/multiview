"""Query expansion presets for evaluation.

These presets define different LM models for generating summaries during
query expansion evaluation. All use the same prompt template but different
models for cost/quality tradeoffs.
"""

from __future__ import annotations

from ._base import InferenceConfig

# Gemini Flash Lite - Fast and cheap (default)
QUERY_EXPANSION_SUMMARY_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template="prompts/eval/query_expansion_summary.txt",
    parser="json",
    temperature=0.0,  # Deterministic for evaluation
    max_tokens=2048,
)

# Gemini Flash - Higher quality than lite
QUERY_EXPANSION_SUMMARY_GEMINI_FLASH = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash",
    prompt_template="prompts/eval/query_expansion_summary.txt",
    parser="json",
    temperature=0.0,
    max_tokens=2048,
)

# Gemini Pro - Highest quality
QUERY_EXPANSION_SUMMARY_GEMINI_PRO = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="prompts/eval/query_expansion_summary.txt",
    parser="json",
    temperature=0.0,
    max_tokens=2048,
)

# Claude Haiku - Fast and cheap alternative
QUERY_EXPANSION_SUMMARY_CLAUDE_HAIKU = InferenceConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    prompt_template="prompts/eval/query_expansion_summary.txt",
    parser="json",
    temperature=0.0,
    max_tokens=2048,
)

# Claude Sonnet - Higher quality alternative
QUERY_EXPANSION_SUMMARY_CLAUDE_SONNET = InferenceConfig(
    provider="anthropic",
    model_name="claude-3-7-sonnet-20250219",
    prompt_template="prompts/eval/query_expansion_summary.txt",
    parser="json",
    temperature=0.0,
    max_tokens=2048,
)

__all__ = [
    "QUERY_EXPANSION_SUMMARY_GEMINI",
    "QUERY_EXPANSION_SUMMARY_GEMINI_FLASH",
    "QUERY_EXPANSION_SUMMARY_GEMINI_PRO",
    "QUERY_EXPANSION_SUMMARY_CLAUDE_HAIKU",
    "QUERY_EXPANSION_SUMMARY_CLAUDE_SONNET",
]
