"""Basic LM model presets.

This module defines baseline configurations for general-purpose language models
from various providers (Anthropic, OpenAI, Google Gemini).
"""

from ._base import InferenceConfig

# ============================================================================
# ANTHROPIC MODELS
# ============================================================================

CLAUDE_SONNET = InferenceConfig(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

CLAUDE_HAIKU = InferenceConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# OPENAI MODELS
# ============================================================================

GPT41 = InferenceConfig(
    provider="openai",
    model_name="gpt-4.1",  # Updated to GPT-4.1 (outperforms gpt-4o)
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

GPT41_MINI = InferenceConfig(
    provider="openai",
    model_name="gpt-4.1-mini",  # Updated to GPT-4.1-mini (outperforms gpt-4o-mini)
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# GEMINI MODELS
# ============================================================================

GEMINI_FLASH = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",  # Lite model with higher free tier limits
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

GEMINI_PRO = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)
