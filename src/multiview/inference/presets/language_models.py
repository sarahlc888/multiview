"""Basic language model presets."""

from __future__ import annotations

from ._base import InferenceConfig

LANGUAGE_MODEL_PRESETS = {
    # ========================================================================
    # Anthropic Models
    # ========================================================================
    "claude_sonnet": InferenceConfig(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        prompt_template="{document}",
        parser="json",
        temperature=0.7,
        max_tokens=4096,
    ),
    "claude_haiku": InferenceConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        prompt_template="{document}",
        parser="json",
        temperature=0.7,
        max_tokens=4096,
    ),
    # ========================================================================
    # OpenAI Models
    # ========================================================================
    "gpt51_mini": InferenceConfig(
        provider="openai",
        model_name="gpt-5.1-mini",
        prompt_template="{document}",
        parser="json",
        temperature=0.7,
        max_tokens=4096,
    ),
    # ========================================================================
    # Gemini Models
    # ========================================================================
    "gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="{document}",
        parser="json",
        temperature=0.7,
        max_tokens=4096,
    ),
    "gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="{document}",
        parser="json",
        temperature=0.7,
        max_tokens=4096,
    ),
}
