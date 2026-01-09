"""Base configuration class for inference.

This module defines the InferenceConfig dataclass used throughout the multiview
inference system.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class InferenceConfig:
    """Configuration for inference with LMs or embedding models.

    This replaces the old YAML-based annotator configs with a type-safe
    Python dataclass that supports the same functionality.
    """

    # Provider and model
    provider: str  # "openai", "anthropic", "hf_api"
    model_name: str

    # Prompt templates
    # Can be inline strings or paths to template files
    prompt_template: str
    embed_query_instr_template: str | None = None
    embed_doc_instr_template: str | None = None
    force_prefill_template: str | None = None

    # Model parameters
    temperature: float = 0.0
    max_tokens: int | None = None

    # Concurrency settings
    max_workers: int = 5  # Max concurrent API requests
    max_retries: int = 5  # Max retries per request
    initial_retry_delay: float = 1.0  # Initial backoff delay (seconds)
    retry_backoff_factor: float = 2.0  # Backoff multiplier

    # Model type
    is_embedding: bool = False

    # Output parsing
    parser: str = "text"  # "json", "vector", "text"
    parser_kwargs: dict | None = None

    # Additional kwargs to pass to completion function
    extra_kwargs: dict = field(default_factory=dict)

    def with_overrides(self, **kwargs) -> InferenceConfig:
        """Return a new config with overrides applied.

        Args:
            **kwargs: Fields to override

        Returns:
            New InferenceConfig with overrides applied
        """
        return replace(self, **kwargs)

    def to_completion_kwargs(self) -> dict:
        """Convert config to kwargs for completion functions."""
        kwargs = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_workers": self.max_workers,
            "max_retries": self.max_retries,
            "initial_retry_delay": self.initial_retry_delay,
            "retry_backoff_factor": self.retry_backoff_factor,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        # Merge in any extra kwargs
        kwargs.update(self.extra_kwargs)

        return kwargs
