"""Simple API cost tracking.

Tracks API usage across all providers by model name and token counts.

For pricing information, see:
- Gemini API: https://ai.google.dev/gemini-api/docs/pricing
- OpenAI API: https://openai.com/api/pricing/
- Anthropic API: https://docs.anthropic.com/en/docs/about-claude/pricing
"""

from collections import defaultdict

# Pricing data (per 1M tokens, in USD)
# Updated January 2026
MODEL_PRICING = {
    # Gemini models
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash-exp": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    # OpenAI models
    "gpt-4.1": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    # Anthropic models
    "claude-opus-4.5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


class CostTracker:
    """Simple tracker for API usage."""

    def __init__(self):
        """Initialize empty usage tracker."""
        self.usage: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        )
        self.cache_hits: dict[str, dict[str, int]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "requests": 0}
        )

    def record(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record usage for a single API call.

        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        self.usage[model_name]["input_tokens"] += input_tokens
        self.usage[model_name]["output_tokens"] += output_tokens
        self.usage[model_name]["requests"] += 1

    def record_cache_hit(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Record a cache hit (API call avoided).

        Args:
            model_name: Name of the model that would have been used
            input_tokens: Number of input tokens that would have been used
            output_tokens: Number of output tokens that would have been used
        """
        self.cache_hits[model_name]["input_tokens"] += input_tokens
        self.cache_hits[model_name]["output_tokens"] += output_tokens
        self.cache_hits[model_name]["requests"] += 1

    def get_summary(self) -> str:
        """Get formatted summary of API usage with cost calculation.

        Returns:
            Formatted string with usage statistics and costs
        """
        if not self.usage and not self.cache_hits:
            return "No API usage recorded"

        lines = [
            "",
            "=" * 80,
            "API USAGE SUMMARY",
            "=" * 80,
        ]

        total_input = 0
        total_output = 0
        total_requests = 0
        total_cost = 0.0

        for model_name in sorted(self.usage.keys()):
            stats = self.usage[model_name]
            input_tokens = stats["input_tokens"]
            output_tokens = stats["output_tokens"]
            requests = stats["requests"]

            total_input += input_tokens
            total_output += output_tokens
            total_requests += requests

            # Calculate cost for this model
            cost = 0.0
            if model_name in MODEL_PRICING:
                pricing = MODEL_PRICING[model_name]
                cost = (input_tokens / 1_000_000) * pricing["input"] + (
                    output_tokens / 1_000_000
                ) * pricing["output"]
                total_cost += cost

            lines.append(f"\n{model_name}:")
            lines.append(f"  Requests: {requests:,}")
            lines.append(f"  Input tokens: {input_tokens:,}")
            lines.append(f"  Output tokens: {output_tokens:,}")
            lines.append(f"  Total tokens: {input_tokens + output_tokens:,}")
            if model_name in MODEL_PRICING:
                lines.append(f"  Cost: ${cost:.4f}")
            else:
                lines.append(
                    f"  Cost: Unknown (pricing not available for {model_name})"
                )

        lines.append("\nTOTAL ACROSS ALL MODELS:")
        lines.append(f"  Requests: {total_requests:,}")
        lines.append(f"  Input tokens: {total_input:,}")
        lines.append(f"  Output tokens: {total_output:,}")
        lines.append(f"  Total tokens: {total_input + total_output:,}")
        lines.append(f"  Total cost: ${total_cost:.4f}")

        # Add cache savings section if there are cache hits
        if self.cache_hits:
            lines.append("")
            lines.append("CACHE SAVINGS (from disk cache):")
            lines.append("-" * 80)

            cache_saved_cost = 0.0
            total_cache_requests = 0

            for model_name in sorted(self.cache_hits.keys()):
                stats = self.cache_hits[model_name]
                input_tokens = stats["input_tokens"]
                output_tokens = stats["output_tokens"]
                requests = stats["requests"]
                total_cache_requests += requests

                # Calculate saved cost
                saved_cost = 0.0
                if model_name in MODEL_PRICING:
                    pricing = MODEL_PRICING[model_name]
                    saved_cost = (input_tokens / 1_000_000) * pricing["input"] + (
                        output_tokens / 1_000_000
                    ) * pricing["output"]
                    cache_saved_cost += saved_cost

                lines.append(f"\n{model_name}:")
                lines.append(f"  Cached requests: {requests:,}")
                lines.append(f"  Saved tokens: {input_tokens + output_tokens:,}")
                if model_name in MODEL_PRICING:
                    lines.append(f"  Cost saved: ${saved_cost:.4f}")

            lines.append("\nTOTAL CACHE SAVINGS:")
            lines.append(f"  Cached requests: {total_cache_requests:,}")
            lines.append(f"  Total saved: ${cache_saved_cost:.4f}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def reset(self):
        """Clear all usage data."""
        self.usage.clear()
        self.cache_hits.clear()


# Global cost tracker instance
_global_tracker = CostTracker()


def record_usage(model_name: str, input_tokens: int = 0, output_tokens: int = 0):
    """Record usage to global tracker.

    Args:
        model_name: Name of the model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """
    _global_tracker.record(model_name, input_tokens, output_tokens)


def record_cache_hit(model_name: str, input_tokens: int = 0, output_tokens: int = 0):
    """Record cache hit to global tracker.

    Args:
        model_name: Name of the model that would have been used
        input_tokens: Number of input tokens that would have been used
        output_tokens: Number of output tokens that would have been used
    """
    _global_tracker.record_cache_hit(model_name, input_tokens, output_tokens)


def print_summary():
    """Print usage summary using logger (appears in both console and log file)."""
    import logging

    logger = logging.getLogger(__name__)
    summary = _global_tracker.get_summary()
    # Use logger instead of print() so it appears in both console and log file
    # Log the entire summary as one message to preserve formatting
    logger.info("\n" + summary)
