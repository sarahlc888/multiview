"""Tests for inference functionality.

Tests cover:
- Basic inference with different providers
- Caching behavior
- Force refresh
- Presets vs custom configs
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from multiview.inference import InferenceConfig, get_preset, list_presets, run_inference


class TestPresets:
    """Test preset loading and listing."""

    def test_list_presets(self):
        """Test that list_presets returns expected presets."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0
        assert "gemini_flash" in presets
        assert "claude_haiku" in presets
        assert "openai_embedding_large" in presets

    def test_get_preset(self):
        """Test getting a preset by name."""
        config = get_preset("gemini_flash")
        assert isinstance(config, InferenceConfig)
        assert config.provider == "gemini"
        assert config.model_name == "gemini-2.5-flash-lite"

    def test_get_invalid_preset(self):
        """Test that getting invalid preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent_preset")


class TestGeminiInference:
    """Test Gemini LM inference."""

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_basic_inference(self):
        """Test basic Gemini inference."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",  # Lite model with higher free tier limits
            prompt_template="What is 2+2? Answer with just the number.",
            parser="text",
            temperature=0.0,
            max_tokens=50,
        )

        results = run_inference(
            inputs={"dummy": ["dummy"]},  # Template doesn't use input
            config=config,
        )

        assert len(results) == 1
        assert isinstance(results[0], str)
        assert "4" in results[0]

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_json_parsing(self):
        """Test Gemini inference with JSON parsing."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",  # Lite model with higher free tier limits
            prompt_template='Return JSON: {{"number": 42}}',
            parser="json",
            parser_kwargs={"annotation_key": "number"},
            temperature=0.0,
            max_tokens=50,
        )

        results = run_inference(
            inputs={"dummy": ["dummy"]},
            config=config,
        )

        assert len(results) == 1
        assert results[0] == 42

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_force_prefill(self):
        """Test Gemini inference with force_prefill to verify it actually forces a specific start."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",  # Lite model with higher free tier limits
            prompt_template="Hi, what is your name",
            force_prefill_template="The sky is",
            parser="text",
            temperature=0.0,
            max_tokens=100,
        )

        results = run_inference(
            inputs={"dummy": ["dummy"]},
            config=config,
        )

        assert len(results) == 1
        print(f"Force prefill output: {results[0]}")
        # The response should start with the prefill text
        assert results[0].startswith("The sky is")


class TestCaching:
    """Test caching behavior."""

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_caching_on_second_request(self):
        """Test that second identical request uses cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"

            config = InferenceConfig(
                provider="gemini",
                model_name="gemini-2.5-flash-lite",  # Lite model with higher free tier limits
                prompt_template="Say 'hello': {text}",
                parser="text",
                temperature=0.0,
                max_tokens=20,
            )

            inputs = {"text": ["world"]}

            # First request - should hit API
            results1 = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )

            # Check cache file was created
            assert cache_path.exists()
            with open(cache_path) as f:
                cache_data = json.load(f)
            assert "completions" in cache_data
            assert len(cache_data["completions"]) == 1
            initial_cache_size = len(cache_data["completions"])

            # Second request - should use cache (no new API call)
            results2 = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )

            # Results should be identical
            assert results1 == results2

            # Cache should not have grown (no new entries)
            with open(cache_path) as f:
                cache_data = json.load(f)
            assert len(cache_data["completions"]) == initial_cache_size

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_force_refresh_bypasses_cache(self):
        """Test that force_refresh=True bypasses cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"

            config = InferenceConfig(
                provider="openai",
                model_name="gpt-4.1-mini",
                prompt_template="Count to {n}",
                parser="text",
                temperature=0.0,
                max_tokens=50,
            )

            inputs = {"n": ["3"]}

            # First request - populate cache
            results1 = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )

            # Manually modify cache to test force_refresh
            with open(cache_path) as f:
                cache_data = json.load(f)
            # Modify the cached value (new format has nested "result")
            for key in cache_data["completions"]:
                cache_data["completions"][key]["result"]["text"] = "CACHED_VALUE"
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)

            # Request without force_refresh - should use modified cache
            results2 = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )
            assert results2[0] == "CACHED_VALUE"

            # Request with force_refresh - should bypass cache and get fresh result
            results3 = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
                force_refresh=True,
            )
            # Should NOT be the cached value
            assert results3[0] != "CACHED_VALUE"
            # Should contain actual response (numbers)
            assert any(char.isdigit() for char in results3[0])

    def test_deduplication(self):
        """Test that duplicate inputs are deduplicated before API calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_cache.json"

            config = InferenceConfig(
                provider="gemini",
                model_name="gemini-2.5-flash-lite",  # Lite model with higher free tier limits
                prompt_template="Echo: {text}",
                parser="text",
                temperature=0.0,
                max_tokens=20,
            )

            # Duplicate inputs
            inputs = {"text": ["hello", "hello", "world", "hello"]}

            results = run_inference(
                inputs=inputs,
                config=config,
                cache_path=str(cache_path),
            )

            # Should have 4 results (matching input length)
            assert len(results) == 4

            # Check cache - should only have 2 unique entries (hello, world)
            with open(cache_path) as f:
                cache_data = json.load(f)
            # Should have deduped to 2 unique prompts
            assert len(cache_data["completions"]) == 2


class TestMultipleProviders:
    """Test multiple providers work."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_anthropic_inference(self):
        """Test Anthropic provider works."""
        config = InferenceConfig(
            provider="anthropic",
            model_name="claude-3-5-haiku-20241022",
            prompt_template="What is 1+1? Answer with just the number.",
            parser="text",
            temperature=0.0,
            max_tokens=10,
        )

        results = run_inference(
            inputs={"dummy": ["dummy"]},
            config=config,
        )

        assert len(results) == 1
        assert "2" in results[0]

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set",
    )
    def test_openai_inference(self):
        """Test OpenAI provider works."""
        config = InferenceConfig(
            provider="openai",
            model_name="gpt-4.1-mini",  # Updated to GPT-4.1-mini
            prompt_template="What is 3+3? Answer with just the number.",
            parser="text",
            temperature=0.0,
            max_tokens=10,
        )

        results = run_inference(
            inputs={"dummy": ["dummy"]},
            config=config,
        )
        print(f"{results=}")
        assert len(results) == 1
        assert "6" in results[0]


class TestPresetUsage:
    """Test using presets."""

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_preset_by_name(self):
        """Test using preset by string name."""
        # Override the template since default is for generic document
        results = run_inference(
            inputs={"document": ["What is 5+5? Just the number."]},
            config="gemini_flash",
            parser="text",  # Override parser
        )

        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_preset_with_overrides(self):
        """Test using preset with overrides."""
        # Get preset and modify it
        config = get_preset("gemini_flash")
        modified_config = config.with_overrides(
            prompt_template="Custom: {text}",
            max_tokens=50,
        )

        assert modified_config.prompt_template == "Custom: {text}"
        assert modified_config.max_tokens == 50
        # Other fields should remain
        assert modified_config.provider == "gemini"
        assert modified_config.model_name == "gemini-2.5-flash-lite"
