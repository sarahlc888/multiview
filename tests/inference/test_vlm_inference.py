"""Tests for VLM inference with Gemini.

Tests cover:
- Basic VLM inference with images
- Mixed image and text-only prompts
- Batch VLM processing
- Error handling
"""

import os

import pytest

from multiview.inference import InferenceConfig, run_inference


@pytest.mark.external
class TestGeminiVLMInference:
    """Test Gemini Vision-Language Model inference."""

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_vlm_basic_image_from_url(self):
        """Test basic Gemini VLM inference with image from URL."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.0-flash-exp",
            prompt_template="{document}",
            parser="text",
            temperature=0.0,
            max_tokens=100,
        )

        # Use a simple public domain image
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"

        results = run_inference(
            inputs={
                "documents": ["What animal is in this image? Answer in one word."],
                "images": [image_url],
            },
            config=config,
        )

        assert len(results) == 1
        assert isinstance(results[0], str)
        # Should recognize it's a cat
        assert "cat" in results[0].lower()

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_vlm_batch_processing(self):
        """Test batch processing with multiple images."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.0-flash-exp",
            prompt_template="{document}",
            parser="text",
            temperature=0.0,
            max_tokens=50,
        )

        # Two different cat images
        images = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/400px-Cat_November_2010-1a.jpg",
        ]

        prompts = [
            "What animal is this?",
            "What animal is this?",
        ]

        results = run_inference(
            inputs={
                "documents": prompts,
                "images": images,
            },
            config=config,
        )

        assert len(results) == 2
        # Both should recognize cats
        for result in results:
            assert isinstance(result, str)
            assert "cat" in result.lower()

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_vlm_mixed_image_and_text(self):
        """Test mixed batch with some images and some text-only prompts."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.0-flash-exp",
            prompt_template="{document}",
            parser="text",
            temperature=0.0,
            max_tokens=50,
        )

        # Mix of text-only and image prompts
        results = run_inference(
            inputs={
                "documents": [
                    "What is 2 + 2? Answer with just the number.",
                    "What animal is in this image?",
                ],
                "images": [
                    None,  # No image for first prompt
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg",
                ],
            },
            config=config,
        )

        assert len(results) == 2

        # First should answer math question
        assert "4" in results[0]

        # Second should recognize cat
        assert "cat" in results[1].lower()

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_vlm_with_invalid_image_url(self):
        """Test that invalid image URL falls back gracefully to text-only."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.0-flash-exp",
            prompt_template="{document}",
            parser="text",
            temperature=0.0,
            max_tokens=50,
        )

        # Invalid URL should fall back to text-only processing
        results = run_inference(
            inputs={
                "documents": ["What is 2 + 2?"],
                "images": ["https://invalid-url-that-does-not-exist.com/image.jpg"],
            },
            config=config,
        )

        # Should still get a response (falls back to text-only)
        assert len(results) == 1
        assert isinstance(results[0], str)

    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"),
        reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
    )
    def test_gemini_vlm_image_description(self):
        """Test detailed image description task."""
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.0-flash-exp",
            prompt_template="{document}",
            parser="text",
            temperature=0.0,
            max_tokens=200,
        )

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg"

        results = run_inference(
            inputs={
                "documents": ["Describe what you see in this image in 2-3 sentences."],
                "images": [image_url],
            },
            config=config,
        )

        assert len(results) == 1
        description = results[0].lower()

        # Should mention key elements
        assert "cat" in description or "feline" in description
        # Should have reasonable length
        assert len(description) > 20
