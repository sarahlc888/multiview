#!/usr/bin/env python3
"""Example: Using Vision-Language Models (VLMs) for image analysis.

This script demonstrates how to use Gemini Vision with the multiview inference pipeline
to analyze images. It shows:
1. Loading images from URLs or local paths
2. Sending images to Gemini Vision API
3. Getting text completions about images

Requirements:
- Set GEMINI_API_KEY environment variable
- Install dependencies: pip install google-genai pillow requests

Usage:
    python examples/vlm_usage.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiview.inference import run_inference  # noqa: E402
from multiview.inference.presets import InferenceConfig  # noqa: E402


def example_image_url_analysis():
    """Example 1: Analyze an image from a URL."""
    print("\n" + "=" * 60)
    print("Example 1: Analyzing image from URL")
    print("=" * 60)

    # Check API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY=your_key_here")
        return

    # Example image URL (replace with your own)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

    # Define inference config for VLM
    config = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
        prompt_template="{document}",
        temperature=0.0,
        max_tokens=512,
        parser="text",
    )

    # Run inference with image
    results = run_inference(
        inputs={
            "documents": [
                "Describe what you see in this image. Be specific about the subject, setting, and any notable details."
            ],
            "images": [image_url],
        },
        config=config,
    )

    print(f"\nPrompt: {results['inputs']['documents'][0]}")
    print(f"Image: {image_url}")
    print(f"\nResponse:\n{results['parsed_outputs'][0]}")


def example_local_image_analysis():
    """Example 2: Analyze a local image file."""
    print("\n" + "=" * 60)
    print("Example 2: Analyzing local image file")
    print("=" * 60)

    # Check API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return

    # Path to local image (replace with your own)
    # For this example, we'll skip if file doesn't exist
    image_path = "data/example_images/sample1.jpg"

    if not Path(image_path).exists():
        print(f"SKIPPED: Image file not found at {image_path}")
        print("To run this example:")
        print("  1. Create directory: mkdir -p data/example_images")
        print(f"  2. Add an image: cp your_image.jpg {image_path}")
        return

    # Define inference config for VLM
    config = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
        prompt_template="{document}",
        temperature=0.0,
        max_tokens=512,
        parser="text",
    )

    # Run inference with local image
    results = run_inference(
        inputs={
            "documents": [
                "What is the main subject of this image? Describe it in one sentence."
            ],
            "images": [image_path],
        },
        config=config,
    )

    print(f"\nPrompt: {results['inputs']['documents'][0]}")
    print(f"Image: {image_path}")
    print(f"\nResponse:\n{results['parsed_outputs'][0]}")


def example_batch_image_analysis():
    """Example 3: Batch analysis of multiple images."""
    print("\n" + "=" * 60)
    print("Example 3: Batch analysis of multiple images")
    print("=" * 60)

    # Check API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return

    # Multiple images (mix of URLs and local paths)
    images = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/400px-Cat_November_2010-1a.jpg",
    ]

    prompts = [
        "Is this a cat or a dog?",
        "Is this a cat or a dog?",
    ]

    # Define inference config
    config = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
        prompt_template="{document}",
        temperature=0.0,
        max_tokens=100,
        parser="text",
    )

    # Run batch inference
    results = run_inference(
        inputs={
            "documents": prompts,
            "images": images,
        },
        config=config,
    )

    print(f"\nAnalyzed {len(images)} images:")
    for i, (prompt, image, response) in enumerate(
        zip(prompts, images, results["parsed_outputs"], strict=False)
    ):
        print(f"\n--- Image {i+1} ---")
        print(f"Prompt: {prompt}")
        print(f"Image: {image}")
        print(f"Response: {response}")


def example_mixed_image_text():
    """Example 4: Mixed batch with some images and some text-only."""
    print("\n" + "=" * 60)
    print("Example 4: Mixed image and text-only prompts")
    print("=" * 60)

    # Check API key
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        return

    # Define inference config
    config = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.0-flash-exp",
        prompt_template="{document}",
        temperature=0.0,
        max_tokens=100,
        parser="text",
    )

    # Mix of image and text-only prompts (use None for text-only)
    results = run_inference(
        inputs={
            "documents": [
                "What is 2 + 2?",  # Text-only
                "What animal is in this image?",  # Image + text
            ],
            "images": [
                None,  # No image for first prompt
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/400px-Cat03.jpg",
            ],
        },
        config=config,
    )

    print("\nResults:")
    print(f"Text-only: {results['parsed_outputs'][0]}")
    print(f"Image+text: {results['parsed_outputs'][1]}")


if __name__ == "__main__":
    # Run all examples
    try:
        example_image_url_analysis()
        example_local_image_analysis()
        example_batch_image_analysis()
        example_mixed_image_text()

        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
