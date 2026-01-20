"""Tests for in-one-word evaluation method.

The in-one-word method uses hidden state extraction from HuggingFace local models.

Run specific test types:
  pytest -m needs_gpu tests/inference/test_in_one_word.py  # GPU tests only
"""

import numpy as np
import pytest

from multiview.eval import evaluate_with_in_one_word


# ============================================================================
# HF Local Tests (requires GPU)
# ============================================================================


@pytest.mark.needs_gpu
class TestInOneWordHFLocal:
    """Test in-one-word with HF local hidden state extraction (GPU required)."""

    def test_hf_local_basic(self):
        """Test basic HF local in-one-word evaluation."""
        # Simple math word problems
        documents = [
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast. How many eggs does she have left?",
            "Tom has 5 apples. He buys 3 more apples. How many apples does Tom have?",
            "The weather is sunny today.",
            "A car drives 60 miles per hour for 2 hours. How far does it travel?",
        ]

        # Create category schema (arithmetic operations)
        annotations = [
            {
                "category_schema": {
                    "categories": [
                        {"name": "addition", "description": "Adding numbers"},
                        {"name": "subtraction", "description": "Subtracting numbers"},
                        {"name": "multiplication", "description": "Multiplying numbers"},
                        {"name": "non_math", "description": "Not a math problem"},
                    ]
                }
            }
        ] * len(documents)

        # Triplets: math problems should be closer to each other than to non-math
        triplet_ids = [
            (0, 1, 2),  # subtraction vs (addition, weather)
            (0, 3, 2),  # subtraction vs (multiplication, weather)
        ]

        results = evaluate_with_in_one_word(
            documents=documents,
            triplet_ids=triplet_ids,
            annotations=annotations,
            preset="inoneword_hf_qwen3_4b",  # Use smaller model for testing
        )

        # Check structure
        assert "positive_scores" in results
        assert "negative_scores" in results
        assert "outcomes" in results
        assert "triplet_logs" in results

        # Check counts
        assert len(results["positive_scores"]) == len(triplet_ids)
        assert len(results["negative_scores"]) == len(triplet_ids)
        assert len(results["outcomes"]) == len(triplet_ids)
        assert len(results["triplet_logs"]) == len(triplet_ids)

        # Check that scores are valid (between -1 and 1 for cosine similarity)
        for score in results["positive_scores"]:
            assert -1.0 <= score <= 1.0

        for score in results["negative_scores"]:
            assert -1.0 <= score <= 1.0

        # Check outcomes are valid
        for outcome in results["outcomes"]:
            assert outcome in {-1, 0, 1}

    def test_hf_local_with_preset_overrides(self):
        """Test HF local with custom preset overrides."""
        documents = [
            "2 + 2 = 4",
            "5 - 3 = 2",
            "The sky is blue",
        ]

        annotations = [
            {
                "category_schema": {
                    "categories": [
                        {"name": "addition", "description": "Adding"},
                        {"name": "subtraction", "description": "Subtracting"},
                        {"name": "other", "description": "Other"},
                    ]
                }
            }
        ] * len(documents)

        triplet_ids = [(0, 1, 2)]

        results = evaluate_with_in_one_word(
            documents=documents,
            triplet_ids=triplet_ids,
            annotations=annotations,
            preset="inoneword_hf_qwen3_4b",
            preset_overrides={
                "hidden_layer_idx": -2,  # Use second-to-last layer
                "batch_size": 2,
                "max_length": 512,
            },
        )

        assert len(results["positive_scores"]) == 1
        assert len(results["negative_scores"]) == 1


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.needs_gpu
class TestInOneWordIntegration:
    """Integration tests for in-one-word evaluation."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline with category conditioning."""
        # Diverse set of documents
        documents = [
            "Calculate the sum of 15 and 27.",
            "Find the difference between 50 and 23.",
            "What is 8 times 6?",
            "Divide 36 by 9.",
            "The capital of France is Paris.",
            "Mount Everest is the tallest mountain.",
        ]

        # Rich category schema
        annotations = [
            {
                "category_schema": {
                    "categories": [
                        {"name": "addition", "description": "Addition operations"},
                        {"name": "subtraction", "description": "Subtraction operations"},
                        {"name": "multiplication", "description": "Multiplication operations"},
                        {"name": "division", "description": "Division operations"},
                        {"name": "geography", "description": "Geographic facts"},
                    ]
                }
            }
        ] * len(documents)

        # Multiple triplets testing different relationships
        triplet_ids = [
            (0, 1, 4),  # addition vs (subtraction, geography)
            (0, 2, 5),  # addition vs (multiplication, geography)
            (2, 3, 4),  # multiplication vs (division, geography)
        ]

        results = evaluate_with_in_one_word(
            documents=documents,
            triplet_ids=triplet_ids,
            annotations=annotations,
            preset="inoneword_hf_qwen3_4b",
            cache_alias="test_integration",
        )

        # Validate complete results
        assert len(results["triplet_logs"]) == len(triplet_ids)

        for log in results["triplet_logs"]:
            assert "triplet_idx" in log
            assert "anchor_id" in log
            assert "positive_id" in log
            assert "negative_id" in log
            assert "positive_score" in log
            assert "negative_score" in log
            assert "outcome" in log

        # Check average scores
        assert "avg_positive_score" in results
        assert "avg_negative_score" in results
        assert isinstance(results["avg_positive_score"], float)
        assert isinstance(results["avg_negative_score"], float)
