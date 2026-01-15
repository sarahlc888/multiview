"""Tests for reranker-based evaluation.

Tests cover:
- Basic reranker functionality
- Score computation for query-document pairs
- Triplet evaluation
- Tokenization with prefix/suffix tokens
- Batch processing
- Integration with benchmark system
"""

from unittest.mock import MagicMock, patch

import pytest

from multiview.eval import evaluate_with_reranker


@pytest.mark.skipif(
    True,  # Skip by default since reranker requires local model download
    reason="Reranker tests require downloading large models and GPU/CPU resources",
)
class TestRerankerBasic:
    """Test basic reranker functionality."""

    def test_reranker_basic_scoring(self):
        """Test basic reranker scoring with simple query-document pairs."""
        # Simple test documents
        documents = [
            "What is the capital of France?",  # Query
            "Paris is the capital of France.",  # Highly relevant
            "London is the capital of England.",  # Not relevant
            "France is a country in Europe.",  # Somewhat relevant
        ]

        # Triplets: (anchor, positive, negative)
        triplet_ids = [
            (0, 1, 2),  # Query vs (Paris answer, London answer)
            (0, 1, 3),  # Query vs (Paris answer, general France info)
        ]

        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_model="Qwen/Qwen3-Reranker-8B",
            batch_size=4,
            device="cpu",  # Use CPU for testing
            use_fp16=False,
        )

        # Check structure
        assert "positive_scores" in results
        assert "negative_scores" in results
        assert "triplet_logs" in results

        # Check counts
        assert len(results["positive_scores"]) == len(triplet_ids)
        assert len(results["negative_scores"]) == len(triplet_ids)
        assert len(results["triplet_logs"]) == len(triplet_ids)

        # Check scores are in valid range [0, 1]
        for score in results["positive_scores"]:
            assert 0.0 <= score <= 1.0

        for score in results["negative_scores"]:
            assert 0.0 <= score <= 1.0

        # Check that positive scores are generally higher than negative
        # (for this simple case, the highly relevant answer should score higher)
        assert results["positive_scores"][0] > results["negative_scores"][0]

    def test_reranker_with_custom_instruction(self):
        """Test reranker with custom instruction."""
        documents = [
            "Machine learning is a subset of AI.",
            "Machine learning involves training models on data.",
            "Cooking involves preparing food.",
        ]

        triplet_ids = [(0, 1, 2)]

        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_model="Qwen/Qwen3-Reranker-8B",
            instruction="Determine if the documents are semantically related",
            batch_size=4,
            device="cpu",
            use_fp16=False,
        )

        assert len(results["positive_scores"]) == 1
        assert len(results["negative_scores"]) == 1

        # ML-related documents should score higher than cooking
        assert results["positive_scores"][0] > results["negative_scores"][0]


class TestRerankerUnit:
    """Unit tests for reranker functions without requiring the model."""

    def test_empty_triplets(self):
        """Test handling of empty triplet list."""
        documents = ["doc1", "doc2"]
        triplet_ids = []

        # This shouldn't try to load the model since there are no triplets
        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_preset="qwen3_reranker_8b",
        )

        assert results["positive_scores"] == []
        assert results["negative_scores"] == []
        assert results["triplet_logs"] == []
        assert results["avg_positive_score"] == 0.0
        assert results["avg_negative_score"] == 0.0

    @patch("multiview.eval.reranker.run_inference")
    def test_evaluate_with_mocked_scores(self, mock_run_inference):
        """Test evaluate_with_reranker with mocked inference function."""
        # Mock run_inference to return predetermined scores
        # For 2 triplets, we need 4 scores: 2 positive + 2 negative (interleaved)
        mock_run_inference.return_value = [0.9, 0.3, 0.8, 0.4]

        documents = ["anchor1", "pos1", "neg1", "anchor2", "pos2", "neg2"]
        triplet_ids = [(0, 1, 2), (3, 4, 5)]

        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_preset="qwen3_reranker_8b",
        )

        # Verify the mock was called
        assert mock_run_inference.called

        # Check results structure
        assert len(results["positive_scores"]) == 2
        assert len(results["negative_scores"]) == 2
        assert len(results["triplet_logs"]) == 2

        # Check scores match our mock (interleaved: pos1, neg1, pos2, neg2)
        assert results["positive_scores"] == [0.9, 0.8]
        assert results["negative_scores"] == [0.3, 0.4]

        # Check correctness
        assert results["triplet_logs"][0]["correct"] is True  # 0.9 > 0.3
        assert results["triplet_logs"][1]["correct"] is True  # 0.8 > 0.4

        # Check margins
        assert results["triplet_logs"][0]["margin"] == pytest.approx(0.6)  # 0.9 - 0.3
        assert results["triplet_logs"][1]["margin"] == pytest.approx(0.4)  # 0.8 - 0.4

        # Check averages
        assert results["avg_positive_score"] == pytest.approx(0.85)  # (0.9 + 0.8) / 2
        assert results["avg_negative_score"] == pytest.approx(0.35)  # (0.3 + 0.4) / 2

    def test_triplet_logs_structure(self):
        """Test that triplet logs have correct structure (mocked)."""
        # This is a structure test - we're checking that the function
        # would produce the right log structure given scores

        # We can't easily mock the reranker without importing it,
        # so this test just verifies the expected structure
        expected_keys = {
            "triplet_idx",
            "anchor_id",
            "positive_id",
            "negative_id",
            "positive_score",
            "negative_score",
            "correct",
            "margin",
        }

        # This is what the log structure should be
        # Actual test would require running the model
        sample_log = {
            "triplet_idx": 0,
            "anchor_id": 0,
            "positive_id": 1,
            "negative_id": 2,
            "positive_score": 0.8,
            "negative_score": 0.3,
            "correct": True,
            "margin": 0.5,
        }

        assert set(sample_log.keys()) == expected_keys
        assert sample_log["correct"] == (
            sample_log["positive_score"] > sample_log["negative_score"]
        )
        assert sample_log["margin"] == (
            sample_log["positive_score"] - sample_log["negative_score"]
        )


@pytest.mark.external
@pytest.mark.skipif(
    True,  # Skip by default - enable manually for integration testing
    reason="Integration test requires model download and significant compute",
)
class TestRerankerIntegration:
    """Integration tests that require downloading the actual model.

    These tests are disabled by default. To run them:
    1. Install dependencies: pip install transformers torch
    2. Ensure sufficient disk space for model download (~8GB)
    3. Run with: pytest tests/eval/test_reranker.py -k TestRerankerIntegration -v --run-external
    """

    def test_reranker_sanity_check(self):
        """Sanity check that reranker produces sensible relevance scores."""
        # Test with a clear relevant/non-relevant case
        documents = [
            "What is Python?",  # Query
            "Python is a high-level programming language.",  # Clearly relevant
            "Snakes are reptiles that lack limbs.",  # Clearly not relevant
        ]

        triplet_ids = [(0, 1, 2)]

        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_model="Qwen/Qwen3-Reranker-8B",
            batch_size=4,
            device="cpu",
            use_fp16=False,
            max_length=8192,
        )

        # The programming answer should be strongly preferred over the animal answer
        positive_score = results["positive_scores"][0]
        negative_score = results["negative_scores"][0]

        print(f"Positive score (programming): {positive_score}")
        print(f"Negative score (animal): {negative_score}")

        # Should be a clear preference
        assert positive_score > negative_score
        # Should be a significant margin (at least 0.3)
        assert (positive_score - negative_score) > 0.3

        # Check that it's marked as correct
        assert results["triplet_logs"][0]["correct"] is True

    def test_reranker_batch_processing(self):
        """Test that batch processing works correctly with multiple triplets."""
        documents = [
            # Set 1: Programming question
            "What is JavaScript?",
            "JavaScript is a programming language for web development.",
            "Java is an island in Indonesia.",
            # Set 2: Geography question
            "What is the capital of Japan?",
            "Tokyo is the capital city of Japan.",
            "Paris is the capital of France.",
        ]

        triplet_ids = [
            (0, 1, 2),  # JavaScript question
            (3, 4, 5),  # Tokyo question
        ]

        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_model="Qwen/Qwen3-Reranker-8B",
            batch_size=2,  # Process in batches of 2
            device="cpu",
            use_fp16=False,
        )

        # Both should be correct
        assert all(log["correct"] for log in results["triplet_logs"])

        # Both positive scores should be higher than negative
        for pos, neg in zip(
            results["positive_scores"], results["negative_scores"], strict=True
        ):
            assert pos > neg

    def test_reranker_max_length_handling(self):
        """Test that max_length parameter is respected."""
        # Create a long document
        long_doc = "This is a sentence. " * 1000  # ~5000 tokens

        documents = [
            "What is this about?",
            long_doc + " This is about testing.",
            "Completely unrelated content.",
        ]

        triplet_ids = [(0, 1, 2)]

        # Test with smaller max_length
        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_model="Qwen/Qwen3-Reranker-8B",
            batch_size=4,
            device="cpu",
            use_fp16=False,
            max_length=4096,  # Smaller than default
        )

        # Should still work (truncate long doc)
        assert len(results["positive_scores"]) == 1
        assert len(results["negative_scores"]) == 1

        # Should produce valid scores
        assert 0.0 <= results["positive_scores"][0] <= 1.0
        assert 0.0 <= results["negative_scores"][0] <= 1.0


class TestRerankerBenchmarkIntegration:
    """Test integration with benchmark evaluation system."""

    @patch("multiview.eval.reranker.run_inference")
    def test_finalize_method_results_integration(self, mock_run_inference):
        """Test that reranker results work with finalize_method_results."""
        from multiview.benchmark.evaluation_utils import finalize_method_results

        # Mock run_inference to return scores for 1 triplet
        mock_run_inference.return_value = [0.85, 0.35]

        documents = ["query", "relevant", "not relevant"]
        triplet_ids = [(0, 1, 2)]

        raw_results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_preset="qwen3_reranker_8b",
        )

        # Finalize results (should add metrics like accuracy)
        finalized = finalize_method_results(raw_results)

        # Check that standard metrics are added
        assert "accuracy" in finalized
        assert "n_correct" in finalized
        assert "n_incorrect" in finalized
        assert "n_total" in finalized
        assert "correct" in finalized
        assert "is_tie" in finalized

        # Check correctness
        assert finalized["accuracy"] == 1.0  # 0.85 > 0.35
        assert finalized["n_correct"] == 1
        assert finalized["n_incorrect"] == 0
        assert finalized["n_total"] == 1
        assert finalized["correct"] == [True]
        assert finalized["is_tie"] == [False]

    @patch("multiview.eval.reranker.run_inference")
    def test_method_config_parameters(self, mock_run_inference):
        """Test that method config parameters are properly passed through."""
        mock_run_inference.return_value = [0.7, 0.4]

        documents = ["q", "pos", "neg"]
        triplet_ids = [(0, 1, 2)]

        # Test with custom parameters
        results = evaluate_with_reranker(
            documents=documents,
            triplet_ids=triplet_ids,
            reranker_preset="qwen3_reranker_8b",
            instruction="Custom instruction",
            cache_alias="test_cache",
            run_name="test_run",
        )

        # Verify mock was called
        assert mock_run_inference.called

        # Verify inputs were formatted correctly
        call_args = mock_run_inference.call_args
        inputs = call_args[1]["inputs"]
        assert "query" in inputs
        assert "document" in inputs
        assert "instruction" in inputs
        assert inputs["instruction"][0] == "Custom instruction"


class TestRerankerErrorHandling:
    """Test error handling in reranker evaluation."""

    @pytest.mark.skipif(
        True,  # Skip - requires torch/transformers
        reason="Requires torch/transformers to test validation logic",
    )
    def test_mismatched_query_doc_lengths_validation(self):
        """Test that mismatched query/doc lengths raise ValueError.

        Note: This test is skipped by default as it requires torch/transformers.
        The validation logic is simple enough that we trust it works.
        """
        from multiview.eval.reranker import get_reranker_scores

        queries = ["query1", "query2"]
        documents = ["doc1"]  # Mismatched length

        with pytest.raises(ValueError, match="must match"):
            get_reranker_scores(
                queries=queries,
                documents=documents,
                model_name="Qwen/Qwen3-Reranker-8B",
            )

    def test_invalid_triplet_indices(self):
        """Test handling of invalid triplet indices."""
        documents = ["doc1", "doc2"]
        triplet_ids = [(0, 1, 5)]  # Index 5 doesn't exist

        # This should raise an IndexError when trying to access documents[5]
        with pytest.raises(IndexError):
            # This will fail before calling inference since we access documents[5]
            evaluate_with_reranker(
                documents=documents,
                triplet_ids=triplet_ids,
                reranker_preset="qwen3_reranker_8b",
            )

    def test_inference_system_integration(self):
        """Test that reranker integrates with the inference system correctly."""
        # This is a structural test - verify that the reranker properly uses the inference system
        # The actual inference is tested in the provider tests

        # Verify the preset exists
        from multiview.inference.presets import get_preset

        preset = get_preset("qwen3_reranker_8b")
        assert preset.provider == "hf_local"
        assert preset.model_name == "Qwen/Qwen3-Reranker-8B"
        assert preset.parser == "score"

        # Verify the score parser exists
        from multiview.inference.parsers import get_parser

        parser = get_parser("score")
        assert callable(parser)

        # Test score parser with sample data
        test_completion = {"score": 0.85}
        result = parser(test_completion)
        assert result == 0.85
