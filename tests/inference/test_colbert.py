"""Tests for ColBERT multi-vector embedding support.

Tests cover:
- Similarity functions (cosine, MaxSim, dispatcher)
- ColBERT provider functionality
- Integration with embeddings evaluation
- Multi-vector embedding handling
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from multiview.eval.similarity import (
    compute_similarity,
    cosine_similarity,
    maxsim_similarity,
)


class TestSimilarityFunctions:
    """Test similarity computation functions."""

    def test_cosine_similarity_basic(self):
        """Test basic cosine similarity computation."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]

        # Identical vectors should have similarity 1.0
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]

        # Orthogonal vectors should have similarity 0.0
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]

        # Opposite vectors should have similarity -1.0
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vectors."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]

        # Zero vector should return 0.0
        sim = cosine_similarity(vec_a, vec_b)
        assert sim == 0.0

    def test_cosine_similarity_numpy_arrays(self):
        """Test cosine similarity with numpy arrays."""
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([4.0, 5.0, 6.0])

        sim = cosine_similarity(vec_a, vec_b)
        assert isinstance(sim, float)
        assert 0.0 <= sim <= 1.0

    def test_maxsim_similarity_identical_single_token(self):
        """Test MaxSim with identical single-token embeddings."""
        # Each embedding has 1 token with 3 dimensions
        multi_vec_a = np.array([[1.0, 0.0, 0.0]])
        multi_vec_b = np.array([[1.0, 0.0, 0.0]])

        sim = maxsim_similarity(multi_vec_a, multi_vec_b)
        # Identical normalized vectors should have MaxSim ~1.0
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_maxsim_similarity_multiple_tokens(self):
        """Test MaxSim with multiple tokens."""
        # Query: 2 tokens, Doc: 3 tokens
        multi_vec_a = np.array([
            [1.0, 0.0, 0.0],  # Token 1
            [0.0, 1.0, 0.0],  # Token 2
        ])
        multi_vec_b = np.array([
            [1.0, 0.0, 0.0],  # Token 1 - matches query token 1
            [0.5, 0.5, 0.0],  # Token 2 - partial match
            [0.0, 0.0, 1.0],  # Token 3 - no match
        ])

        sim = maxsim_similarity(multi_vec_a, multi_vec_b)
        # Each query token finds its best match in doc
        # Token 1 matches perfectly (sim=1.0)
        # Token 2 matches with token 2 (sim~0.7)
        # Average should be between 0.7 and 1.0
        assert 0.7 <= sim <= 1.0

    def test_maxsim_similarity_with_padding(self):
        """Test MaxSim filters out zero-padded tokens."""
        # Simulated padded embeddings (ColBERT pads with zeros)
        multi_vec_a = np.array([
            [1.0, 0.0, 0.0],  # Real token
            [0.0, 0.0, 0.0],  # Padding (should be filtered)
            [0.0, 0.0, 0.0],  # Padding (should be filtered)
        ])
        multi_vec_b = np.array([
            [1.0, 0.0, 0.0],  # Real token
            [0.0, 0.0, 0.0],  # Padding (should be filtered)
        ])

        sim = maxsim_similarity(multi_vec_a, multi_vec_b)
        # Only real tokens should contribute, should be ~1.0
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_maxsim_similarity_empty_after_filtering(self):
        """Test MaxSim with all-zero embeddings."""
        multi_vec_a = np.zeros((3, 128))
        multi_vec_b = np.zeros((2, 128))

        sim = maxsim_similarity(multi_vec_a, multi_vec_b)
        # Should return 0.0 for empty embeddings
        assert sim == 0.0

    def test_compute_similarity_single_vector(self):
        """Test compute_similarity dispatcher with 1D vectors."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]

        # Should use cosine_similarity for 1D
        sim = compute_similarity(vec_a, vec_b)
        assert sim == pytest.approx(1.0)

    def test_compute_similarity_multi_vector(self):
        """Test compute_similarity dispatcher with 2D vectors."""
        multi_vec_a = np.array([[1.0, 0.0, 0.0]])
        multi_vec_b = np.array([[1.0, 0.0, 0.0]])

        # Should use maxsim_similarity for 2D
        sim = compute_similarity(multi_vec_a, multi_vec_b)
        assert sim == pytest.approx(1.0, abs=0.01)

    def test_compute_similarity_dimension_mismatch_error(self):
        """Test compute_similarity raises error for mismatched dimensions."""
        vec_1d = [1.0, 0.0, 0.0]
        vec_2d = np.array([[1.0, 0.0, 0.0]])

        with pytest.raises(ValueError, match="dimension mismatch"):
            compute_similarity(vec_1d, vec_2d)

    def test_compute_similarity_unsupported_dimension_error(self):
        """Test compute_similarity raises error for 3D+ arrays."""
        vec_3d = np.zeros((2, 3, 4))

        with pytest.raises(ValueError, match="Unsupported embedding shape"):
            compute_similarity(vec_3d, vec_3d)


class TestColBERTProviderUnit:
    """Unit tests for ColBERT provider without requiring model download."""

    @pytest.mark.skipif(
        True,  # Skip mocked tests - they require pylate/torch to be importable for patching
        reason="Mocked tests require pylate/torch packages to be installed for patching",
    )
    def test_colbert_provider_basic_mock(self):
        """Test basic ColBERT provider functionality with mocked model."""
        # Import inside to mock properly
        with patch("pylate.models.ColBERT") as mock_colbert_class, \
             patch("torch.compile") as mock_compile, \
             patch("torch.inference_mode") as mock_inference_mode:

            from multiview.inference.providers.hf_local_colbert import (
                hf_local_colbert_completions,
            )

            # Mock the model
            mock_model = MagicMock()
            mock_colbert_class.return_value = mock_model
            mock_compile.return_value = mock_model

            # Mock context manager for inference_mode
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=None)
            mock_inference_mode.return_value = mock_cm

            # Mock encode to return multi-vector embeddings
            # Shape: (batch_size=2, num_tokens=10, embedding_dim=128)
            mock_embeddings = np.random.randn(2, 10, 128).astype(np.float32)
            mock_model.encode.return_value = mock_embeddings

            # Test with 2 prompts
            prompts = ["First document", "Second document"]
            results = hf_local_colbert_completions(
                prompts=prompts,
                model_name="lightonai/Reason-ModernColBERT",
                device="cuda",
                batch_size=2,
            )

            # Check results structure
            assert "completions" in results
            assert len(results["completions"]) == 2

            # Check each completion has a vector
            for completion in results["completions"]:
                assert "vector" in completion
                vector = completion["vector"]
                # Should be 2D (multi-vector)
                assert isinstance(vector, np.ndarray)
                assert vector.ndim == 2

    @pytest.mark.skipif(
        True,  # Skip mocked tests - they require pylate/torch to be importable for patching
        reason="Mocked tests require pylate/torch packages to be installed for patching",
    )
    def test_colbert_jina_special_handling_mock(self):
        """Test that jina-colbert-v2 gets special prefix handling."""
        with patch("pylate.models.ColBERT") as mock_colbert_class, \
             patch("torch.compile") as mock_compile, \
             patch("torch.inference_mode") as mock_inference_mode:

            from multiview.inference.providers.hf_local_colbert import (
                hf_local_colbert_completions,
            )

            mock_model = MagicMock()
            mock_colbert_class.return_value = mock_model
            mock_compile.return_value = mock_model

            # Mock context manager
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=None)
            mock_inference_mode.return_value = mock_cm

            mock_embeddings = np.random.randn(1, 10, 128).astype(np.float32)
            mock_model.encode.return_value = mock_embeddings

            # Call with jina model
            hf_local_colbert_completions(
                prompts=["test"],
                model_name="jinaai/jina-colbert-v2",
                device="cuda",
            )

            # Check that ColBERT was called with jina-specific kwargs
            call_kwargs = mock_colbert_class.call_args[1]
            assert "query_prefix" in call_kwargs
            assert call_kwargs["query_prefix"] == "[QueryMarker]"
            assert "document_prefix" in call_kwargs
            assert call_kwargs["document_prefix"] == "[DocumentMarker]"
            assert call_kwargs["trust_remote_code"] is True

    @pytest.mark.skipif(
        True,  # Skip mocked tests - they require pylate/torch to be importable for patching
        reason="Mocked tests require pylate/torch packages to be installed for patching",
    )
    def test_colbert_with_instructions_mock(self):
        """Test ColBERT provider with query instructions."""
        with patch("pylate.models.ColBERT") as mock_colbert_class, \
             patch("torch.compile") as mock_compile, \
             patch("torch.inference_mode") as mock_inference_mode:

            from multiview.inference.providers.hf_local_colbert import (
                hf_local_colbert_completions,
            )

            mock_model = MagicMock()
            mock_colbert_class.return_value = mock_model
            mock_compile.return_value = mock_model

            # Mock context manager
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=None)
            mock_inference_mode.return_value = mock_cm

            mock_embeddings = np.random.randn(2, 10, 128).astype(np.float32)
            mock_model.encode.return_value = mock_embeddings

            # Test with instructions
            prompts = ["doc1", "doc2"]
            instructions = ["instr1", "instr2"]

            hf_local_colbert_completions(
                prompts=prompts,
                model_name="lightonai/Reason-ModernColBERT",
                instructions=instructions,
                device="cuda",
                batch_size=2,
            )

            # Verify encode was called
            assert mock_model.encode.called

    def test_colbert_provider_registration(self):
        """Test that ColBERT provider is registered."""
        from multiview.inference.providers import get_completion_fn

        # Should be able to get the provider
        completion_fn = get_completion_fn("hf_local_colbert")
        assert callable(completion_fn)

    def test_colbert_presets_exist(self):
        """Test that ColBERT presets are registered."""
        from multiview.inference.presets import get_preset

        # Test all ColBERT presets
        presets = [
            "colbert_reason_modern",
            "colbert_jina_v2",
            "colbert_gte_modern_v1",
            "instr_colbert_reason_modern",
        ]

        for preset_name in presets:
            preset = get_preset(preset_name)
            assert preset.provider == "hf_local_colbert"
            assert preset.parser == "vector"

    def test_colbert_in_gpu_required_providers(self):
        """Test that ColBERT is marked as GPU-required."""
        from multiview.inference.presets import GPU_REQUIRED_PROVIDERS

        assert "hf_local_colbert" in GPU_REQUIRED_PROVIDERS


class TestColBERTEmbeddingsIntegration:
    """Test ColBERT integration with embeddings evaluation."""

    @patch("multiview.eval.embeddings.run_inference")
    def test_embeddings_with_multi_vector(self, mock_run_inference):
        """Test that embeddings evaluation works with multi-vector embeddings."""
        from multiview.eval.embeddings import evaluate_with_embeddings

        # Mock run_inference to return multi-vector embeddings
        # 3 documents, each with 5 tokens of 128 dims
        mock_embeddings = [
            np.random.randn(5, 128).astype(np.float32),
            np.random.randn(5, 128).astype(np.float32),
            np.random.randn(5, 128).astype(np.float32),
        ]
        mock_run_inference.return_value = mock_embeddings

        documents = ["doc1", "doc2", "doc3"]
        triplet_ids = [(0, 1, 2)]

        results = evaluate_with_embeddings(
            documents=documents,
            triplet_ids=triplet_ids,
            embedding_preset="colbert_reason_modern",
        )

        # Check results structure
        assert "positive_scores" in results
        assert "negative_scores" in results
        assert len(results["positive_scores"]) == 1
        assert len(results["negative_scores"]) == 1

        # Scores should be floats in valid range
        assert isinstance(results["positive_scores"][0], float)
        assert isinstance(results["negative_scores"][0], float)

    @patch("multiview.eval.embeddings.run_inference")
    def test_embeddings_mixed_single_and_multi_vector(self, mock_run_inference):
        """Test that compute_similarity handles both types in same call."""
        from multiview.eval.embeddings import evaluate_with_embeddings

        # Return single-vector embeddings (would work with compute_similarity too)
        mock_embeddings = [
            [0.1] * 128,  # Single vector
            [0.2] * 128,  # Single vector
            [0.3] * 128,  # Single vector
        ]
        mock_run_inference.return_value = mock_embeddings

        documents = ["doc1", "doc2", "doc3"]
        triplet_ids = [(0, 1, 2)]

        # Should work with single-vector embeddings via compute_similarity
        results = evaluate_with_embeddings(
            documents=documents,
            triplet_ids=triplet_ids,
            embedding_preset="openai_embedding_small",
        )

        assert len(results["positive_scores"]) == 1
        assert len(results["negative_scores"]) == 1


@pytest.mark.needs_gpu
@pytest.mark.slow
@pytest.mark.skipif(
    True,  # Skip by default - requires downloading large models
    reason="Integration test requires downloading ColBERT models (~2GB) and GPU",
)
class TestColBERTIntegration:
    """Integration tests that require downloading actual ColBERT models.

    These tests are disabled by default. To run them:
    1. Install dependencies: pip install pylate transformers torch
    2. Ensure GPU is available
    3. Ensure sufficient disk space (~2GB per model)
    4. Run with: pytest tests/inference/test_colbert.py::TestColBERTIntegration -v
    """

    def test_colbert_reason_modern_basic(self):
        """Test Reason-ModernColBERT model produces valid embeddings."""
        from multiview.inference import run_inference

        documents = ["Machine learning is a subset of AI.", "Python is a programming language."]

        results = run_inference(
            inputs={"document": documents},
            config="colbert_reason_modern",
        )

        # Should return 2 embeddings
        assert len(results) == 2

        # Each should be a multi-vector embedding (2D numpy array)
        for emb in results:
            assert isinstance(emb, np.ndarray)
            assert emb.ndim == 2
            # Should have token dimension and embedding dimension
            assert emb.shape[0] > 0  # At least 1 token
            assert emb.shape[1] > 0  # Embedding dimension

    def test_colbert_jina_v2_basic(self):
        """Test jina-colbert-v2 model produces valid embeddings."""
        from multiview.inference import run_inference

        documents = ["Test document one", "Test document two"]

        results = run_inference(
            inputs={"document": documents},
            config="colbert_jina_v2",
        )

        assert len(results) == 2
        for emb in results:
            assert isinstance(emb, np.ndarray)
            assert emb.ndim == 2

    def test_colbert_with_instruction(self):
        """Test ColBERT with instruction template."""
        from multiview.inference import run_inference

        documents = ["Machine learning", "Deep learning"]

        results = run_inference(
            inputs={"document": documents, "criterion": "technical topics"},
            config="instr_colbert_reason_modern",
        )

        assert len(results) == 2
        for emb in results:
            assert isinstance(emb, np.ndarray)
            assert emb.ndim == 2

    def test_colbert_maxsim_scores_reasonable(self):
        """Test that MaxSim scores are in reasonable range."""
        from multiview.eval.embeddings import evaluate_with_embeddings

        # Similar documents should have higher score
        documents = [
            "Machine learning is used in AI.",  # Anchor
            "AI systems use machine learning techniques.",  # Similar (positive)
            "Cooking involves preparing food.",  # Different (negative)
        ]

        triplet_ids = [(0, 1, 2)]

        results = evaluate_with_embeddings(
            documents=documents,
            triplet_ids=triplet_ids,
            embedding_preset="colbert_reason_modern",
        )

        pos_score = results["positive_scores"][0]
        neg_score = results["negative_scores"][0]

        # Scores should be in [0, 1] for normalized embeddings
        assert 0.0 <= pos_score <= 1.0
        assert 0.0 <= neg_score <= 1.0

        # Similar documents should score higher
        assert pos_score > neg_score

    def test_colbert_model_caching(self):
        """Test that ColBERT model is cached between calls."""
        from multiview.inference import run_inference

        # First call - should load model
        results1 = run_inference(
            inputs={"document": ["Test 1"]},
            config="colbert_reason_modern",
        )

        # Second call - should use cached model
        results2 = run_inference(
            inputs={"document": ["Test 2"]},
            config="colbert_reason_modern",
        )

        # Both should return valid embeddings
        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0].ndim == 2
        assert results2[0].ndim == 2


class TestColBERTBatchProcessing:
    """Test ColBERT batching logic."""

    @pytest.mark.skipif(
        True,  # Skip mocked tests - they require pylate/torch to be importable for patching
        reason="Mocked tests require pylate/torch packages to be installed for patching",
    )
    def test_batch_by_instruction_mock(self):
        """Test that prompts are batched by instruction type."""
        with patch("pylate.models.ColBERT") as mock_colbert_class, \
             patch("torch.compile") as mock_compile, \
             patch("torch.inference_mode") as mock_inference_mode:

            from multiview.inference.providers.hf_local_colbert import (
                hf_local_colbert_completions,
            )

            mock_model = MagicMock()
            mock_colbert_class.return_value = mock_model
            mock_compile.return_value = mock_model

            # Mock context manager
            mock_cm = MagicMock()
            mock_cm.__enter__ = MagicMock(return_value=None)
            mock_cm.__exit__ = MagicMock(return_value=None)
            mock_inference_mode.return_value = mock_cm

            # Return different shaped embeddings for different batches
            def mock_encode(**kwargs):
                batch_size = len(kwargs["sentences"])
                return np.random.randn(batch_size, 10, 128).astype(np.float32)

            mock_model.encode.side_effect = mock_encode

            # 4 prompts with 2 different instructions
            prompts = ["doc1", "doc2", "doc3", "doc4"]
            instructions = ["instr_a", "instr_a", "instr_b", "instr_b"]

            results = hf_local_colbert_completions(
                prompts=prompts,
                model_name="lightonai/Reason-ModernColBERT",
                instructions=instructions,
                batch_size=2,
            )

            # Should return 4 results
            assert len(results["completions"]) == 4

            # Each should have a vector
            for completion in results["completions"]:
                assert "vector" in completion
