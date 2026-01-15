"""Tests for annotator presets matching old repo configs.

These tests verify that the 5 key annotators work correctly and match
the behavior of the old repo's annotator configs.
"""

import os

import pytest

from multiview.inference import (
    get_preset,
    list_presets,
    run_inference,
)

pytestmark = pytest.mark.dev


class TestPresetListing:
    """Test listing and loading annotator presets."""

    def test_list_presets(self):
        """Test that all core annotators are listed."""
        presets = list_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

        # Verify the core annotators are present
        expected = [
            "embed_plaintext_hfapi",
            "rewrite_plaintext_freeform_gemini",
            "lmjudge_pair_plaintext_likerthard_gemini",
            "lmjudge_triplet_plaintext_binaryhard_gemini",
            "lmjudge_pair_norewrite_binaryhard_gemini",
        ]
        for name in expected:
            assert name in presets

    def test_get_preset(self):
        """Test getting annotator preset by name."""
        config = get_preset("embed_plaintext_hfapi")
        assert config.provider == "hf_embedding"
        assert (
            config.model_name == "Qwen/Qwen3-Embedding-8B"
        )  # 4B not available via Inference API

    def test_direct_import(self):
        """Test that annotator configs can be accessed via get_preset()."""
        embed_config = get_preset("embed_plaintext_hfapi")
        assert embed_config.provider == "hf_embedding"

        rewrite_config = get_preset("rewrite_plaintext_freeform_gemini")
        assert rewrite_config.provider == "gemini"

        lmjudge_config = get_preset("lmjudge_pair_plaintext_likerthard_gemini")
        assert lmjudge_config.provider == "gemini"


@pytest.mark.external
class TestEmbedPlaintextHFAPI:
    """Test embed_plaintext_hfapi annotator."""

    @pytest.mark.skipif(
        not (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_API_KEY")
            or os.getenv("HUGGINGFACE_API_KEY")
        ),
        reason="HF_TOKEN not set",
    )
    def test_embed_plaintext_hfapi(self):
        """Test HF API embeddings with Qwen3-8B."""
        results = run_inference(
            inputs={"document": ["Test document for embedding"]},
            config="embed_plaintext_hfapi",
        )

        assert len(results) == 1
        vector = results[0]
        assert isinstance(vector, list)
        # Qwen3-Embedding-8B should have thousands of dimensions (4096)
        assert len(vector) > 1000
        assert all(isinstance(x, (int, float)) for x in vector)


@pytest.mark.external
class TestRewritePlaintextFreeformGemini:
    """Test rewrite_plaintext_freeform_gemini annotator."""

    @pytest.mark.skipif(
        not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        reason="GEMINI_API_KEY not set",
    )
    def test_rewrite_freeform(self):
        """Test Gemini rewriter with criteria-based summarization."""
        results = run_inference(
            inputs={
                "document": [
                    "Machine learning is a subset of artificial intelligence. "
                    "It involves training algorithms on data to make predictions."
                ],
                "similarity_criteria": ["Main technical concepts"],
            },
            config="rewrite_plaintext_freeform_gemini",
        )

        assert len(results) == 1
        summary = results[0]
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain relevant terms
        assert any(
            term.lower() in summary.lower()
            for term in ["machine", "learning", "algorithm", "data", "AI"]
        )


@pytest.mark.external
class TestLMJudgePairLikert:
    """Test lmjudge_pair_plaintext_likerthard_gemini annotator."""

    @pytest.mark.skipif(
        not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        reason="GEMINI_API_KEY not set",
    )
    def test_likert_judge_similar_texts(self):
        """Test Likert judge on similar texts (should be 4-5)."""
        results = run_inference(
            inputs={
                "similarity_criteria": ["Topic"],
                "document_a": ["The cat sat on the mat."],
                "document_b": ["A feline was sitting on the rug."],
            },
            config="lmjudge_pair_plaintext_likerthard_gemini",
        )

        assert len(results) == 1
        judgment = results[0]
        assert isinstance(judgment, int)
        assert 1 <= judgment <= 7
        # Similar texts should score high (4-7)
        assert judgment >= 3

    @pytest.mark.skipif(
        not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        reason="GEMINI_API_KEY not set",
    )
    def test_likert_judge_different_texts(self):
        """Test Likert judge on different texts (should be 1-3)."""
        results = run_inference(
            inputs={
                "similarity_criteria": ["Topic"],
                "document_a": ["The weather is nice today."],
                "document_b": ["Quantum mechanics is complex."],
            },
            config="lmjudge_pair_plaintext_likerthard_gemini",
        )

        assert len(results) == 1
        judgment = results[0]
        assert isinstance(judgment, int)
        assert 1 <= judgment <= 7


@pytest.mark.external
class TestLMJudgeTripletBinary:
    """Test lmjudge_triplet_plaintext_binaryhard_gemini annotator."""

    @pytest.mark.skipif(
        not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        reason="GEMINI_API_KEY not set",
    )
    def test_triplet_judge(self):
        """Test triplet judge picks the more similar text."""
        results = run_inference(
            inputs={
                "similarity_criteria": ["Topic"],
                "document_a": ["Dogs are popular pets."],
                "document_b": ["Cats are also popular pets."],  # Similar topic
                "document_c": [
                    "The stock market crashed yesterday."
                ],  # Different topic
            },
            config="lmjudge_triplet_plaintext_binaryhard_gemini",
        )

        assert len(results) == 1
        judgment = results[0]
        # Should return 1 (b more similar), -1 (c more similar), or 0.0 (tie)
        assert judgment in [1, -1, 0.0]
        # (b) should be more similar to (a) than (c)
        assert judgment == 1


@pytest.mark.external
class TestLMJudgePairBinary:
    """Test lmjudge_pair_norewrite_binaryhard_gemini annotator."""

    @pytest.mark.skipif(
        not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        reason="GEMINI_API_KEY not set",
    )
    def test_binary_judge_same(self):
        """Test binary judge on same/matching texts (should be 0)."""
        results = run_inference(
            inputs={
                "similarity_criteria": ["Main idea"],
                "document_a": ["The earth is round."],
                "document_b": ["Earth has a spherical shape."],
            },
            config="lmjudge_pair_norewrite_binaryhard_gemini",
        )

        assert len(results) == 1
        judgment = results[0]
        # Should return 0 (same) or 1 (different)
        assert judgment in [0, 1]
        # Same main idea should be judged as 0
        assert judgment == 0

    @pytest.mark.skipif(
        not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        reason="GEMINI_API_KEY not set",
    )
    def test_binary_judge_different(self):
        """Test binary judge on different texts (should be 1)."""
        results = run_inference(
            inputs={
                "similarity_criteria": ["Topic"],
                "document_a": ["Python is a programming language."],
                "document_b": ["Elephants are large mammals."],
            },
            config="lmjudge_pair_norewrite_binaryhard_gemini",
        )

        assert len(results) == 1
        judgment = results[0]
        assert judgment in [0, 1]
        # Different topics should be judged as 1
        assert judgment == 1


class TestAnnotatorWithOverrides:
    """Test using annotators with custom overrides."""

    def test_override_model(self):
        """Test overriding model in annotator preset."""
        config = get_preset("rewrite_plaintext_freeform_gemini")
        modified = config.with_overrides(
            model_name="gemini-1.5-pro",
            max_tokens=1024,
        )

        assert modified.model_name == "gemini-1.5-pro"
        assert modified.max_tokens == 1024
        # Other fields unchanged
        assert modified.provider == "gemini"
        assert modified.parser == "json"

    def test_get_via_main_preset_function(self):
        """Test that annotator presets work with main get_preset()."""
        from multiview.inference import get_preset

        config = get_preset("embed_plaintext_hfapi")
        assert config.provider == "hf_embedding"
        assert (
            config.model_name == "Qwen/Qwen3-Embedding-8B"
        )  # 4B not available via Inference API
