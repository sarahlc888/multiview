"""Tests for BM25 utilities with advanced tokenization."""

import numpy as np
import pytest

from multiview.benchmark.bm25_utils import (
    LUCENE_STOPWORDS,
    compute_bm25_matrix,
    compute_bm25_scores,
    tokenize_lucene_like,
)


class TestTokenizeLuceneLike:
    """Tests for tokenize_lucene_like function."""

    def test_basic_tokenization(self):
        """Test basic lowercase and splitting."""
        text = "Hello World"
        tokens = tokenize_lucene_like(text)
        assert tokens == ["hello", "world"]

    def test_stopword_removal(self):
        """Test that stopwords are removed."""
        text = "the quick brown fox"
        tokens = tokenize_lucene_like(text)
        # "the" should be removed
        assert "the" not in tokens
        assert tokens == ["quick", "brown", "fox"]

    def test_all_stopwords_removed(self):
        """Test that all Lucene stopwords are filtered."""
        # Test a sample of stopwords
        for stopword in ["the", "and", "is", "to", "for", "with"]:
            tokens = tokenize_lucene_like(f"test {stopword} word")
            assert stopword not in tokens
            assert tokens == ["test", "word"]

    def test_possessive_removal(self):
        """Test that possessives are correctly removed."""
        # Test 's removal
        tokens = tokenize_lucene_like("the company's data")
        assert tokens == ["company", "data"]

        # Test s' removal (removes trailing apostrophe only)
        tokens = tokenize_lucene_like("the students' work")
        assert tokens == ["students", "work"]

    def test_unicode_normalization(self):
        """Test Unicode NFKC normalization."""
        # Full-width characters should be normalized to ASCII
        text = "ｈｅｌｌｏ"  # Full-width
        tokens = tokenize_lucene_like(text)
        assert tokens == ["hello"]

        # Test with regular ASCII too
        text = "café"
        tokens = tokenize_lucene_like(text)
        assert "café" in tokens or "cafe" in tokens  # NFKC may normalize accents

    def test_min_length_filtering(self):
        """Test minimum token length filtering."""
        text = "a bb ccc"
        tokens = tokenize_lucene_like(text, min_token_length=2)
        # "a" should be filtered out (length 1)
        assert tokens == ["bb", "ccc"]

        tokens = tokenize_lucene_like(text, min_token_length=3)
        assert tokens == ["ccc"]

    def test_empty_input(self):
        """Test empty input handling."""
        assert tokenize_lucene_like("") == []
        assert tokenize_lucene_like(None) == []
        assert tokenize_lucene_like("   ") == []

    def test_disable_stopword_removal(self):
        """Test disabling stopword removal."""
        text = "the quick brown fox"
        tokens = tokenize_lucene_like(text, remove_stopwords=False)
        assert "the" in tokens
        assert tokens == ["the", "quick", "brown", "fox"]

    def test_punctuation_handling(self):
        """Test that punctuation is properly handled."""
        text = "Hello, world! How are you?"
        tokens = tokenize_lucene_like(text)
        # Punctuation should be removed, and "are" is a stopword
        assert tokens == ["hello", "world", "how", "you"]

    def test_contractions(self):
        """Test that contractions are kept together."""
        text = "we're going there"
        tokens = tokenize_lucene_like(text)
        # "we're" should be kept as one token, then processed
        assert "we're" in tokens or "were" in tokens

    def test_numbers(self):
        """Test that numbers are tokenized."""
        text = "test123 456"
        tokens = tokenize_lucene_like(text)
        assert "test123" in tokens
        assert "456" in tokens


class TestComputeBM25Scores:
    """Tests for compute_bm25_scores function."""

    def test_basic_scoring(self):
        """Test basic BM25 scoring."""
        docs = ["machine learning", "machine vision", "natural language"]
        scores = compute_bm25_scores(docs, anchor_idx=0)

        assert len(scores) == 3
        assert isinstance(scores, np.ndarray)
        # "machine vision" should be more similar to "machine learning" than "natural language"
        assert scores[1] > scores[2]

    def test_anchor_idx_negative(self):
        """Test negative anchor index (e.g., -1 for last)."""
        docs = ["hello world", "world peace", "hello peace"]
        scores = compute_bm25_scores(docs, anchor_idx=-1)

        assert len(scores) == 3
        # Should be same as anchor_idx=2
        scores_pos = compute_bm25_scores(docs, anchor_idx=2)
        np.testing.assert_array_equal(scores, scores_pos)

    def test_empty_documents_error(self):
        """Test that empty documents raise ValueError."""
        with pytest.raises(ValueError, match="documents cannot be empty"):
            compute_bm25_scores([], anchor_idx=0)

    def test_invalid_anchor_idx(self):
        """Test that invalid anchor index raises ValueError."""
        docs = ["hello world", "world peace"]
        with pytest.raises(ValueError, match="out of range"):
            compute_bm25_scores(docs, anchor_idx=5)

    def test_all_empty_docs_error(self):
        """Test that all empty documents raise ValueError."""
        docs = ["", "   ", ""]
        with pytest.raises(ValueError, match="All documents tokenized to empty"):
            compute_bm25_scores(docs, anchor_idx=0)

    def test_stopword_improvement(self):
        """Test that stopword removal improves scoring."""
        # Documents with many stopwords should still match well
        docs = [
            "the company data is important",
            "company data matters",
            "random other text"
        ]
        scores = compute_bm25_scores(docs, anchor_idx=0)

        # Doc 1 should be more similar to doc 0 than doc 2
        assert scores[1] > scores[2]

    def test_custom_tokenizer(self):
        """Test using a custom tokenizer."""
        docs = ["HELLO WORLD", "WORLD PEACE", "HELLO PEACE"]

        # Custom tokenizer that doesn't lowercase
        def custom_tokenizer(text):
            return text.split()

        scores = compute_bm25_scores(docs, anchor_idx=0, tokenizer=custom_tokenizer)
        assert len(scores) == 3


class TestComputeBM25Matrix:
    """Tests for compute_bm25_matrix function."""

    def test_matrix_shape(self):
        """Test that matrix has correct shape."""
        docs = ["hello world", "world peace", "hello peace"]
        matrix = compute_bm25_matrix(docs)

        assert matrix.shape == (3, 3)
        assert isinstance(matrix, np.ndarray)

    def test_diagonal_values(self):
        """Test that diagonal values are computed correctly."""
        docs = ["unique document one", "unique document two", "unique document three"]
        matrix = compute_bm25_matrix(docs)

        # Diagonal values should exist and be non-negative scores
        # Note: BM25 doesn't guarantee self-similarity is highest
        for i in range(3):
            assert matrix[i][i] is not None
            assert not np.isnan(matrix[i][i])

    def test_matrix_consistency_with_scores(self):
        """Test that matrix matches compute_bm25_scores."""
        docs = ["hello world", "world peace", "hello peace"]
        matrix = compute_bm25_matrix(docs)

        # Compare with compute_bm25_scores
        for anchor_idx in range(3):
            scores = compute_bm25_scores(docs, anchor_idx=anchor_idx)
            np.testing.assert_array_almost_equal(matrix[anchor_idx], scores)

    def test_empty_documents_error(self):
        """Test that empty documents raise ValueError."""
        with pytest.raises(ValueError, match="documents cannot be empty"):
            compute_bm25_matrix([])

    def test_all_empty_docs_error(self):
        """Test that all empty documents raise ValueError."""
        docs = ["", "   ", ""]
        with pytest.raises(ValueError, match="All documents tokenized to empty"):
            compute_bm25_matrix(docs)

    def test_large_matrix(self):
        """Test with a larger document set."""
        docs = [f"document {i} with unique text {i*10}" for i in range(10)]
        matrix = compute_bm25_matrix(docs)

        assert matrix.shape == (10, 10)
        # Verify all values are computed (no NaN)
        assert not np.isnan(matrix).any()

    def test_custom_tokenizer(self):
        """Test using a custom tokenizer."""
        docs = ["HELLO WORLD", "WORLD PEACE", "HELLO PEACE"]

        # Custom tokenizer that doesn't lowercase
        def custom_tokenizer(text):
            return text.split()

        matrix = compute_bm25_matrix(docs, tokenizer=custom_tokenizer)
        assert matrix.shape == (3, 3)


class TestIntegration:
    """Integration tests for BM25 utilities."""

    def test_realistic_documents(self):
        """Test with realistic documents containing possessives and stopwords."""
        docs = [
            "The company's data is stored in the cloud.",
            "Students' work should be submitted by Friday.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Compute scores
        scores = compute_bm25_scores(docs, anchor_idx=0)
        assert len(scores) == 3

        # Compute matrix
        matrix = compute_bm25_matrix(docs)
        assert matrix.shape == (3, 3)

        # Verify consistency
        np.testing.assert_array_almost_equal(matrix[0], scores)

    def test_stopwords_constant(self):
        """Test that LUCENE_STOPWORDS is properly defined."""
        assert isinstance(LUCENE_STOPWORDS, frozenset)
        assert len(LUCENE_STOPWORDS) == 33
        assert "the" in LUCENE_STOPWORDS
        assert "and" in LUCENE_STOPWORDS
        assert "is" in LUCENE_STOPWORDS

    def test_end_to_end_workflow(self):
        """Test complete workflow from tokenization to scoring."""
        # Sample documents
        docs = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning is a type of machine learning",
            "Natural language processing uses machine learning",
            "Computer vision is an application of deep learning",
        ]

        # Test tokenization on first doc
        tokens = tokenize_lucene_like(docs[0])
        assert "the" not in tokens  # Stopword removed
        assert "machine" in tokens
        assert "learning" in tokens

        # Test scoring
        scores = compute_bm25_scores(docs, anchor_idx=0)
        assert len(scores) == 4
        # Docs with "machine learning" should score high
        assert scores[1] > scores[3]  # Deep learning doc vs computer vision doc

        # Test matrix
        matrix = compute_bm25_matrix(docs)
        assert matrix.shape == (4, 4)

        # Verify that the matrix approach gives same results as scores
        for i in range(4):
            scores_i = compute_bm25_scores(docs, anchor_idx=i)
            np.testing.assert_array_almost_equal(matrix[i], scores_i)
