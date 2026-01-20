"""Scoring methods to evaluate.

This package contains implementations of various methods for evaluating triplet accuracy,
including LM judges, BM25, embedding models, and in-one-word hidden state extraction.
"""

from __future__ import annotations

from .bm25 import evaluate_with_bm25
from .document_summary import evaluate_with_document_rewrite
from .embeddings import evaluate_with_embeddings
from .in_one_word import evaluate_with_in_one_word
from .lm_judge import evaluate_with_lm_judge_triplet
from .lm_judge_pair import evaluate_with_lm_judge_pair
from .multisummary import evaluate_with_multisummary
from .pseudologit import evaluate_with_pseudologit
from .query_relevance_vectors import evaluate_with_query_relevance_vectors
from .reranker import evaluate_with_reranker

__all__ = [
    "evaluate_with_lm_judge_triplet",
    "evaluate_with_lm_judge_pair",
    "evaluate_with_embeddings",
    "evaluate_with_pseudologit",
    "evaluate_with_bm25",
    "evaluate_with_document_rewrite",
    "evaluate_with_multisummary",
    "evaluate_with_query_relevance_vectors",
    "evaluate_with_reranker",
    "evaluate_with_in_one_word",
]
