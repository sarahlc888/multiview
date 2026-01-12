"""Scoring methods to evaluate.

This package contains implementations of various methods for evaluating triplet accuracy,
including LM judges, BM25, and embedding models.
"""

from __future__ import annotations

from .bm25 import evaluate_with_bm25
from .embeddings import evaluate_with_embeddings
from .lm_judge import evaluate_with_lm_judge_triplet
from .lm_judge_pair import evaluate_with_lm_judge_pair
from .query_expansion import evaluate_with_query_expansion, evaluate_with_query_expansion_bm25

__all__ = [
    "evaluate_with_lm_judge_triplet",
    "evaluate_with_lm_judge_pair",
    "evaluate_with_embeddings",
    "evaluate_with_bm25",
    "evaluate_with_query_expansion",
    "evaluate_with_query_expansion_bm25",  # Backwards compatibility
]
