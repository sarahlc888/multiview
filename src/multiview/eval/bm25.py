"""BM25 evaluation method.

This module implements evaluation methods that use BM25 scoring to judge
similarity between documents in triplets.
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.utils.bm25_utils import compute_bm25_matrix

logger = logging.getLogger(__name__)


def evaluate_with_bm25(
    documents: list[str | dict],
    triplet_ids: list[tuple[int, int, int]],
    preset: str = "bm25_lexical",
) -> dict[str, Any]:
    """Evaluate triplets using BM25 scoring.

    Args:
        documents: List of documents (strings or dicts)
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        preset: BM25 preset name (e.g., "bm25_lexical", "bm25_raw")

    Returns:
        Dict with positive_scores, negative_scores, and triplet_logs
    """
    if not triplet_ids:
        logger.warning("No triplets provided for evaluation")
        return {
            "positive_scores": [],
            "negative_scores": [],
            "triplet_logs": [],
        }

    logger.info(f"Evaluating {len(triplet_ids)} triplets with BM25 (preset={preset})")
    logger.info(f"Building BM25 matrix over {len(documents)} documents")

    similarity_matrix = compute_bm25_matrix(documents)

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        pos_score = similarity_matrix[anchor_id][positive_id]
        neg_score = similarity_matrix[anchor_id][negative_id]
        positive_scores.append(float(pos_score))
        negative_scores.append(float(neg_score))

        if pos_score > neg_score:
            outcome = 1
        elif neg_score > pos_score:
            outcome = -1
        else:
            outcome = 0

        triplet_logs.append(
            {
                "triplet_idx": i,
                "method_type": "bm25",
                "preset": preset,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "anchor": documents[anchor_id],
                "positive": documents[positive_id],
                "negative": documents[negative_id],
                "positive_score": float(pos_score),
                "negative_score": float(neg_score),
                "outcome": outcome,
                "correct": outcome == 1,
                "is_tie": outcome == 0,
            }
        )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "triplet_logs": triplet_logs,
        "similarity_matrix": similarity_matrix,  # Return full matrix for visualization
    }
