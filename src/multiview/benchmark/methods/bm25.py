"""BM25 evaluation method.

This module implements evaluation methods that use BM25 scoring to judge
similarity between documents in triplets.
"""

import logging
from typing import Any

from multiview.benchmark.bm25_utils import compute_bm25_matrix

logger = logging.getLogger(__name__)


def evaluate_with_bm25(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    use_annotations: bool = False,
) -> dict[str, Any]:
    """Evaluate triplets using BM25 scoring.

    This method builds a BM25 index over all documents and scores each triplet
    by comparing BM25 scores between anchor-positive and anchor-negative pairs.

    Args:
        documents: List of all document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
            where IDs are indices into the documents list
        use_annotations: If True, use annotation fields instead of raw document text
            (Note: annotation support temporarily simplified)

    Returns:
        Dict with evaluation metrics:
            - accuracy: Float between 0 and 1
            - n_correct: Number of triplets where positive scored higher than negative
            - n_incorrect: Number of triplets where negative scored higher than positive
            - n_ties: Number of triplets with identical scores
            - n_total: Total number of triplets evaluated

    Example:
        >>> documents = ["doc1", "doc2", "doc3"]
        >>> triplet_ids = [(0, 1, 2)]
        >>> results = evaluate_with_bm25(documents, triplet_ids)
        >>> print(f"Accuracy: {results['accuracy']:.2f}")
        Accuracy: 0.65
    """
    if not triplet_ids:
        logger.warning("No triplets provided for evaluation")
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_incorrect": 0,
            "n_ties": 0,
            "n_total": 0,
        }

    logger.info(f"Evaluating {len(triplet_ids)} triplets with BM25")
    logger.info(f"Building BM25 matrix over {len(documents)} documents")

    # Precompute BM25 similarity matrix once for all documents
    similarity_matrix = compute_bm25_matrix(documents)

    # Evaluate each triplet
    n_correct = 0
    n_incorrect = 0
    n_ties = 0

    for anchor_id, positive_id, negative_id in triplet_ids:
        # Get scores from precomputed matrix
        pos_score = similarity_matrix[anchor_id][positive_id]
        neg_score = similarity_matrix[anchor_id][negative_id]

        # Compare scores
        if pos_score > neg_score:
            n_correct += 1
        elif neg_score > pos_score:
            n_incorrect += 1
        else:
            n_ties += 1

    n_total = n_correct + n_incorrect + n_ties

    # Calculate accuracy (excluding ties)
    n_judged = n_correct + n_incorrect
    accuracy = n_correct / n_judged if n_judged > 0 else 0.0

    logger.info(f"Evaluation complete: {n_correct}/{n_total} correct ({accuracy:.2%})")
    if n_ties > 0:
        logger.info(
            f"Note: {n_ties} triplets had identical scores (excluded from accuracy)"
        )

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_ties": n_ties,
        "n_total": n_total,
    }
