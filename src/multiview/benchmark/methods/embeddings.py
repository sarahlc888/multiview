"""Embedding-based evaluation method.

This module implements evaluation methods that use embedding models to compute
similarity scores via cosine similarity, then evaluate triplet accuracy.
"""

import logging
from typing import Any

import numpy as np

from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    # Handle zero vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def evaluate_with_embeddings(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    embedding_preset: str = "openai_embedding_small",
    cache_alias: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using embedding-based cosine similarity.

    This method computes embeddings for all documents, calculates cosine similarity
    between anchor-positive and anchor-negative pairs, then evaluates triplet accuracy.

    Args:
        documents: List of all document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
            where IDs are indices into the documents list
        embedding_preset: Preset name for embedding model (default: openai_embedding_small)
        cache_alias: Cache alias for inference caching
        preset_overrides: Dict of config overrides to pass to run_inference
            (e.g., {"embed_query_instr_template": None} to remove instruction prefix)

    Returns:
        Dict with evaluation metrics:
            - accuracy: Float between 0 and 1
            - n_correct: Number of triplets where positive scored higher than negative
            - n_incorrect: Number of triplets where negative scored higher than positive
            - n_ties: Number of triplets with identical scores
            - n_total: Total number of triplets evaluated
            - avg_positive_score: Average cosine similarity for anchor-positive pairs
            - avg_negative_score: Average cosine similarity for anchor-negative pairs

    Example:
        >>> documents = ["doc1", "doc2", "doc3"]
        >>> triplet_ids = [(0, 1, 2)]
        >>> results = evaluate_with_embeddings(
        ...     documents,
        ...     triplet_ids,
        ...     embedding_preset="openai_embedding_small"
        ... )
        >>> print(f"Accuracy: {results['accuracy']:.2f}")
        Accuracy: 0.92
    """
    if not triplet_ids:
        logger.warning("No triplets provided for evaluation")
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_incorrect": 0,
            "n_ties": 0,
            "n_total": 0,
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
        }

    logger.info(f"Evaluating {len(triplet_ids)} triplets with embeddings")
    logger.info(f"Using preset: {embedding_preset}")
    logger.info(f"Computing embeddings for {len(documents)} documents")

    # Get embeddings for all documents
    # Build kwargs for run_inference, including optional overrides
    inference_kwargs = {"verbose": False}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)

    embeddings = run_inference(
        inputs={"document": documents},
        config=embedding_preset,
        cache_alias=cache_alias,
        **inference_kwargs,
    )

    # Evaluate each triplet using cosine similarity
    n_correct = 0
    n_incorrect = 0
    n_ties = 0
    positive_scores = []
    negative_scores = []

    for anchor_id, positive_id, negative_id in triplet_ids:
        anchor_emb = embeddings[anchor_id]
        positive_emb = embeddings[positive_id]
        negative_emb = embeddings[negative_id]

        # Compute cosine similarities
        pos_score = cosine_similarity(anchor_emb, positive_emb)
        neg_score = cosine_similarity(anchor_emb, negative_emb)

        positive_scores.append(pos_score)
        negative_scores.append(neg_score)

        # Compare scores
        if pos_score > neg_score:
            n_correct += 1
        elif neg_score > pos_score:
            n_incorrect += 1
        else:
            n_ties += 1

    n_total = len(triplet_ids)

    # Calculate accuracy (excluding ties)
    n_judged = n_correct + n_incorrect
    accuracy = n_correct / n_judged if n_judged > 0 else 0.0

    # Calculate average scores
    avg_positive_score = (
        sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    )
    avg_negative_score = (
        sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    )

    logger.info(f"Evaluation complete: {n_correct}/{n_total} correct ({accuracy:.2%})")
    logger.info(
        f"Average positive score: {avg_positive_score:.3f}, Average negative score: {avg_negative_score:.3f}"
    )
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
        "avg_positive_score": avg_positive_score,
        "avg_negative_score": avg_negative_score,
    }
