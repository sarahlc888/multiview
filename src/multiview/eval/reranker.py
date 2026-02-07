"""Reranker-based evaluation method using inference system.

This module implements evaluation methods that use reranker models to compute
relevance scores for query-document pairs, then evaluate triplet accuracy.

This follows the same architecture pattern as embeddings.py - using run_inference()
with reranker presets instead of loading models directly.
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def evaluate_with_reranker(
    documents: list[str | dict],
    triplet_ids: list[tuple[int, int, int]],
    reranker_preset: str = "qwen3_reranker_8b",
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using a reranker model via the inference system.

    For each triplet (anchor, positive, negative):
    - Compute reranker score for (anchor, positive)
    - Compute reranker score for (anchor, negative)
    - Triplet is correct if positive score > negative score

    Args:
        documents: List of documents (text strings or dicts with optional image_path)
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        reranker_preset: Inference preset to use (e.g., "qwen3_reranker_8b")
        cache_alias: Optional cache identifier
        run_name: Optional experiment/run name for cache organization
        preset_overrides: Optional preset configuration overrides
                         Use instruction to set embedding instruction

    Returns:
        Dict with positive_scores, negative_scores, and triplet_logs
    """
    if not triplet_ids:
        logger.warning("No triplets provided for evaluation")
        return {
            "positive_scores": [],
            "negative_scores": [],
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
            "triplet_logs": [],
        }

    logger.info(f"Evaluating {len(triplet_ids)} triplets with reranker")
    logger.info(f"Using preset: {reranker_preset}")

    # Prepare inputs for all query-document pairs
    # For each triplet, we need two scores: (anchor, positive) and (anchor, negative)
    queries: list[str | dict] = []
    docs: list[str | dict] = []

    for anchor_id, positive_id, negative_id in triplet_ids:
        # Add (anchor, positive) pair
        queries.append(documents[anchor_id])
        docs.append(documents[positive_id])

        # Add (anchor, negative) pair
        queries.append(documents[anchor_id])
        docs.append(documents[negative_id])

    # Get all scores using run_inference
    # Instructions are handled via instruction field in preset_overrides
    logger.info(
        f"Computing scores for {len(queries)} query-document pairs via inference system"
    )

    inference_kwargs = {"verbose": False}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)
    scores = run_inference(
        inputs={
            "query": queries,
            "document": docs,
        },
        config=reranker_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )

    # Split scores back into positive and negative
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    for i in range(0, len(scores), 2):
        pos_score = scores[i]
        neg_score = scores[i + 1]

        positive_scores.append(pos_score)
        negative_scores.append(neg_score)

        triplet_idx = i // 2
        anchor_id, positive_id, negative_id = triplet_ids[triplet_idx]

        triplet_logs.append(
            {
                "triplet_idx": triplet_idx,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "positive_score": pos_score,
                "negative_score": neg_score,
                "correct": pos_score > neg_score,
                "margin": pos_score - neg_score,
            }
        )

    # Compute aggregate statistics
    avg_positive = (
        sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    )
    avg_negative = (
        sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    )

    logger.info(f"Average positive score: {avg_positive:.4f}")
    logger.info(f"Average negative score: {avg_negative:.4f}")

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "avg_positive_score": avg_positive,
        "avg_negative_score": avg_negative,
        "triplet_logs": triplet_logs,
    }
