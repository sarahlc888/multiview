"""Pseudologit-based evaluation method.

This module implements evaluation methods that use pseudologit embeddings
(distribution vectors over taxonomy classes) to compute similarity scores
via cosine similarity, then evaluate triplet accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.eval.similarity import cosine_similarity
from multiview.inference.inference import run_inference
from multiview.inference.presets import get_preset

logger = logging.getLogger(__name__)


def evaluate_with_pseudologit(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    preset: str = "pseudologit_gemini_n100",
    classes_file: str = "prompts/custom/gsm8k_classes.json",
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using pseudologit embedding-based cosine similarity.

    Args:
        documents: List of document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        preset: Inference preset to use (default: pseudologit_gemini_n100)
        classes_file: Path to JSON file with taxonomy classes
        cache_alias: Optional cache identifier
        run_name: Optional experiment/run name for cache organization
        preset_overrides: Optional preset configuration overrides

    Returns:
        Dict with evaluation results including scores, averages, and logs
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

    logger.info(f"Evaluating {len(triplet_ids)} triplets with pseudologit embeddings")
    logger.info(f"Using preset: {preset}")
    logger.info(f"Using classes file: {classes_file}")
    logger.info(f"Computing pseudologit embeddings for {len(documents)} documents")

    # Merge classes_file into extra_kwargs for the preset
    preset_config = get_preset(preset)
    merged_extra_kwargs = preset_config.extra_kwargs.copy()
    merged_extra_kwargs["classes_file"] = classes_file

    inference_kwargs = {"verbose": False, "extra_kwargs": merged_extra_kwargs}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)

    # Generate pseudologit embeddings
    embeddings = run_inference(
        inputs={"document": documents},
        config=preset,
        cache_alias=cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        anchor_emb = embeddings[anchor_id]
        positive_emb = embeddings[positive_id]
        negative_emb = embeddings[negative_id]

        pos_score = cosine_similarity(anchor_emb, positive_emb)
        neg_score = cosine_similarity(anchor_emb, negative_emb)

        positive_scores.append(pos_score)
        negative_scores.append(neg_score)

        if pos_score > neg_score:
            outcome = 1
        elif neg_score > pos_score:
            outcome = -1
        else:
            outcome = 0

        triplet_logs.append(
            {
                "triplet_idx": i,
                "method_type": "pseudologit",
                "preset": preset,
                "classes_file": classes_file,
                "cache_alias": cache_alias,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "anchor": documents[anchor_id],
                "positive": documents[positive_id],
                "negative": documents[negative_id],
                "positive_score": pos_score,
                "negative_score": neg_score,
                "outcome": outcome,
                "correct": outcome == 1,
                "is_tie": outcome == 0,
            }
        )

    avg_positive_score = (
        sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    )
    avg_negative_score = (
        sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    )

    logger.info(
        f"Average positive score: {avg_positive_score:.3f}, Average negative score: {avg_negative_score:.3f}"
    )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "avg_positive_score": avg_positive_score,
        "avg_negative_score": avg_negative_score,
        "triplet_logs": triplet_logs,
    }
