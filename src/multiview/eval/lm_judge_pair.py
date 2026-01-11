"""Pairwise LM judge evaluation method.

This module implements evaluation methods that use language models to judge
pairwise similarity between documents, then compute triplet accuracy.
"""

import logging
from typing import Any

from multiview.benchmark.triplets.utils import (
    add_annotation_summaries_to_inputs,
    triplet_annotation_summary,
    triplet_full_annotation,
)
from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def _raw_to_text(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get("text", raw))
    return str(raw)


def evaluate_with_lm_judge_pair(
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    lm_judge_preset: str = "lmjudge_pair_plaintext_likerthard_gemini",
    cache_alias: str | None = None,
    run_name: str | None = None,
    annotations: list[dict] | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using pairwise LM judge scoring.

    Args:
        triplets: List of triplet dictionaries
        criterion: Name of the similarity criterion
        criterion_description: Optional detailed description
        lm_judge_preset: Inference preset to use
        cache_alias: Optional cache identifier
        run_name: Optional experiment/run name for cache organization
        annotations: Optional document annotations
    """
    if not triplets:
        logger.warning("No triplets provided for evaluation")
        return {
            "positive_scores": [],
            "negative_scores": [],
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
            "triplet_logs": [],
        }

    logger.info(f"Evaluating {len(triplets)} triplets with pairwise LM judge")
    logger.info(f"Using preset: {lm_judge_preset}")

    has_annotations = annotations is not None and len(annotations) > 0
    criterion_text = criterion_description or criterion

    positive_pairs_inputs = {
        "similarity_criteria": [criterion_text] * len(triplets),
        "document_a": [t["anchor"] for t in triplets],
        "document_b": [t["positive"] for t in triplets],
    }

    if has_annotations:
        logger.info("Using annotations in evaluation")
        add_annotation_summaries_to_inputs(
            positive_pairs_inputs,
            triplets=triplets,
            annotations=annotations,
            triplet_keys_by_input_suffix={"a": "anchor", "b": "positive"},
        )

    logger.debug("Scoring anchor-positive pairs")
    positive_scores, positive_raw = run_inference(
        inputs=positive_pairs_inputs,
        config=lm_judge_preset,
        cache_alias=f"{cache_alias}_positive" if cache_alias else None,
        run_name=run_name,
        verbose=False,
        return_raw=True,
    )

    negative_pairs_inputs = {
        "similarity_criteria": [criterion_text] * len(triplets),
        "document_a": [t["anchor"] for t in triplets],
        "document_b": [t["negative"] for t in triplets],
    }

    if has_annotations:
        add_annotation_summaries_to_inputs(
            negative_pairs_inputs,
            triplets=triplets,
            annotations=annotations,
            triplet_keys_by_input_suffix={"a": "anchor", "b": "negative"},
        )

    logger.debug("Scoring anchor-negative pairs")
    negative_scores, negative_raw = run_inference(
        inputs=negative_pairs_inputs,
        config=lm_judge_preset,
        cache_alias=f"{cache_alias}_negative" if cache_alias else None,
        run_name=run_name,
        verbose=False,
        return_raw=True,
    )

    triplet_logs: list[dict[str, Any]] = []

    for i, t in enumerate(triplets):
        pos_score = positive_scores[i] if i < len(positive_scores) else None
        neg_score = negative_scores[i] if i < len(negative_scores) else None

        if pos_score is None or neg_score is None:
            outcome = None
        elif pos_score > neg_score:
            outcome = 1
        elif neg_score > pos_score:
            outcome = -1
        else:
            outcome = 0

        record: dict[str, Any] = {
            "triplet_idx": i,
            "method_type": "lm_judge_pair",
            "lm_judge_preset": lm_judge_preset,
            "cache_alias_positive": f"{cache_alias}_positive" if cache_alias else None,
            "cache_alias_negative": f"{cache_alias}_negative" if cache_alias else None,
            "criterion": criterion,
            "criterion_description": criterion_text,
            "positive_score": pos_score,
            "negative_score": neg_score,
            "outcome": outcome,
            "anchor": t.get("anchor"),
            "positive": t.get("positive"),
            "negative": t.get("negative"),
            "anchor_id": t.get("anchor_id"),
            "positive_id": t.get("positive_id"),
            "negative_id": t.get("negative_id"),
        }

        if i < len(positive_raw):
            record["lm_reasoning_positive"] = _raw_to_text(positive_raw[i])
        if i < len(negative_raw):
            record["lm_reasoning_negative"] = _raw_to_text(negative_raw[i])

        if has_annotations:
            record["annotation_anchor_summary"] = triplet_annotation_summary(
                triplets=triplets,
                triplet_idx=i,
                triplet_key="anchor",
                annotations=annotations,
            )
            record["annotation_positive_summary"] = triplet_annotation_summary(
                triplets=triplets,
                triplet_idx=i,
                triplet_key="positive",
                annotations=annotations,
            )
            record["annotation_negative_summary"] = triplet_annotation_summary(
                triplets=triplets,
                triplet_idx=i,
                triplet_key="negative",
                annotations=annotations,
            )
            record["annotation_anchor_full"] = triplet_full_annotation(
                triplets=triplets,
                triplet_idx=i,
                triplet_key="anchor",
                annotations=annotations,
            )
            record["annotation_positive_full"] = triplet_full_annotation(
                triplets=triplets,
                triplet_idx=i,
                triplet_key="positive",
                annotations=annotations,
            )
            record["annotation_negative_full"] = triplet_full_annotation(
                triplets=triplets,
                triplet_idx=i,
                triplet_key="negative",
                annotations=annotations,
            )

        triplet_logs.append(record)

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
