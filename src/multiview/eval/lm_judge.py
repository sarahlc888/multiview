"""LM judge evaluation method.

This module implements evaluation methods that use language models to judge
similarity between documents in triplets.
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


def evaluate_with_lm_judge_triplet(
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    lm_judge_preset: str = "lmjudge_triplet_plaintext_binaryhard_gemini",
    cache_alias: str | None = None,
    annotations: list[dict] | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using an LM judge."""
    if not triplets:
        logger.warning("No triplets provided for evaluation")
        return {
            "outcomes": [],
            "triplet_logs": [],
        }

    logger.info(f"Evaluating {len(triplets)} triplets with LM judge")
    logger.info(f"Using preset: {lm_judge_preset}")

    has_annotations = annotations is not None and len(annotations) > 0

    inputs = {
        "similarity_criteria": [criterion_description or criterion] * len(triplets),
        "document_a": [t["anchor"] for t in triplets],
        "document_b": [t["positive"] for t in triplets],
        "document_c": [t["negative"] for t in triplets],
    }

    if has_annotations:
        logger.info("Using annotations in evaluation")
        add_annotation_summaries_to_inputs(
            inputs,
            triplets=triplets,
            annotations=annotations,
            triplet_keys_by_input_suffix={
                "a": "anchor",
                "b": "positive",
                "c": "negative",
            },
        )

    results, raw_responses = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        verbose=False,
        return_raw=True,
    )

    triplet_logs: list[dict[str, Any]] = []
    crit_desc = criterion_description or criterion

    for i, t in enumerate(triplets):
        outcome = results[i] if i < len(results) else None
        record: dict[str, Any] = {
            "triplet_idx": i,
            "method_type": "lm_judge_triplet",
            "lm_judge_preset": lm_judge_preset,
            "cache_alias": cache_alias,
            "criterion": criterion,
            "criterion_description": crit_desc,
            "outcome": outcome,
            "anchor": t.get("anchor"),
            "positive": t.get("positive"),
            "negative": t.get("negative"),
            "anchor_id": t.get("anchor_id"),
            "positive_id": t.get("positive_id"),
            "negative_id": t.get("negative_id"),
        }

        if i < len(raw_responses):
            record["lm_judge_reasoning"] = _raw_to_text(raw_responses[i])

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

    return {
        "outcomes": results,
        "triplet_logs": triplet_logs,
    }
