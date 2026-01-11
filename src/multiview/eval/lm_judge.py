"""LM judge evaluation method.

This module implements evaluation methods that use language models to judge
similarity between documents in triplets.

The main function, evaluate_with_lm_judge_triplet(), supports bidirectional
evaluation (default) to eliminate position bias by testing each triplet in
both (b=pos,c=neg) and (b=neg,c=pos) orderings.
"""

from __future__ import annotations

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
    run_name: str | None = None,
    annotations: list[dict] | None = None,
    bidirectional: bool = True,
) -> dict[str, Any]:
    """Evaluate triplets using an LM judge.

    Args:
        triplets: List of triplet dictionaries
        criterion: Name of the similarity criterion
        criterion_description: Optional detailed description
        lm_judge_preset: Inference preset to use
        cache_alias: Optional cache identifier
        run_name: Optional experiment/run name for cache organization
        annotations: Optional document annotations
        bidirectional: If True (default), evaluate each triplet in both directions
            to eliminate position bias:
            - Forward: (a=anchor, b=positive, c=negative) → expect answer (b)
            - Reversed: (a=anchor, b=negative, c=positive) → expect answer (c)
            This produces 2N outcomes from N triplets. The reversed outcomes are
            flipped so both directions contribute correctly to mean accuracy.
            Set to False to restore single-direction evaluation.

    Returns:
        Dict with 'outcomes' (list of 1/-1/0) and 'triplet_logs' (detailed records).
        When bidirectional=True, length is 2N with 'direction' field in logs.
    """
    if not triplets:
        logger.warning("No triplets provided for evaluation")
        return {
            "outcomes": [],
            "triplet_logs": [],
        }

    n = len(triplets)
    has_annotations = annotations is not None and len(annotations) > 0

    if not bidirectional:
        # Single direction evaluation (old behavior)
        logger.info(f"Evaluating {n} triplets with LM judge (single direction)")
        logger.info(f"Using preset: {lm_judge_preset}")

        inputs = {
            "similarity_criteria": [criterion_description or criterion] * n,
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
            run_name=run_name,
            verbose=False,
            return_raw=True,
        )

        triplets_to_log = triplets

    else:
        # Bidirectional evaluation: test both (b=pos,c=neg) and (b=neg,c=pos)
        # to eliminate position bias. Batch both directions in one inference call.
        logger.info(
            f"Evaluating {n} triplets with LM judge (bidirectional: 2x{n} = {2*n} total)"
        )
        logger.info(f"Using preset: {lm_judge_preset}")

        # Direction 1 (forward): b=positive, c=negative
        inputs_forward = {
            "similarity_criteria": [criterion_description or criterion] * n,
            "document_a": [t["anchor"] for t in triplets],
            "document_b": [t["positive"] for t in triplets],
            "document_c": [t["negative"] for t in triplets],
        }

        # Direction 2 (reversed): b=negative, c=positive (SWAPPED)
        inputs_reversed = {
            "similarity_criteria": [criterion_description or criterion] * n,
            "document_a": [t["anchor"] for t in triplets],
            "document_b": [t["negative"] for t in triplets],  # SWAPPED
            "document_c": [t["positive"] for t in triplets],  # SWAPPED
        }

        # Add annotations if present (for both directions)
        if has_annotations:
            logger.info("Using annotations in evaluation")
            # Forward direction annotations
            add_annotation_summaries_to_inputs(
                inputs_forward,
                triplets=triplets,
                annotations=annotations,
                triplet_keys_by_input_suffix={
                    "a": "anchor",
                    "b": "positive",
                    "c": "negative",
                },
            )
            # Reversed direction annotations (swapped b and c)
            add_annotation_summaries_to_inputs(
                inputs_reversed,
                triplets=triplets,
                annotations=annotations,
                triplet_keys_by_input_suffix={
                    "a": "anchor",
                    "b": "negative",  # SWAPPED
                    "c": "positive",  # SWAPPED
                },
            )

        # Combine both directions into single batch
        inputs_combined = {
            key: inputs_forward[key] + inputs_reversed[key]
            for key in inputs_forward.keys()
        }

        # Run inference on all 2N examples at once
        raw_results, raw_responses = run_inference(
            inputs=inputs_combined,
            config=lm_judge_preset,
            cache_alias=cache_alias,
            run_name=run_name,
            verbose=False,
            return_raw=True,
        )

        # Split results into forward and reversed
        results_forward = raw_results[:n]
        results_reversed = raw_results[n:]
        raw_responses_forward = raw_responses[:n]
        raw_responses_reversed = raw_responses[n:]

        # Flip outcomes for reversed direction:
        # In reversed direction, positive is in position (c), so choosing (c) is correct.
        # The parser returns +1 for (b), -1 for (c), but we need to flip this.
        results_reversed_flipped = []
        for outcome in results_reversed:
            if outcome == 1:  # Chose (b) = chose negative → wrong
                results_reversed_flipped.append(-1)
            elif outcome == -1:  # Chose (c) = chose positive → correct
                results_reversed_flipped.append(1)
            else:  # Tie or other
                results_reversed_flipped.append(outcome)

        # Create expanded triplets with direction labels
        triplets_expanded = []
        for t in triplets:
            # Forward direction
            t_forward = t.copy()
            t_forward["direction"] = "forward"
            triplets_expanded.append(t_forward)

            # Reversed direction
            t_reversed = t.copy()
            t_reversed["direction"] = "reversed"
            triplets_expanded.append(t_reversed)

        # Interleave results to match expanded triplets:
        # [trip0_fwd, trip0_rev, trip1_fwd, trip1_rev, ...]
        results = []
        raw_responses = []
        for i in range(n):
            results.append(results_forward[i])
            raw_responses.append(raw_responses_forward[i])
            results.append(results_reversed_flipped[i])
            raw_responses.append(raw_responses_reversed[i])

        triplets_to_log = triplets_expanded

    triplet_logs: list[dict[str, Any]] = []
    crit_desc = criterion_description or criterion

    for i, t in enumerate(triplets_to_log):
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

        # Include direction if present (for bidirectional evaluation)
        if "direction" in t:
            record["direction"] = t["direction"]

        if i < len(raw_responses):
            record["lm_judge_reasoning"] = _raw_to_text(raw_responses[i])

        if has_annotations:
            # For bidirectional evaluation, map expanded index back to original
            orig_idx = i // 2 if bidirectional else i

            record["annotation_anchor_summary"] = triplet_annotation_summary(
                triplets=triplets,
                triplet_idx=orig_idx,
                triplet_key="anchor",
                annotations=annotations,
            )
            record["annotation_positive_summary"] = triplet_annotation_summary(
                triplets=triplets,
                triplet_idx=orig_idx,
                triplet_key="positive",
                annotations=annotations,
            )
            record["annotation_negative_summary"] = triplet_annotation_summary(
                triplets=triplets,
                triplet_idx=orig_idx,
                triplet_key="negative",
                annotations=annotations,
            )
            record["annotation_anchor_full"] = triplet_full_annotation(
                triplets=triplets,
                triplet_idx=orig_idx,
                triplet_key="anchor",
                annotations=annotations,
            )
            record["annotation_positive_full"] = triplet_full_annotation(
                triplets=triplets,
                triplet_idx=orig_idx,
                triplet_key="positive",
                annotations=annotations,
            )
            record["annotation_negative_full"] = triplet_full_annotation(
                triplets=triplets,
                triplet_idx=orig_idx,
                triplet_key="negative",
                annotations=annotations,
            )

        triplet_logs.append(record)

    return {
        "outcomes": results,
        "triplet_logs": triplet_logs,
    }
