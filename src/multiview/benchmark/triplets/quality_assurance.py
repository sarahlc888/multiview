"""Triplet quality utilities.

This module lives under `benchmark/triplets/` because quality is used as a
selection/filtering step on triplets (even though the quality rating itself is
produced by an LM judge).
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.benchmark.triplets.utils import add_annotation_summaries_to_inputs
from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)

# Single source of truth for the 1-4 quality scale.
QUALITY_SCALE: dict[int, dict[str, str]] = {
    1: {
        "label": "invalid",
        "class": "Invalid",
        "description": "invalid - the anchor is not closer to pos than neg",
    },
    2: {
        "label": "ambiguous",
        "class": "Ambiguous",
        "description": "ambiguous - the anchor is arguably closer to pos than neg but it is very ambiguous",
    },
    3: {
        "label": "trivial",
        "class": "Trivial",
        "description": "trivial - the anchor is closer to pos than neg, but this is obvious and the triplet is not challenging at all",
    },
    4: {
        "label": "ideal",
        "class": "Ideal",
        "description": "ideal - it's a good hard negative triplet",
    },
}


def rate_triplet_quality(
    *,
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    lm_judge_preset: str = "lmjudge_quality_rating_gemini",
    cache_alias: str | None = None,
    run_name: str | None = None,
    annotations: list[dict] | None = None,
) -> dict[str, Any]:
    """Rate the quality of triplets using an LM judge.

    This method uses a language model to assess triplet quality on a 1-4 scale:
    1. Invalid - anchor is not closer to positive than negative
    2. Ambiguous - arguably closer but very ambiguous
    3. Trivial - obviously closer, not challenging
    4. Ideal - good hard negative triplet

    Args:
        triplets: List of triplet dicts with anchor/positive/negative
        criterion: Criterion name for similarity assessment
        criterion_description: Optional detailed description of criterion
        lm_judge_preset: Preset for LM judge
        cache_alias: Optional cache alias for inference calls
        run_name: Optional experiment/run name for cache organization
        annotations: Optional list of annotations for triplet documents

    Returns:
        Dict with ratings, counts, percentages, and triplets_with_ratings
    """
    if not triplets:
        logger.warning("No triplets provided for quality rating")
        return {
            "ratings": [],
            "counts": {1: 0, 2: 0, 3: 0, 4: 0},
            "percentages": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
            "n_total": 0,
            "triplets_with_ratings": [],
        }

    logger.info(f"Rating quality of {len(triplets)} triplets with LM judge")
    logger.info(f"Using preset: {lm_judge_preset}")

    # Check if we have annotations
    has_annotations = annotations is not None and len(annotations) > 0

    # Prepare inputs for batch inference
    inputs = {
        "similarity_criteria": [criterion_description or criterion] * len(triplets),
        "document_a": [t["anchor"] for t in triplets],
        "document_b": [t["positive"] for t in triplets],
        "document_c": [t["negative"] for t in triplets],
    }

    # Add annotations if provided
    if has_annotations:
        logger.info("Using annotations in quality rating")
        add_annotation_summaries_to_inputs(
            inputs=inputs,
            triplets=triplets,
            annotations=annotations,
            triplet_keys_by_input_suffix={
                "a": "anchor",
                "b": "positive",
                "c": "negative",
            },
        )

    # Run inference
    ratings = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )

    # Count ratings
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    n_invalid = 0
    for rating in ratings:
        if rating in counts:
            counts[rating] += 1
        else:
            logger.error(
                f"⚠️  PARSING FAILED: Unexpected quality rating: {rating}. "
                f"This should not happen - check LM judge output format!"
            )
            n_invalid += 1

    n_total = len(ratings)
    n_valid = n_total - n_invalid
    # Calculate percentages based on valid ratings only
    percentages = {
        k: (v / n_valid * 100) if n_valid > 0 else 0.0 for k, v in counts.items()
    }

    # Create triplets with ratings attached
    triplets_with_ratings = []
    for triplet, rating in zip(triplets, ratings, strict=False):
        triplet_with_rating = triplet.copy()
        triplet_with_rating["quality_rating"] = rating
        # If rating is not in scale (already warned above), use "unknown" labels
        quality_info = QUALITY_SCALE.get(
            rating, {"label": "unknown", "class": "Unknown"}
        )
        triplet_with_rating["quality_label"] = quality_info.get("label", "unknown")
        triplet_with_rating["quality_class"] = quality_info.get("class", "Unknown")
        triplets_with_ratings.append(triplet_with_rating)

    # Log summary
    logger.info("Quality rating complete:")
    for level in [1, 2, 3, 4]:
        label = QUALITY_SCALE[level]["label"]
        count = counts[level]
        pct = percentages[level]
        logger.info(f"  {level} ({label:10s}): {count:4d} ({pct:5.1f}%)")
    if n_invalid > 0:
        logger.warning(
            f"  Invalid/None ratings: {n_invalid} ({n_invalid/n_total*100:.1f}%) - parsing failed"
        )

    return {
        "ratings": ratings,
        "counts": counts,
        "percentages": percentages,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "triplets_with_ratings": triplets_with_ratings,
    }


def filter_triplets_by_quality(
    triplets: list[dict],
    *,
    min_quality: int = 3,
) -> tuple[list[dict], dict[str, Any]]:
    """Filter triplets by minimum quality rating.

    Args:
        triplets: List of triplet dicts with "quality_rating" key
        min_quality: Minimum quality rating to keep (1-4, default 3)

    Returns:
        Tuple of (filtered_triplets, filter_stats).
    """
    if not triplets:
        return [], {
            "n_total": 0,
            "n_kept": 0,
            "n_filtered": 0,
            "min_quality": min_quality,
        }

    filtered_triplets = [
        t
        for t in triplets
        if (min_quality is None or t.get("quality_rating", 0) >= min_quality)
    ]

    # Check if any triplets were missing quality ratings
    n_missing_ratings = sum(1 for t in triplets if "quality_rating" not in t)
    if n_missing_ratings > 0:
        logger.error(
            f"⚠️  FALLBACK TRIGGERED: {n_missing_ratings}/{len(triplets)} triplets missing 'quality_rating' key. "
            f"These will be treated as quality=0 and filtered out. "
            f"Did you forget to call rate_triplet_quality() before filtering?"
        )

    n_total = len(triplets)
    n_kept = len(filtered_triplets)
    n_filtered = n_total - n_kept

    logger.info(f"Filtered triplets by quality >= {min_quality}")
    logger.info(f"  Kept: {n_kept}/{n_total} ({n_kept/n_total*100:.1f}%)")
    logger.info(
        f"  Filtered out: {n_filtered}/{n_total} ({n_filtered/n_total*100:.1f}%)"
    )

    return filtered_triplets, {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_filtered": n_filtered,
        "min_quality": min_quality,
        "filter_rate": n_filtered / n_total if n_total > 0 else 0.0,
    }
