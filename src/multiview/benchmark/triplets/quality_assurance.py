"""Triplet quality utilities.

Rate and filter triplets based on quality (1-5 scale) using LM judge.

Quality Scale
-------------
1 - Invalid: Anchor NOT closer to positive than negative (incorrect triplet)
2 - Weak/Borderline: Somewhat unclear which is more similar
3 - Trivial: Correct but too easy (negative completely unrelated)
4 - Acceptable: Reasonable difficulty, usable for evaluation
5 - Ideal: Excellent hard negative, highly discriminative

Usage
-----
    from multiview.benchmark.triplets.quality_assurance import (
        rate_and_filter_quality_workflow,
    )

    results = rate_and_filter_quality_workflow(
        triplets=triplets,
        criterion="arithmetic",
        min_quality=4,  # Keep only acceptable/ideal (filters ~40-60%)
    )

Common min_quality values:
- 3: Production benchmarks (filters ~10-20%)
- 4: Challenging benchmarks (filters ~40-60%)
- None: Rate only, no filtering
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.benchmark.triplets.utils import add_annotation_summaries_to_inputs
from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)

# Single source of truth for the 1-5 quality scale.
QUALITY_SCALE: dict[int, dict[str, str]] = {
    1: {
        "label": "invalid",
        "class": "Invalid",
        "description": "invalid - the anchor is not closer to pos than neg",
    },
    2: {
        "label": "weak/borderline",
        "class": "Weak/Borderline",
        "description": "weak/borderline - the anchor is arguably closer to pos than neg but it is somewhat unclear",
    },
    3: {
        "label": "trivial",
        "class": "Trivial",
        "description": "trivial - the anchor is closer to pos than neg, but this is obvious and the triplet is not challenging at all",
    },
    4: {
        "label": "acceptable",
        "class": "Acceptable",
        "description": "acceptable - the anchor is closer to pos than neg with reasonable difficulty, forming a usable triplet",
    },
    5: {
        "label": "ideal",
        "class": "Ideal",
        "description": "ideal - it's a good hard negative triplet with excellent difficulty balance",
    },
}


def _rate_triplet_quality(
    *,
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    lm_judge_preset: str = "lmjudge_quality_rating_gemini",
    cache_alias: str | None = None,
    run_name: str | None = None,
    annotations: list[dict] | None = None,
) -> dict[str, Any]:
    """Rate triplet quality using LM judge on 1-5 scale.

    Returns dict with ratings, counts, percentages, and triplets_with_ratings.
    """
    if not triplets:
        return _empty_rating_result()

    logger.info(f"Rating {len(triplets)} triplets with {lm_judge_preset}")

    # Prepare inputs and run inference
    inputs = _build_quality_rating_inputs(
        triplets, criterion, criterion_description, annotations
    )
    ratings, raw_outputs = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
        return_raw=True,
    )

    # Process and return results
    return _process_quality_ratings(triplets, ratings, raw_outputs)


def _empty_rating_result() -> dict[str, Any]:
    """Return empty rating result."""
    return {
        "ratings": [],
        "counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "percentages": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        "n_total": 0,
        "triplets_with_ratings": [],
    }


def _has_any_annotation_content(annotations: list[dict] | None) -> bool:
    """Check if annotations list has any meaningful content."""
    if not annotations:
        return False

    # Sample first few to check if any have content
    for ann in annotations[: min(5, len(annotations))]:
        if not isinstance(ann, dict):
            continue
        # Check for any meaningful fields
        if ann.get("category") or ann.get("tags") or ann.get("summary"):
            return True

    return False


def _filter_preset_ratings_to_match_triplets(
    result: dict,
    source_triplets: list[dict],
    selected_triplets: list[dict],
) -> None:
    """Filter individual preset ratings/reasoning to match selected triplets.

    Modifies result dict in place to update ratings_with_annotations,
    ratings_without_annotations, and corresponding reasoning lists.

    Args:
        result: Result dict containing ratings and reasoning lists
        source_triplets: Source list of triplets to build index mapping from
        selected_triplets: Filtered/selected triplets to keep
    """
    if (
        "ratings_with_annotations" not in result
        or "ratings_without_annotations" not in result
    ):
        return

    # Build mapping from triplet to index in source list
    triplet_to_idx = {
        (t["anchor_id"], t["positive_id"], t["negative_id"]): i
        for i, t in enumerate(source_triplets)
    }

    # Extract ratings/reasoning for selected triplets only
    result["ratings_with_annotations"] = [
        result["ratings_with_annotations"][
            triplet_to_idx[(t["anchor_id"], t["positive_id"], t["negative_id"])]
        ]
        for t in selected_triplets
    ]
    result["ratings_without_annotations"] = [
        result["ratings_without_annotations"][
            triplet_to_idx[(t["anchor_id"], t["positive_id"], t["negative_id"])]
        ]
        for t in selected_triplets
    ]
    result["reasoning_with_annotations"] = [
        result["reasoning_with_annotations"][
            triplet_to_idx[(t["anchor_id"], t["positive_id"], t["negative_id"])]
        ]
        for t in selected_triplets
    ]
    result["reasoning_without_annotations"] = [
        result["reasoning_without_annotations"][
            triplet_to_idx[(t["anchor_id"], t["positive_id"], t["negative_id"])]
        ]
        for t in selected_triplets
    ]


def _build_quality_rating_inputs(
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None,
    annotations: list[dict] | None,
) -> dict:
    """Build inputs for quality rating inference."""
    from multiview.benchmark.annotations.annotation_utils import (
        add_document_inputs_with_images,
    )

    similarity_criteria = (
        f"{criterion}: {criterion_description}" if criterion_description else criterion
    )

    inputs = {
        "similarity_criteria": [similarity_criteria] * len(triplets),
    }

    # Use shared function to handle document/image extraction properly
    add_document_inputs_with_images(
        inputs,
        {
            "a": [t["anchor"] for t in triplets],
            "b": [t["positive"] for t in triplets],
            "c": [t["negative"] for t in triplets],
        },
    )

    if annotations:
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

    return inputs


def _process_quality_ratings(
    triplets: list[dict], ratings: list[int], raw_outputs: list[str]
) -> dict[str, Any]:
    """Process ratings and build result dict."""
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    n_invalid = 0

    for rating in ratings:
        if rating in counts:
            counts[rating] += 1
        else:
            logger.error(f"⚠️ Invalid rating: {rating}")
            n_invalid += 1

    n_total = len(ratings)
    n_valid = n_total - n_invalid
    percentages = {
        k: (v / n_valid * 100) if n_valid > 0 else 0.0 for k, v in counts.items()
    }

    # Attach ratings to triplets
    triplets_with_ratings = [
        {
            **triplet,
            "quality_rating": rating,
            "quality_label": QUALITY_SCALE.get(rating, {}).get("label", "unknown"),
            "quality_class": QUALITY_SCALE.get(rating, {}).get("class", "Unknown"),
            "quality_reasoning": raw_output,
        }
        for triplet, rating, raw_output in zip(
            triplets, ratings, raw_outputs, strict=False
        )
    ]

    # Log summary
    logger.info("Quality rating complete:")
    for level in [1, 2, 3, 4, 5]:
        logger.info(
            f"  {level} ({QUALITY_SCALE[level]['label']:10s}): "
            f"{counts[level]:4d} ({percentages[level]:5.1f}%)"
        )
    if n_invalid > 0:
        logger.warning(f"  Invalid ratings: {n_invalid} ({n_invalid/n_total*100:.1f}%)")

    return {
        "ratings": ratings,
        "counts": counts,
        "percentages": percentages,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "triplets_with_ratings": triplets_with_ratings,
    }


def _log_filter_stats(
    n_total: int, n_kept: int, n_filtered: int, condition: str
) -> None:
    """Log filtering statistics."""
    logger.info(f"Filtered by {condition}:")
    logger.info(f"  Kept: {n_kept}/{n_total} ({n_kept/n_total*100:.1f}%)")
    logger.info(f"  Filtered: {n_filtered}/{n_total} ({n_filtered/n_total*100:.1f}%)")


def _check_consistency(
    original_rating: int | None,
    swapped_rating: int | None,
    min_threshold: int = 3,
    max_threshold: int = 1,
) -> tuple[bool, str | None]:
    """Check if triplet passes strict consistency validation.

    Original must be ≥ min_threshold (at least Trivial quality)
    Swapped must be ≤ max_threshold (strictly Invalid - anchor NOT closer to negative than positive)

    Args:
        original_rating: Quality rating for (anchor, positive, negative), or None if parsing failed
        swapped_rating: Quality rating for (anchor, negative, positive) - swapped, or None if parsing failed
        min_threshold: Minimum rating for original to pass (default: 3)
        max_threshold: Maximum rating for swapped to pass (default: 1)

    Returns:
        Tuple of (passed, failure_reason) where failure_reason is None if passed
    """
    if original_rating is None:
        return False, "original_parse_failed"
    if swapped_rating is None:
        return False, "swapped_parse_failed"
    if original_rating < min_threshold:
        return False, f"original_too_low_{original_rating}"
    if swapped_rating > max_threshold:
        return False, f"swapped_too_high_{swapped_rating}"
    return True, None


def validate_triplet_consistency(
    *,
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    annotations: list[dict] | None = None,
    min_quality_threshold: int = 3,
    max_invalid_threshold: int = 1,
    lm_judge_preset: str = "lmjudge_quality_rating_gemini",
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Validate triplet quality consistency by rating swapped versions.

    For each triplet (anchor, positive, negative):
    - Original (a, b, c) already has quality_rating from prior validation
    - Evaluates swapped (a, c, b) to ensure it gets rating ≤ max_invalid_threshold (Invalid)

    This ensures the positive is actually closer to the anchor than the negative.

    Args:
        triplets: List of triplet dicts with anchor/positive/negative keys
                  AND quality_rating field from prior validation
        criterion: Similarity criterion name
        criterion_description: Optional detailed description
        annotations: Optional document annotations
        min_quality_threshold: Minimum rating for original to pass (default: 3)
        max_invalid_threshold: Maximum rating for swapped to pass (default: 1)
        lm_judge_preset: Quality rating preset to use
        cache_alias: Optional cache identifier
        run_name: Optional experiment/run name

    Returns:
        Dict with:
        - original_ratings: List of existing ratings (from triplet["quality_rating"])
        - swapped_ratings: List of ratings for swapped triplets (newly evaluated)
        - swapped_reasoning: List of reasoning for swapped ratings
        - consistency_passed: List of bools (True if both checks pass)
        - failure_reasons: List of failure reasons or None
        - n_total: Total triplets evaluated
        - n_passed: Number passing both checks
        - n_failed: Number failing either check
        - failure_breakdown: Dict with failure reasons
    """
    if not triplets:
        return {
            "original_ratings": [],
            "swapped_ratings": [],
            "swapped_reasoning": [],
            "consistency_passed": [],
            "failure_reasons": [],
            "n_total": 0,
            "n_passed": 0,
            "n_failed": 0,
            "failure_breakdown": {},
        }

    logger.info(f"Validating consistency for {len(triplets)} triplets...")

    # Extract existing quality ratings (already computed)
    original_ratings = [t["quality_rating"] for t in triplets]

    # Build inputs for SWAPPED triplets only (a, c, b)
    from multiview.benchmark.annotations.annotation_utils import (
        add_document_inputs_with_images,
    )

    similarity_criteria = (
        f"{criterion}: {criterion_description}" if criterion_description else criterion
    )

    n = len(triplets)
    inputs_swapped = {
        "similarity_criteria": [similarity_criteria] * n,
    }
    # Use shared function to handle document/image extraction properly
    add_document_inputs_with_images(
        inputs_swapped,
        {
            "a": [t["anchor"] for t in triplets],
            "b": [t["negative"] for t in triplets],  # SWAPPED
            "c": [t["positive"] for t in triplets],  # SWAPPED
        },
    )

    # Add annotations if available (with swapped roles)
    if annotations:
        add_annotation_summaries_to_inputs(
            inputs=inputs_swapped,
            triplets=triplets,
            annotations=annotations,
            triplet_keys_by_input_suffix={
                "a": "anchor",
                "b": "negative",  # SWAPPED
                "c": "positive",  # SWAPPED
            },
        )

    # Single inference call for N swapped evaluations
    swapped_ratings, swapped_reasoning = run_inference(
        inputs=inputs_swapped,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
        return_raw=True,
    )

    # Check consistency
    consistency_passed = []
    failure_reasons = []
    failure_breakdown = {}

    for orig_rating, swap_rating in zip(
        original_ratings, swapped_ratings, strict=False
    ):
        passed, reason = _check_consistency(
            orig_rating,
            swap_rating,
            min_quality_threshold,
            max_invalid_threshold,
        )
        consistency_passed.append(passed)
        failure_reasons.append(reason)

        if reason is not None:
            failure_breakdown[reason] = failure_breakdown.get(reason, 0) + 1

    n_passed = sum(consistency_passed)
    n_failed = len(consistency_passed) - n_passed

    # Log results
    logger.info("=" * 60)
    logger.info("CONSISTENCY VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total triplets: {n}")
    logger.info(f"Passed: {n_passed} ({n_passed/n*100:.1f}%)")
    logger.info(f"Failed: {n_failed} ({n_failed/n*100:.1f}%)")

    if failure_breakdown:
        logger.info("\nFailure breakdown:")
        for reason, count in sorted(failure_breakdown.items()):
            logger.info(f"  {reason}: {count}")
    logger.info("=" * 60)

    return {
        "original_ratings": original_ratings,
        "swapped_ratings": swapped_ratings,
        "swapped_reasoning": swapped_reasoning,
        "consistency_passed": consistency_passed,
        "failure_reasons": failure_reasons,
        "n_total": n,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "failure_breakdown": failure_breakdown,
    }


def rate_and_filter_quality_workflow(
    *,
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    annotations: list[dict] | None = None,
    min_quality: int | None = None,
    max_triplets: int | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
    quality_rating_preset: str | None = None,
    quality_rating_preset_with_annotations: str | None = None,
    validate_consistency: bool = True,
    consistency_min_quality: int = 3,
    consistency_max_invalid: int = 1,
) -> dict[str, Any]:
    """Rate triplets, filter by quality, validate consistency, optionally select top-N.

    Main entry point for quality rating/filtering. Automatically uses multiple
    presets if annotations available (takes minimum rating for conservative filtering).

    NEW: Consistency validation now runs by default BEFORE top-N selection.
    Ensures that (a, b, c) gets good score AND (a, c, b) gets rating 1 (Invalid).

    Workflow:
    1. Rate all triplets with LM judge
    2. Filter by min_quality threshold
    3. Validate consistency (NEW - enabled by default)
    4. Select top max_triplets from consistent ones

    Args:
        min_quality: Threshold (3=production, 4=challenging, None=no filtering)
        max_triplets: If set, select top-N by rating after filtering and consistency
        validate_consistency: Whether to run consistency validation (default: True)
        consistency_min_quality: Min rating for original triplet (default: 3)
        consistency_max_invalid: Max rating for swapped triplet (default: 1, strict)

    Returns:
        Dict with kept_triplets, dropped_triplets, ratings, reasoning, stats, consistency_stats
    """
    # Determine which presets to use
    has_annotations = _has_any_annotation_content(annotations)

    # Use custom presets if provided, otherwise fall back to defaults
    preset_without_annotations = (
        quality_rating_preset or "lmjudge_quality_rating_gemini"
    )
    preset_with_annotations = (
        quality_rating_preset_with_annotations
        or "lmjudge_quality_rating_with_annotation_gemini"
    )

    if has_annotations:
        # Conservative: rate with both presets and take minimum
        presets_config = [
            {
                "name": "without_annotations",
                "preset": preset_without_annotations,
                "use_annotations": False,
            },
            {
                "name": "with_annotations",
                "preset": preset_with_annotations,
                "use_annotations": True,
            },
        ]
    else:
        # Single preset: without annotations
        presets_config = [
            {
                "name": "without_annotations",
                "preset": preset_without_annotations,
                "use_annotations": False,
            },
        ]

    # Rate with each preset
    all_ratings = {}
    all_reasoning = {}
    all_results = {}

    for preset_cfg in presets_config:
        name = preset_cfg["name"]
        preset = preset_cfg["preset"]
        use_anns = preset_cfg["use_annotations"]

        cache_key = f"{cache_alias}_{name}" if cache_alias else None

        results = _rate_triplet_quality(
            triplets=triplets,
            criterion=criterion,
            criterion_description=criterion_description,
            lm_judge_preset=preset,
            cache_alias=cache_key,
            run_name=run_name,
            annotations=annotations if use_anns else None,
        )

        all_ratings[name] = results["ratings"]
        all_reasoning[name] = [
            t.get("quality_reasoning") for t in results["triplets_with_ratings"]
        ]
        all_results[name] = results

    # Compute minimum rating across all presets (conservative)
    if len(presets_config) > 1:
        min_ratings = [
            min(
                all_ratings[cfg["name"]][i]
                for cfg in presets_config
                if all_ratings[cfg["name"]][i] is not None
            )
            if any(all_ratings[cfg["name"]][i] is not None for cfg in presets_config)
            else None
            for i in range(len(triplets))
        ]
        # Use "with_annotations" as primary for reasoning and stats
        primary_name = "with_annotations"
        primary_ratings = min_ratings
        primary_reasoning = all_reasoning[primary_name]
        primary_stats = all_results[primary_name]
    else:
        # Single preset
        primary_name = presets_config[0]["name"]
        primary_ratings = all_ratings[primary_name]
        primary_reasoning = all_reasoning[primary_name]
        primary_stats = all_results[primary_name]
        min_ratings = primary_ratings

    # Attach min ratings to triplets
    triplets_with_ratings = [
        {**triplet, "quality_rating": rating, "quality_reasoning": reasoning}
        for triplet, rating, reasoning in zip(
            triplets, min_ratings, primary_reasoning, strict=False
        )
    ]

    # Build result dict
    result = {
        "ratings": primary_ratings,
        "reasoning": primary_reasoning,
        "stats": primary_stats,
    }

    # Add individual preset ratings if multiple
    if len(presets_config) > 1:
        for cfg in presets_config:
            name = cfg["name"]
            result[f"ratings_{name}"] = all_ratings[name]
            result[f"reasoning_{name}"] = all_reasoning[name]

    # Filter by minimum quality if requested
    if min_quality is not None:
        kept = [
            t
            for t in triplets_with_ratings
            if t.get("quality_rating", 0) >= min_quality
        ]
        dropped = [
            t for t in triplets_with_ratings if t.get("quality_rating", 0) < min_quality
        ]

        filter_mode = (
            "min_across_presets" if len(presets_config) > 1 else "single_preset"
        )
        _log_filter_stats(
            len(triplets),
            len(kept),
            len(dropped),
            f"quality >= {min_quality} ({filter_mode})",
        )

        result.update(
            {
                "kept_triplets": kept,
                "dropped_triplets": dropped,
                "stats": {
                    **result["stats"],
                    "n_total": len(triplets),
                    "n_kept": len(kept),
                    "n_filtered": len(dropped),
                    "min_quality": min_quality,
                    "filter_mode": filter_mode,
                },
            }
        )

        # Filter individual preset ratings to match kept triplets
        _filter_preset_ratings_to_match_triplets(result, triplets, kept)
    else:
        result["kept_triplets"] = triplets_with_ratings
        result["dropped_triplets"] = []

    # Step 3: ALWAYS run consistency validation on kept triplets (NEW)
    # NOTE: We only evaluate the SWAPPED versions - original ratings already exist
    if validate_consistency and result["kept_triplets"]:
        logger.info(
            f"Running consistency validation on {len(result['kept_triplets'])} triplets..."
        )
        consistency_result = validate_triplet_consistency(
            triplets=result["kept_triplets"],  # Already have quality_rating field
            criterion=criterion,
            criterion_description=criterion_description,
            annotations=annotations,
            min_quality_threshold=consistency_min_quality,
            max_invalid_threshold=consistency_max_invalid,
            lm_judge_preset=preset_without_annotations,
            cache_alias=f"{cache_alias}_consistency" if cache_alias else None,
            run_name=run_name,
        )

        # Filter by consistency
        kept_consistent = []
        dropped_inconsistent = []

        for triplet, passed, swap_rating, failure_reason in zip(
            result["kept_triplets"],
            consistency_result["consistency_passed"],
            consistency_result["swapped_ratings"],
            consistency_result["failure_reasons"],
            strict=False,
        ):
            # Add consistency metadata to triplet
            triplet["consistency_check"] = {
                "swapped_rating": swap_rating,
                "passed": passed,
                "failure_reason": failure_reason,
            }

            if passed:
                kept_consistent.append(triplet)
            else:
                dropped_inconsistent.append(triplet)

        _log_filter_stats(
            len(result["kept_triplets"]),
            len(kept_consistent),
            len(dropped_inconsistent),
            f"consistency check (min_orig={consistency_min_quality}, max_swap={consistency_max_invalid})",
        )

        # Save reference to source before overwriting
        source_before_consistency = result["kept_triplets"]

        result["kept_triplets"] = kept_consistent
        result["dropped_triplets"].extend(dropped_inconsistent)
        result["consistency_stats"] = consistency_result

        # Filter main ratings/reasoning to match consistent triplets
        # Build mapping from triplet ID to index before consistency filter
        triplet_to_idx = {
            (t["anchor_id"], t["positive_id"], t["negative_id"]): i
            for i, t in enumerate(source_before_consistency)
        }

        # Filter ratings and reasoning to match kept_consistent
        result["ratings"] = [
            result["ratings"][
                triplet_to_idx[(t["anchor_id"], t["positive_id"], t["negative_id"])]
            ]
            for t in kept_consistent
        ]
        result["reasoning"] = [
            result["reasoning"][
                triplet_to_idx[(t["anchor_id"], t["positive_id"], t["negative_id"])]
            ]
            for t in kept_consistent
        ]

        # Filter individual preset ratings to match consistent triplets
        if len(presets_config) > 1:
            _filter_preset_ratings_to_match_triplets(
                result,
                source_before_consistency,  # Source before consistency filter
                kept_consistent,
            )

    # Step 4: Select top-N by quality if requested (MOVED AFTER CONSISTENCY)
    if max_triplets is not None and len(result["kept_triplets"]) > max_triplets:
        kept_before_selection = result["kept_triplets"]
        n_before = len(kept_before_selection)

        # Sort by quality rating (descending) and take top max_triplets
        sorted_triplets = sorted(
            kept_before_selection,
            key=lambda t: t.get("quality_rating", 0),
            reverse=True,
        )
        selected = sorted_triplets[:max_triplets]
        not_selected = sorted_triplets[max_triplets:]

        logger.info(
            f"Selected top {max_triplets} triplets by quality rating "
            f"(had {n_before} after filtering)"
        )

        # Update result
        result["kept_triplets"] = selected
        result["dropped_triplets"] = result.get("dropped_triplets", []) + not_selected
        result["ratings"] = [t.get("quality_rating") for t in selected]
        result["reasoning"] = [t.get("quality_reasoning") for t in selected]
        result["stats"]["n_kept"] = len(selected)
        result["stats"]["n_selected"] = max_triplets
        result["stats"]["n_not_selected"] = n_before - max_triplets

        # Filter individual preset ratings to match selected triplets
        # Note: Use kept_before_selection as source since ratings were already filtered by quality
        _filter_preset_ratings_to_match_triplets(
            result, kept_before_selection, selected
        )

    return result
