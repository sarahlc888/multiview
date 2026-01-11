"""Validation utilities for synthetic document quality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from multiview.benchmark.triplets.utils import extract_active_tags, jaccard_similarity

logger = logging.getLogger(__name__)


def compute_tag_similarities(
    synth_annotation: dict,
    anchor_annotation: dict,
    tag_key: str = "tags",
) -> dict[str, Any]:
    """Compute Jaccard similarity for a specific tag type.

    Args:
        synth_annotation: Annotation dict for synthetic document
        anchor_annotation: Annotation dict for anchor document
        tag_key: "tags" or "spurious_tags"

    Returns:
        Dict with:
        - similarity: float (Jaccard similarity)
        - synth_tags: set of active tag names in synthetic
        - anchor_tags: set of active tag names in anchor
        - intersection: set of shared tags
        - union: set of all tags
        - num_synth_tags: number of active tags in synthetic
        - num_anchor_tags: number of active tags in anchor
        - num_intersection: number of shared tags
        - num_union: total number of unique tags
    """
    synth_tags = extract_active_tags(synth_annotation, tag_key)
    anchor_tags = extract_active_tags(anchor_annotation, tag_key)

    similarity = jaccard_similarity(synth_tags, anchor_tags)
    intersection = synth_tags.intersection(anchor_tags)
    union = synth_tags.union(anchor_tags)

    return {
        "similarity": similarity,
        "synth_tags": synth_tags,
        "anchor_tags": anchor_tags,
        "intersection": intersection,
        "union": union,
        "num_synth_tags": len(synth_tags),
        "num_anchor_tags": len(anchor_tags),
        "num_intersection": len(intersection),
        "num_union": len(union),
    }


def validate_synthesis(
    documents: list[Any],
    annotations: list[dict],
    synthesis_metadata: dict,
    output_dir: str | Path,
    task_name: str,
    triplets: list[tuple] | None = None,
    quality_ratings: list[int] | None = None,
) -> dict:
    """Validate synthetic document quality via tag similarity analysis.

    Args:
        documents: All documents (original + synthetic)
        annotations: All annotations (original + synthetic)
        synthesis_metadata: Metadata from synthesize_documents()
        output_dir: Directory to save validation reports
        task_name: Name of the task
        triplets: Optional list of triplets (anchor_idx, pos_idx, neg_idx)
        quality_ratings: Optional list of quality ratings (1-4) for each triplet

    Returns:
        Dict with validation statistics
    """
    logger.info(f"Validating synthetic annotations for task: {task_name}")

    num_original = synthesis_metadata["num_original_docs"]
    synthetic_docs_metadata = synthesis_metadata["synthetic_docs"]

    # Compute similarities for each synthetic document
    validation_records = []

    for synth_meta in synthetic_docs_metadata:
        # Map synthetic_idx to actual document index
        synth_doc_idx = num_original + synth_meta["synthetic_idx"]
        anchor_doc_idx = synth_meta["anchor_doc_idx"]

        synth_annotation = annotations[synth_doc_idx]
        anchor_annotation = annotations[anchor_doc_idx]

        # Compute similarities for tags and spurious_tags
        tags_sim = compute_tag_similarities(synth_annotation, anchor_annotation, "tags")
        spurious_sim = compute_tag_similarities(
            synth_annotation, anchor_annotation, "spurious_tags"
        )

        record = {
            "synthetic_doc_idx": synth_doc_idx,
            "anchor_doc_idx": anchor_doc_idx,
            "reference_doc_idx": synth_meta["reference_doc_idx"],
            "type": synth_meta["type"],
            "pair_id": synth_meta["pair_id"],
            # Tag similarities
            "tags_jaccard": tags_sim["similarity"],
            "tags_synth": sorted(tags_sim["synth_tags"]),
            "tags_anchor": sorted(tags_sim["anchor_tags"]),
            "tags_intersection": sorted(tags_sim["intersection"]),
            "tags_union": sorted(tags_sim["union"]),
            "tags_num_synth": tags_sim["num_synth_tags"],
            "tags_num_anchor": tags_sim["num_anchor_tags"],
            "tags_num_intersection": tags_sim["num_intersection"],
            "tags_num_union": tags_sim["num_union"],
            # Spurious tag similarities
            "spurious_jaccard": spurious_sim["similarity"],
            "spurious_synth": sorted(spurious_sim["synth_tags"]),
            "spurious_anchor": sorted(spurious_sim["anchor_tags"]),
            "spurious_intersection": sorted(spurious_sim["intersection"]),
            "spurious_union": sorted(spurious_sim["union"]),
            "spurious_num_synth": spurious_sim["num_synth_tags"],
            "spurious_num_anchor": spurious_sim["num_anchor_tags"],
            "spurious_num_intersection": spurious_sim["num_intersection"],
            "spurious_num_union": spurious_sim["num_union"],
        }

        validation_records.append(record)

    # Compute aggregate statistics
    stats = compute_validation_statistics(validation_records)

    # Compute triplet quality statistics if provided
    triplet_stats = None
    if triplets is not None and quality_ratings is not None:
        triplet_stats = analyze_triplet_quality_by_synthesis(
            triplets=triplets,
            quality_ratings=quality_ratings,
            num_original_docs=num_original,
        )
        stats["triplet_quality"] = triplet_stats

    # Write reports (artifact IO lives in artifacts.py)
    from multiview.benchmark.artifacts import save_synthesis_validation_reports

    outputs = save_synthesis_validation_reports(
        records=validation_records,
        stats=stats,
        output_dir=output_dir,
        task_name=task_name,
    )
    logger.info(
        "Validation complete. Reports saved to %s",
        outputs["json_path"].parent,
    )

    return stats


def analyze_triplet_quality_by_synthesis(
    triplets: list[tuple],
    quality_ratings: list[int],
    num_original_docs: int,
) -> dict:
    """Analyze triplet quality ratings by whether triplets involve synthetic documents.

    Args:
        triplets: List of (anchor_idx, pos_idx, neg_idx) tuples
        quality_ratings: List of quality ratings (1-4) for each triplet
        num_original_docs: Number of original (non-synthetic) documents

    Returns:
        Dict with quality rating distributions for triplets with/without synthetic docs
    """
    with_synthetic = []
    without_synthetic = []

    for triplet, rating in zip(triplets, quality_ratings, strict=False):
        if rating is None:
            # Skip triplets without ratings
            continue

        anchor_idx, pos_idx, neg_idx = triplet
        # Check if any document in the triplet is synthetic
        has_synthetic = (
            anchor_idx >= num_original_docs
            or pos_idx >= num_original_docs
            or neg_idx >= num_original_docs
        )

        if has_synthetic:
            with_synthetic.append(rating)
        else:
            without_synthetic.append(rating)

    def compute_rating_distribution(ratings: list[int]) -> dict:
        """Compute distribution of quality ratings."""
        if not ratings:
            return {
                "count": 0,
                "rating_1": 0,
                "rating_2": 0,
                "rating_3": 0,
                "rating_4": 0,
                "rating_1_pct": 0.0,
                "rating_2_pct": 0.0,
                "rating_3_pct": 0.0,
                "rating_4_pct": 0.0,
                "mean": 0.0,
                "median": 0.0,
            }

        total = len(ratings)
        rating_counts = {i: ratings.count(i) for i in range(1, 5)}

        return {
            "count": total,
            "rating_1": rating_counts[1],
            "rating_2": rating_counts[2],
            "rating_3": rating_counts[3],
            "rating_4": rating_counts[4],
            "rating_1_pct": float(rating_counts[1] / total * 100),
            "rating_2_pct": float(rating_counts[2] / total * 100),
            "rating_3_pct": float(rating_counts[3] / total * 100),
            "rating_4_pct": float(rating_counts[4] / total * 100),
            "mean": float(np.mean(ratings)),
            "median": float(np.median(ratings)),
        }

    return {
        "with_synthetic": compute_rating_distribution(with_synthetic),
        "without_synthetic": compute_rating_distribution(without_synthetic),
        "total_with_synthetic": len(with_synthetic),
        "total_without_synthetic": len(without_synthetic),
    }


def compute_validation_statistics(records: list[dict]) -> dict:
    """Compute aggregate statistics from validation records.

    Args:
        records: List of validation record dicts

    Returns:
        Dict with statistics broken down by document type
    """
    # Separate by type
    hard_pos = [r for r in records if r["type"] == "hard_positive"]
    hard_neg = [r for r in records if r["type"] == "hard_negative"]

    def compute_stats_for_group(group: list[dict], metric_prefix: str) -> dict:
        """Helper to compute stats for a group of records."""
        if not group:
            return {}

        tags_jaccards = [r["tags_jaccard"] for r in group]
        spurious_jaccards = [r["spurious_jaccard"] for r in group]

        return {
            f"{metric_prefix}_count": len(group),
            f"{metric_prefix}_tags_mean": float(np.mean(tags_jaccards)),
            f"{metric_prefix}_tags_median": float(np.median(tags_jaccards)),
            f"{metric_prefix}_tags_std": float(np.std(tags_jaccards)),
            f"{metric_prefix}_tags_min": float(np.min(tags_jaccards)),
            f"{metric_prefix}_tags_max": float(np.max(tags_jaccards)),
            f"{metric_prefix}_tags_p25": float(np.percentile(tags_jaccards, 25)),
            f"{metric_prefix}_tags_p75": float(np.percentile(tags_jaccards, 75)),
            f"{metric_prefix}_spurious_mean": float(np.mean(spurious_jaccards)),
            f"{metric_prefix}_spurious_median": float(np.median(spurious_jaccards)),
            f"{metric_prefix}_spurious_std": float(np.std(spurious_jaccards)),
            f"{metric_prefix}_spurious_min": float(np.min(spurious_jaccards)),
            f"{metric_prefix}_spurious_max": float(np.max(spurious_jaccards)),
            f"{metric_prefix}_spurious_p25": float(
                np.percentile(spurious_jaccards, 25)
            ),
            f"{metric_prefix}_spurious_p75": float(
                np.percentile(spurious_jaccards, 75)
            ),
        }

    stats = {
        "total_synthetic_docs": len(records),
        "num_hard_positives": len(hard_pos),
        "num_hard_negatives": len(hard_neg),
    }

    # Overall stats
    if records:
        stats.update(compute_stats_for_group(records, "overall"))

    # Hard positive stats
    if hard_pos:
        stats.update(compute_stats_for_group(hard_pos, "hard_pos"))

    # Hard negative stats
    if hard_neg:
        stats.update(compute_stats_for_group(hard_neg, "hard_neg"))

    return stats
