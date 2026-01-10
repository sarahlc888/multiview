"""Validation utilities for synthetic document quality."""

from __future__ import annotations

import json
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
) -> dict:
    """Validate synthetic document quality via tag similarity analysis.

    Args:
        documents: All documents (original + synthetic)
        annotations: All annotations (original + synthetic)
        synthesis_metadata: Metadata from synthesize_documents()
        output_dir: Directory to save validation reports
        task_name: Name of the task

    Returns:
        Dict with validation statistics
    """
    output_dir = Path(output_dir)
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

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
            "tags_synth": sorted(list(tags_sim["synth_tags"])),
            "tags_anchor": sorted(list(tags_sim["anchor_tags"])),
            "tags_intersection": sorted(list(tags_sim["intersection"])),
            "tags_num_synth": tags_sim["num_synth_tags"],
            "tags_num_anchor": tags_sim["num_anchor_tags"],
            "tags_num_intersection": tags_sim["num_intersection"],
            # Spurious tag similarities
            "spurious_jaccard": spurious_sim["similarity"],
            "spurious_synth": sorted(list(spurious_sim["synth_tags"])),
            "spurious_anchor": sorted(list(spurious_sim["anchor_tags"])),
            "spurious_intersection": sorted(list(spurious_sim["intersection"])),
            "spurious_num_synth": spurious_sim["num_synth_tags"],
            "spurious_num_anchor": spurious_sim["num_anchor_tags"],
            "spurious_num_intersection": spurious_sim["num_intersection"],
        }

        validation_records.append(record)

    # Compute aggregate statistics
    stats = compute_validation_statistics(validation_records)

    # Generate reports
    generate_validation_report_json(validation_records, stats, task_dir)
    generate_validation_report_markdown(validation_records, stats, task_dir, task_name)

    logger.info(f"Validation complete. Reports saved to {task_dir}")

    return stats


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
            f"{metric_prefix}_spurious_mean": float(np.mean(spurious_jaccards)),
            f"{metric_prefix}_spurious_median": float(np.median(spurious_jaccards)),
            f"{metric_prefix}_spurious_std": float(np.std(spurious_jaccards)),
            f"{metric_prefix}_spurious_min": float(np.min(spurious_jaccards)),
            f"{metric_prefix}_spurious_max": float(np.max(spurious_jaccards)),
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


def generate_validation_report_json(
    records: list[dict],
    stats: dict,
    output_dir: Path,
) -> None:
    """Generate JSON validation report.

    Args:
        records: List of validation record dicts
        stats: Aggregate statistics dict
        output_dir: Directory to save report
    """
    report = {
        "summary": stats,
        "details": records,
    }

    output_file = output_dir / "validation_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved JSON validation report to {output_file}")


def generate_validation_report_markdown(
    records: list[dict],
    stats: dict,
    output_dir: Path,
    task_name: str,
) -> None:
    """Generate Markdown validation report.

    Args:
        records: List of validation record dicts
        stats: Aggregate statistics dict
        output_dir: Directory to save report
        task_name: Name of the task
    """
    output_file = output_dir / "validation_report.md"

    with open(output_file, "w") as f:
        f.write(f"# Synthesis Validation Report: {task_name}\n\n")

        # Summary section
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Synthetic Documents**: {stats.get('total_synthetic_docs', 0)}\n")
        f.write(f"- **Hard Positives**: {stats.get('num_hard_positives', 0)}\n")
        f.write(f"- **Hard Negatives**: {stats.get('num_hard_negatives', 0)}\n\n")

        # Overall metrics table
        if stats.get("overall_count"):
            f.write("### Overall Metrics\n\n")
            f.write(
                "| Metric | Tags (Mean) | Tags (Median) | Tags (Std) | Spurious (Mean) | Spurious (Median) | Spurious (Std) |\n"
            )
            f.write(
                "|--------|-------------|---------------|------------|-----------------|-------------------|----------------|\n"
            )
            f.write(
                f"| Overall | {stats.get('overall_tags_mean', 0):.3f} | "
                f"{stats.get('overall_tags_median', 0):.3f} | "
                f"{stats.get('overall_tags_std', 0):.3f} | "
                f"{stats.get('overall_spurious_mean', 0):.3f} | "
                f"{stats.get('overall_spurious_median', 0):.3f} | "
                f"{stats.get('overall_spurious_std', 0):.3f} |\n\n"
            )

        # Hard positives section
        if stats.get("num_hard_positives", 0) > 0:
            f.write("### Hard Positives\n\n")
            f.write("*(Should preserve anchor's criterion tags, may differ on spurious)*\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Count | {stats.get('hard_pos_count', 0)} |\n")
            f.write(
                f"| Tags Jaccard (Mean ± Std) | {stats.get('hard_pos_tags_mean', 0):.3f} ± {stats.get('hard_pos_tags_std', 0):.3f} |\n"
            )
            f.write(
                f"| Tags Jaccard (Min - Max) | {stats.get('hard_pos_tags_min', 0):.3f} - {stats.get('hard_pos_tags_max', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Mean ± Std) | {stats.get('hard_pos_spurious_mean', 0):.3f} ± {stats.get('hard_pos_spurious_std', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Min - Max) | {stats.get('hard_pos_spurious_min', 0):.3f} - {stats.get('hard_pos_spurious_max', 0):.3f} |\n\n"
            )

        # Hard negatives section
        if stats.get("num_hard_negatives", 0) > 0:
            f.write("### Hard Negatives\n\n")
            f.write("*(Should differ from anchor's criterion tags, may preserve spurious)*\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Count | {stats.get('hard_neg_count', 0)} |\n")
            f.write(
                f"| Tags Jaccard (Mean ± Std) | {stats.get('hard_neg_tags_mean', 0):.3f} ± {stats.get('hard_neg_tags_std', 0):.3f} |\n"
            )
            f.write(
                f"| Tags Jaccard (Min - Max) | {stats.get('hard_neg_tags_min', 0):.3f} - {stats.get('hard_neg_tags_max', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Mean ± Std) | {stats.get('hard_neg_spurious_mean', 0):.3f} ± {stats.get('hard_neg_spurious_std', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Min - Max) | {stats.get('hard_neg_spurious_min', 0):.3f} - {stats.get('hard_neg_spurious_max', 0):.3f} |\n\n"
            )

        # Detailed results
        f.write("## Detailed Results\n\n")
        f.write("### Per-Document Comparisons\n\n")

        # Group by type
        hard_pos = [r for r in records if r["type"] == "hard_positive"]
        hard_neg = [r for r in records if r["type"] == "hard_negative"]

        if hard_pos:
            f.write("#### Hard Positives\n\n")
            f.write(
                "| Synth ID | Anchor ID | Tags Jaccard | Spurious Jaccard | Tags Overlap | Spurious Overlap |\n"
            )
            f.write(
                "|----------|-----------|--------------|------------------|--------------|------------------|\n"
            )
            for r in hard_pos:
                tags_overlap = f"{r['tags_num_intersection']}/{r['tags_num_union']}"
                spur_overlap = (
                    f"{r['spurious_num_intersection']}/{r['spurious_num_union']}"
                )
                f.write(
                    f"| {r['synthetic_doc_idx']} | {r['anchor_doc_idx']} | "
                    f"{r['tags_jaccard']:.3f} | {r['spurious_jaccard']:.3f} | "
                    f"{tags_overlap} | {spur_overlap} |\n"
                )
            f.write("\n")

        if hard_neg:
            f.write("#### Hard Negatives\n\n")
            f.write(
                "| Synth ID | Anchor ID | Tags Jaccard | Spurious Jaccard | Tags Overlap | Spurious Overlap |\n"
            )
            f.write(
                "|----------|-----------|--------------|------------------|--------------|------------------|\n"
            )
            for r in hard_neg:
                tags_overlap = f"{r['tags_num_intersection']}/{r['tags_num_union']}"
                spur_overlap = (
                    f"{r['spurious_num_intersection']}/{r['spurious_num_union']}"
                )
                f.write(
                    f"| {r['synthetic_doc_idx']} | {r['anchor_doc_idx']} | "
                    f"{r['tags_jaccard']:.3f} | {r['spurious_jaccard']:.3f} | "
                    f"{tags_overlap} | {spur_overlap} |\n"
                )
            f.write("\n")

        # Interpretation guide
        f.write("## Interpretation Guide\n\n")
        f.write("### Expected Patterns\n\n")
        f.write("**Hard Positives** (preserve criterion, borrow themes):\n")
        f.write(
            "- **Tags**: High similarity (0.7-1.0) - should preserve anchor's criterion-relevant tags\n"
        )
        f.write(
            "- **Spurious Tags**: Low-medium similarity (0.0-0.6) - themes/surface features from reference doc\n\n"
        )
        f.write("**Hard Negatives** (change criterion, borrow themes):\n")
        f.write("- **Tags**: Low similarity (0.0-0.4) - should differ from anchor's criterion\n")
        f.write(
            "- **Spurious Tags**: Variable similarity - may preserve some surface features from anchor\n\n"
        )
        f.write("### Quality Indicators\n\n")
        f.write("- **Good hard positives**: Tags Jaccard > 0.7\n")
        f.write("- **Good hard negatives**: Tags Jaccard < 0.3\n")
        f.write(
            "- **High spurious diversity**: Different spurious tags between synth and anchor\n"
        )

    logger.info(f"Saved Markdown validation report to {output_file}")
