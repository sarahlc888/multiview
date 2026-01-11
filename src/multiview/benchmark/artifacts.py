"""IO helpers for benchmark artifacts (documents / annotations / triplets).

These functions intentionally live outside `Task` to keep the Task abstraction lean:
Task owns in-memory state and orchestration; artifact IO is handled externally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class _TaskLike(Protocol):
    documents: list[str] | None
    document_annotations: list[dict] | None
    triplets: list[tuple[int, int, int]] | None
    triplet_quality_ratings: list[int] | None

    def get_task_name(self) -> str: ...


def _task_dir(output_dir: str | Path, task_name: str) -> Path:
    task_dir = Path(output_dir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def save_documents_jsonl(
    *, documents: list[str], output_dir: str | Path, task_name: str
) -> Path:
    """Save documents as JSONL to `{output_dir}/{task_name}/documents.jsonl`."""
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "documents.jsonl"
    with open(output_file, "w") as f:
        for i, doc in enumerate(documents):
            f.write(json.dumps({"doc_id": i, "document": doc}) + "\n")
    return output_file


def save_annotations_jsonl(
    *,
    documents: list[str],
    annotations: list[dict],
    output_dir: str | Path,
    task_name: str,
) -> Path:
    """Save document annotations as JSONL to `{output_dir}/{task_name}/annotations.jsonl`."""
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "annotations.jsonl"
    with open(output_file, "w") as f:
        for i, (doc, annotation) in enumerate(
            zip(documents, annotations, strict=False)
        ):
            f.write(json.dumps({"doc_id": i, "document": doc, **annotation}) + "\n")
    return output_file


def save_triplets_jsonl(
    *,
    documents: list[str],
    triplets: list[tuple[int, int, int]],
    output_dir: str | Path,
    task_name: str,
    triplet_quality_ratings: list[int] | None = None,
    document_annotations: list[dict] | None = None,
) -> Path:
    """Save triplets as JSON array to `{output_dir}/{task_name}/triplets.json`."""
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "triplets.json"

    quality_ratings = triplet_quality_ratings
    if quality_ratings is not None:
        # Local import to avoid a hard dependency chain (Task imports stay lean).
        from multiview.benchmark.triplets.quality_assurance import QUALITY_SCALE

    triplet_records = []
    for i, (anchor_id, positive_id, negative_id) in enumerate(triplets):
        payload: dict[str, Any] = {
            "triplet_id": i,
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "anchor": documents[anchor_id],
            "positive": documents[positive_id],
            "negative": documents[negative_id],
        }

        if quality_ratings is not None and i < len(quality_ratings):
            rating = quality_ratings[i]
            payload["quality_rating"] = rating
            payload["quality_label"] = QUALITY_SCALE.get(rating, {}).get(
                "label", "unknown"
            )
            payload["quality_class"] = QUALITY_SCALE.get(rating, {}).get(
                "class", "Unknown"
            )

        # Include annotations if available
        if document_annotations is not None:
            payload["anchor_annotation"] = document_annotations[anchor_id]
            payload["positive_annotation"] = document_annotations[positive_id]
            payload["negative_annotation"] = document_annotations[negative_id]

        triplet_records.append(payload)

    # Write as formatted JSON array for easy browsing
    with open(output_file, "w") as f:
        json.dump(triplet_records, f, indent=2)

    return output_file


def save_task_documents(task: _TaskLike, output_dir: str | Path) -> Path:
    """Save a Task's documents to JSONL under `{output_dir}/{task_name}/documents.jsonl`."""
    if task.documents is None:
        raise RuntimeError("Must call load_documents() before saving documents")
    return save_documents_jsonl(
        documents=task.documents, output_dir=output_dir, task_name=task.get_task_name()
    )


def save_task_annotations(task: _TaskLike, output_dir: str | Path) -> Path:
    """Save a Task's annotations to JSONL under `{output_dir}/{task_name}/annotations.jsonl`."""
    if task.documents is None:
        raise RuntimeError("Must call load_documents() before saving annotations")
    if task.document_annotations is None:
        raise RuntimeError("Must call annotate_documents() before saving annotations")
    return save_annotations_jsonl(
        documents=task.documents,
        annotations=task.document_annotations,
        output_dir=output_dir,
        task_name=task.get_task_name(),
    )


def save_task_triplets(task: _TaskLike, output_dir: str | Path) -> Path:
    """Save a Task's triplets to JSONL under `{output_dir}/{task_name}/triplets.jsonl`."""
    if task.documents is None:
        raise RuntimeError("Must call load_documents() before saving triplets")
    if task.triplets is None:
        raise RuntimeError("Must call create_triplets() before saving triplets")
    return save_triplets_jsonl(
        documents=task.documents,
        triplets=task.triplets,
        output_dir=output_dir,
        task_name=task.get_task_name(),
        triplet_quality_ratings=task.triplet_quality_ratings,
        document_annotations=task.document_annotations,
    )


def save_method_triplet_logs_jsonl(
    *,
    triplet_logs: list[dict[str, Any]],
    output_dir: str | Path,
    task_name: str,
    method_name: str,
) -> Path:
    """Save per-triplet method logs to `{output_dir}/{task_name}/{method_name}.jsonl`."""
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / f"{method_name}.jsonl"
    with open(output_file, "w") as f:
        for record in triplet_logs:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_file


def save_synthesis_validation_reports(
    *,
    records: list[dict],
    stats: dict,
    output_dir: str | Path,
    task_name: str,
) -> dict[str, Path]:
    """Write synthesis validation outputs under `{output_dir}/{task_name}/...`."""
    task_dir = _task_dir(output_dir, task_name)
    json_path = _write_validation_report_json(
        records=records, stats=stats, output_dir=task_dir
    )
    md_path = _write_validation_report_markdown(
        records=records, stats=stats, output_dir=task_dir, task_name=task_name
    )
    return {"json_path": json_path, "markdown_path": md_path}


def _write_validation_report_json(
    *, records: list[dict], stats: dict, output_dir: Path
) -> Path:
    """Write JSON validation report."""
    report = {"summary": stats, "details": records}
    output_file = output_dir / "validation_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved JSON validation report to %s", output_file)
    return output_file


def _write_validation_report_markdown(
    *, records: list[dict], stats: dict, output_dir: Path, task_name: str
) -> Path:
    """Write Markdown validation report."""
    output_file = output_dir / "validation_report.md"

    with open(output_file, "w") as f:
        f.write(f"# Synthesis Validation Report: {task_name}\n\n")

        # Summary section
        f.write("## Summary Statistics\n\n")
        f.write(
            f"- **Total Synthetic Documents**: {stats.get('total_synthetic_docs', 0)}\n"
        )
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
            f.write(
                "*(Should preserve anchor's criterion tags, may differ on spurious)*\n\n"
            )
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
            f.write(
                "*(Should differ from anchor's criterion tags, may preserve spurious)*\n\n"
            )
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
                tags_union = r.get("tags_num_union", len(r.get("tags_union", [])))
                spur_union = r.get(
                    "spurious_num_union", len(r.get("spurious_union", []))
                )
                tags_overlap = f"{r.get('tags_num_intersection', 0)}/{tags_union}"
                spur_overlap = f"{r.get('spurious_num_intersection', 0)}/{spur_union}"
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
                tags_union = r.get("tags_num_union", len(r.get("tags_union", [])))
                spur_union = r.get(
                    "spurious_num_union", len(r.get("spurious_union", []))
                )
                tags_overlap = f"{r.get('tags_num_intersection', 0)}/{tags_union}"
                spur_overlap = f"{r.get('spurious_num_intersection', 0)}/{spur_union}"
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
        f.write(
            "- **Tags**: Low similarity (0.0-0.4) - should differ from anchor's criterion\n"
        )
        f.write(
            "- **Spurious Tags**: Variable similarity - may preserve some surface features from anchor\n\n"
        )
        f.write("### Quality Indicators\n\n")
        f.write("- **Good hard positives**: Tags Jaccard > 0.7\n")
        f.write("- **Good hard negatives**: Tags Jaccard < 0.3\n")
        f.write(
            "- **High spurious diversity**: Different spurious tags between synth and anchor\n"
        )

    logger.info("Saved Markdown validation report to %s", output_file)
    return output_file
