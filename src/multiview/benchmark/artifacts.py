"""IO helpers for benchmark artifacts (documents / annotations / triplets).

These functions intentionally live outside `Task` to keep the Task abstraction lean:
Task owns in-memory state and orchestration; artifact IO is handled externally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


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
) -> Path:
    """Save triplets as JSONL to `{output_dir}/{task_name}/triplets.jsonl`."""
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "triplets.jsonl"

    quality_ratings = triplet_quality_ratings
    if quality_ratings is not None:
        # Local import to avoid a hard dependency chain (Task imports stay lean).
        from multiview.benchmark.triplets.quality_assurance import QUALITY_SCALE

    with open(output_file, "w") as f:
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

            f.write(json.dumps(payload) + "\n")

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
