"""IO helpers for benchmark artifacts (documents / annotations / triplets).

These functions intentionally live outside `Task` to keep the Task abstraction lean:
Task owns in-memory state and orchestration; artifact IO is handled externally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class _TaskLike(Protocol):
    config: dict | Any
    documents: list[str] | None
    document_annotations: list[dict] | None
    triplets: list[tuple[int, int, int]] | None
    triplet_quality_ratings: list[int] | None
    triplet_quality_ratings_with_annotations: list[int] | None
    triplet_quality_ratings_without_annotations: list[int] | None
    triplet_quality_reasoning: list[str] | None
    triplet_quality_reasoning_with_annotations: list[str] | None
    triplet_quality_reasoning_without_annotations: list[str] | None

    def get_task_name(self) -> str: ...


def _task_dir(output_dir: str | Path, task_name: str) -> Path:
    task_dir = Path(output_dir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def _clean_document_for_json(doc: Any) -> Any:
    """Clean a document for JSON serialization.

    Removes internal metadata fields (starting with _) and converts sets to lists.
    Handles nested structures recursively.
    """
    if isinstance(doc, dict):
        clean_doc = {}
        for k, v in doc.items():
            if not k.startswith("_"):
                clean_doc[k] = _clean_document_for_json(v)
        return clean_doc
    elif isinstance(doc, set):
        return [_clean_document_for_json(item) for item in doc]
    elif isinstance(doc, list):
        return [_clean_document_for_json(item) for item in doc]
    else:
        return doc


def save_documents_jsonl(
    *, documents: list[str], output_dir: str | Path, task_name: str
) -> Path:
    """Save documents as JSONL to `{output_dir}/{task_name}/documents.jsonl`."""
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "documents.jsonl"
    with open(output_file, "w") as f:
        for i, doc in enumerate(documents):
            clean_doc = _clean_document_for_json(doc)
            f.write(json.dumps({"doc_id": i, "document": clean_doc}) + "\n")
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


def save_triplets_json(
    *,
    documents: list[str],
    triplets: list[tuple[int, int, int]],
    output_dir: str | Path,
    task_name: str,
    triplet_quality_ratings: list[int] | None = None,
    triplet_quality_ratings_with_annotations: list[int] | None = None,
    triplet_quality_ratings_without_annotations: list[int] | None = None,
    triplet_quality_reasoning: list[str] | None = None,
    triplet_quality_reasoning_with_annotations: list[str] | None = None,
    triplet_quality_reasoning_without_annotations: list[str] | None = None,
    document_annotations: list[dict] | None = None,
) -> Path:
    """Save triplets as JSON array to `{output_dir}/{task_name}/triplets.json`."""
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "triplets.json"

    quality_ratings = triplet_quality_ratings
    if (
        quality_ratings is not None
        or triplet_quality_ratings_with_annotations is not None
        or triplet_quality_ratings_without_annotations is not None
    ):
        # Local import to avoid a hard dependency chain (Task imports stay lean).
        from multiview.benchmark.triplets.quality_assurance import QUALITY_SCALE

    def add_quality_fields(
        payload: dict[str, Any],
        rating: int | None,
        suffix: str,
        reasoning: str | None = None,
    ) -> None:
        if rating is None:
            return

        quality_info = {
            "rating": rating,
            "label": QUALITY_SCALE.get(rating, {}).get("label", "unknown"),
            "class": QUALITY_SCALE.get(rating, {}).get("class", "Unknown"),
        }

        if reasoning is not None:
            quality_info["reasoning"] = reasoning

        # Use "quality_assessment" as the key for the nested dict
        key = f"quality_assessment{suffix}" if suffix else "quality_assessment"
        payload[key] = quality_info

    triplet_records = []
    for i, triplet in enumerate(triplets):
        # Handle both tuple format (anchor_id, positive_id, negative_id)
        # and dict format with additional metadata
        if isinstance(triplet, dict):
            anchor_id = triplet["anchor_id"]
            positive_id = triplet["positive_id"]
            negative_id = triplet["negative_id"]
            triplet_metadata = triplet  # Preserve for consistency_check etc
        else:
            anchor_id, positive_id, negative_id = triplet
            triplet_metadata = None

        payload: dict[str, Any] = {
            "triplet_id": i,
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "anchor": _clean_document_for_json(documents[anchor_id]),
            "positive": _clean_document_for_json(documents[positive_id]),
            "negative": _clean_document_for_json(documents[negative_id]),
        }

        # Include consistency check metadata if available
        if triplet_metadata and "consistency_check" in triplet_metadata:
            payload["consistency_check"] = triplet_metadata["consistency_check"]

        # Only save unsuffixed quality_assessment if it's NOT a duplicate of with_annotations
        # (i.e., only save if we're not doing a with/without comparison)
        if (
            quality_ratings is not None
            and i < len(quality_ratings)
            and triplet_quality_ratings_with_annotations is None
            and triplet_quality_ratings_without_annotations is None
        ):
            reasoning = (
                triplet_quality_reasoning[i]
                if triplet_quality_reasoning and i < len(triplet_quality_reasoning)
                else None
            )
            add_quality_fields(payload, quality_ratings[i], "", reasoning)
        if triplet_quality_ratings_with_annotations is not None and i < len(
            triplet_quality_ratings_with_annotations
        ):
            reasoning_with = (
                triplet_quality_reasoning_with_annotations[i]
                if triplet_quality_reasoning_with_annotations
                and i < len(triplet_quality_reasoning_with_annotations)
                else None
            )
            add_quality_fields(
                payload,
                triplet_quality_ratings_with_annotations[i],
                "_with_annotations",
                reasoning_with,
            )
        if triplet_quality_ratings_without_annotations is not None and i < len(
            triplet_quality_ratings_without_annotations
        ):
            reasoning_without = (
                triplet_quality_reasoning_without_annotations[i]
                if triplet_quality_reasoning_without_annotations
                and i < len(triplet_quality_reasoning_without_annotations)
                else None
            )
            add_quality_fields(
                payload,
                triplet_quality_ratings_without_annotations[i],
                "_without_annotations",
                reasoning_without,
            )

        # Include annotations if available
        if document_annotations is not None:
            # Remove category_schema from annotations (will be saved separately)
            anchor_annot = document_annotations[anchor_id].copy()
            positive_annot = document_annotations[positive_id].copy()
            negative_annot = document_annotations[negative_id].copy()

            anchor_annot.pop("category_schema", None)
            positive_annot.pop("category_schema", None)
            negative_annot.pop("category_schema", None)

            payload["anchor_annotation"] = anchor_annot
            payload["positive_annotation"] = positive_annot
            payload["negative_annotation"] = negative_annot

        triplet_records.append(payload)

    # Save category_schema and tag_schema separately if they exist in annotations
    if document_annotations is not None and len(document_annotations) > 0:
        # Extract category_schema from first non-null annotation
        category_schema = None
        tag_schema = None
        for annot in document_annotations:
            if annot:
                if "category_schema" in annot and category_schema is None:
                    category_schema = annot["category_schema"]
                if "tag_schema" in annot and tag_schema is None:
                    tag_schema = annot["tag_schema"]
                if category_schema is not None and tag_schema is not None:
                    break

        if category_schema is not None:
            schema_file = task_dir / "category_schema.json"
            with open(schema_file, "w") as f:
                json.dump(category_schema, f, indent=2)
            logger.info(f"Saved category schema to {schema_file}")

        if tag_schema is not None:
            schema_file = task_dir / "tag_schema.json"
            with open(schema_file, "w") as f:
                json.dump(tag_schema, f, indent=2)
            logger.info(f"Saved tag schema to {schema_file}")

    # Write as formatted JSON array for easy browsing
    with open(output_file, "w") as f:
        json.dump(triplet_records, f, indent=2)
    logger.info(f"Saved {len(triplet_records)} triplets to {output_file}")

    # Automatically generate HTML viewer for the triplets
    try:
        from multiview.visualization.triplet_viewer import generate_triplet_viewer

        viewer_path = generate_triplet_viewer(output_file)
        logger.info(f"Generated triplet viewer: {viewer_path}")
    except Exception as e:
        logger.warning(f"Failed to generate triplet viewer: {e}")

    return output_file


def save_dropped_triplets_json(
    *,
    documents: list[str],
    dropped_triplets: list[tuple[int, int, int]],
    output_dir: str | Path,
    task_name: str,
    triplet_quality_ratings: list[int] | None = None,
    triplet_quality_ratings_with_annotations: list[int] | None = None,
    triplet_quality_ratings_without_annotations: list[int] | None = None,
    triplet_quality_reasoning: list[str] | None = None,
    triplet_quality_reasoning_with_annotations: list[str] | None = None,
    triplet_quality_reasoning_without_annotations: list[str] | None = None,
    document_annotations: list[dict] | None = None,
    drop_reason: str | None = None,
) -> Path:
    """Save dropped triplets as JSON array to `{output_dir}/{task_name}/dropped_triplets.json`.

    Args:
        documents: List of document strings
        dropped_triplets: List of (anchor_id, positive_id, negative_id) tuples that were filtered out
        output_dir: Output directory path
        task_name: Name of the task
        triplet_quality_ratings: Optional quality ratings for dropped triplets
        triplet_quality_ratings_with_annotations: Optional quality ratings with annotations
        triplet_quality_ratings_without_annotations: Optional quality ratings without annotations
        document_annotations: Optional document annotations
        drop_reason: Optional reason why triplets were dropped (e.g., "quality < 3")

    Returns:
        Path to the saved file
    """
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "dropped_triplets.json"

    quality_ratings = triplet_quality_ratings
    if (
        quality_ratings is not None
        or triplet_quality_ratings_with_annotations is not None
        or triplet_quality_ratings_without_annotations is not None
    ):
        # Local import to avoid a hard dependency chain (Task imports stay lean).
        from multiview.benchmark.triplets.quality_assurance import QUALITY_SCALE

    def add_quality_fields(
        payload: dict[str, Any],
        rating: int | None,
        suffix: str,
        reasoning: str | None = None,
    ) -> None:
        if rating is None:
            return

        quality_info = {
            "rating": rating,
            "label": QUALITY_SCALE.get(rating, {}).get("label", "unknown"),
            "class": QUALITY_SCALE.get(rating, {}).get("class", "Unknown"),
        }

        if reasoning is not None:
            quality_info["reasoning"] = reasoning

        # Use "quality_assessment" as the key for the nested dict
        key = f"quality_assessment{suffix}" if suffix else "quality_assessment"
        payload[key] = quality_info

    triplet_records = []
    for i, triplet in enumerate(dropped_triplets):
        # Handle both tuple format (anchor_id, positive_id, negative_id)
        # and dict format with additional metadata
        if isinstance(triplet, dict):
            anchor_id = triplet["anchor_id"]
            positive_id = triplet["positive_id"]
            negative_id = triplet["negative_id"]
            triplet_metadata = triplet  # Preserve for consistency_check etc
        else:
            anchor_id, positive_id, negative_id = triplet
            triplet_metadata = None

        payload: dict[str, Any] = {
            "triplet_id": i,
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "anchor": documents[anchor_id],
            "positive": documents[positive_id],
            "negative": documents[negative_id],
        }

        if drop_reason is not None:
            payload["drop_reason"] = drop_reason

        # Include consistency check metadata if available
        if triplet_metadata and "consistency_check" in triplet_metadata:
            payload["consistency_check"] = triplet_metadata["consistency_check"]

        # Only save unsuffixed quality_assessment if it's NOT a duplicate of with_annotations
        # (i.e., only save if we're not doing a with/without comparison)
        if (
            quality_ratings is not None
            and i < len(quality_ratings)
            and triplet_quality_ratings_with_annotations is None
            and triplet_quality_ratings_without_annotations is None
        ):
            reasoning = (
                triplet_quality_reasoning[i]
                if triplet_quality_reasoning and i < len(triplet_quality_reasoning)
                else None
            )
            add_quality_fields(payload, quality_ratings[i], "", reasoning)
        if triplet_quality_ratings_with_annotations is not None and i < len(
            triplet_quality_ratings_with_annotations
        ):
            reasoning_with = (
                triplet_quality_reasoning_with_annotations[i]
                if triplet_quality_reasoning_with_annotations
                and i < len(triplet_quality_reasoning_with_annotations)
                else None
            )
            add_quality_fields(
                payload,
                triplet_quality_ratings_with_annotations[i],
                "_with_annotations",
                reasoning_with,
            )
        if triplet_quality_ratings_without_annotations is not None and i < len(
            triplet_quality_ratings_without_annotations
        ):
            reasoning_without = (
                triplet_quality_reasoning_without_annotations[i]
                if triplet_quality_reasoning_without_annotations
                and i < len(triplet_quality_reasoning_without_annotations)
                else None
            )
            add_quality_fields(
                payload,
                triplet_quality_ratings_without_annotations[i],
                "_without_annotations",
                reasoning_without,
            )

        # Include annotations if available
        if document_annotations is not None:
            # Remove category_schema from annotations (will be saved separately)
            anchor_annot = document_annotations[anchor_id].copy()
            positive_annot = document_annotations[positive_id].copy()
            negative_annot = document_annotations[negative_id].copy()

            anchor_annot.pop("category_schema", None)
            positive_annot.pop("category_schema", None)
            negative_annot.pop("category_schema", None)

            payload["anchor_annotation"] = anchor_annot
            payload["positive_annotation"] = positive_annot
            payload["negative_annotation"] = negative_annot

        triplet_records.append(payload)

    # Save category_schema and tag_schema separately if they exist in annotations
    # (same schema file as main triplets)
    if document_annotations is not None and len(document_annotations) > 0:
        # Extract schemas from first non-null annotation
        category_schema = None
        tag_schema = None
        for annot in document_annotations:
            if annot:
                if "category_schema" in annot and category_schema is None:
                    category_schema = annot["category_schema"]
                if "tag_schema" in annot and tag_schema is None:
                    tag_schema = annot["tag_schema"]
                if category_schema is not None and tag_schema is not None:
                    break

        if category_schema is not None:
            schema_file = task_dir / "category_schema.json"
            # Only write if it doesn't exist (avoid overwriting from main triplets)
            if not schema_file.exists():
                with open(schema_file, "w") as f:
                    json.dump(category_schema, f, indent=2)

        if tag_schema is not None:
            schema_file = task_dir / "tag_schema.json"
            # Only write if it doesn't exist (avoid overwriting from main triplets)
            if not schema_file.exists():
                with open(schema_file, "w") as f:
                    json.dump(tag_schema, f, indent=2)

    # Write as formatted JSON array for easy browsing
    with open(output_file, "w") as f:
        json.dump(triplet_records, f, indent=2)

    # Automatically generate HTML viewer for dropped triplets
    try:
        from multiview.visualization.triplet_viewer import generate_triplet_viewer

        viewer_path = generate_triplet_viewer(
            output_file,
            output_html_path=task_dir / "dropped_triplets_viewer.html",
        )
        logger.info(f"Generated dropped triplets viewer: {viewer_path}")
    except Exception as e:
        logger.warning(f"Failed to generate dropped triplets viewer: {e}")

    return output_file


def save_dropped_triplets_from_quality_result(
    *,
    documents: list[str],
    quality_result: dict,
    output_dir: str | Path,
    task_name: str,
    min_quality: int,
    document_annotations: list[dict] | None = None,
) -> Path:
    """Save dropped triplets from quality rating/filtering result.

    This is a convenience wrapper around save_dropped_triplets_json that handles
    extracting the ratings/reasoning from the quality workflow result dict.

    Args:
        documents: List of document strings
        quality_result: Result dict from rate_and_filter_quality_workflow
        output_dir: Output directory path
        task_name: Name of the task
        min_quality: Minimum quality threshold used for filtering
        document_annotations: Optional document annotations

    Returns:
        Path to the saved file
    """
    dropped_triplets_dicts = quality_result["dropped_triplets"]
    if not dropped_triplets_dicts:
        raise ValueError("No dropped triplets to save")

    # Extract triplet IDs
    dropped_triplets = [
        (t["anchor_id"], t["positive_id"], t["negative_id"])
        for t in dropped_triplets_dicts
    ]

    # Determine drop reason and prepare kwargs
    is_dual_filter = "ratings_without_annotations" in quality_result
    drop_reason = (
        f"Dual-filter: BOTH ratings must be >= {min_quality}"
        if is_dual_filter
        else f"quality < {min_quality}"
    )

    kwargs = {
        "documents": documents,
        "dropped_triplets": dropped_triplets,
        "output_dir": output_dir,
        "task_name": task_name,
        "document_annotations": document_annotations,
        "drop_reason": drop_reason,
    }

    # Add ratings/reasoning based on filter type
    if is_dual_filter:
        # Extract ratings for dropped triplets from the full rating lists
        dropped_ids = {
            (t["anchor_id"], t["positive_id"], t["negative_id"])
            for t in dropped_triplets_dicts
        }
        all_triplets = quality_result["kept_triplets"] + dropped_triplets_dicts
        all_ids = [
            (t["anchor_id"], t["positive_id"], t["negative_id"]) for t in all_triplets
        ]

        kwargs.update(
            {
                "triplet_quality_ratings_with_annotations": [
                    t.get("quality_rating") for t in dropped_triplets_dicts
                ],
                "triplet_quality_ratings_without_annotations": [
                    v
                    for v, tid in zip(
                        quality_result["ratings_without_annotations"],
                        all_ids,
                        strict=False,
                    )
                    if tid in dropped_ids
                ],
                "triplet_quality_reasoning_with_annotations": [
                    t.get("quality_reasoning") for t in dropped_triplets_dicts
                ],
                "triplet_quality_reasoning_without_annotations": [
                    v
                    for v, tid in zip(
                        quality_result["reasoning_without_annotations"],
                        all_ids,
                        strict=False,
                    )
                    if tid in dropped_ids
                ],
            }
        )
    else:
        # Single filter mode
        kwargs.update(
            {
                "triplet_quality_ratings": [
                    t.get("quality_rating") for t in dropped_triplets_dicts
                ],
                "triplet_quality_reasoning": [
                    t.get("quality_reasoning") for t in dropped_triplets_dicts
                ],
            }
        )

    return save_dropped_triplets_json(**kwargs)


def load_documents_from_jsonl(
    *,
    output_dir: str | Path,
    task_name: str,
) -> list[str]:
    """Load documents from saved JSONL file.

    Returns:
        List of document strings (includes both real and synthetic documents
        if they were saved together)
    """
    task_dir = Path(output_dir) / task_name
    documents_file = task_dir / "documents.jsonl"

    if not documents_file.exists():
        raise FileNotFoundError(f"Documents file not found: {documents_file}")

    documents = []
    with open(documents_file) as f:
        for line in f:
            record = json.loads(line)
            documents.append(record["document"])

    logger.debug(f"Loaded {len(documents)} documents from {documents_file}")
    return documents


def load_schema_from_json(
    *,
    output_dir: str | Path,
    task_name: str,
) -> dict | None:
    """Load category/tag schema from saved JSON file.

    Returns:
        Schema dict (category_schema or tag_schema), or None if not found
    """
    task_dir = Path(output_dir) / task_name

    # Try category_schema first
    category_schema_file = task_dir / "category_schema.json"
    if category_schema_file.exists():
        with open(category_schema_file) as f:
            schema = json.load(f)
        logger.info(f"Loaded category schema from {category_schema_file}")
        return {"category_schema": schema}

    # Try tag_schema
    tag_schema_file = task_dir / "tag_schema.json"
    if tag_schema_file.exists():
        with open(tag_schema_file) as f:
            schema = json.load(f)
        logger.info(f"Loaded tag schema from {tag_schema_file}")
        return {"tag_schema": schema}

    logger.debug(f"No schema files found in {task_dir}")
    return None


def load_triplets_from_json(
    *,
    output_dir: str | Path,
    task_name: str,
) -> tuple[list[tuple[int, int, int]], list[int] | None]:
    """Load triplets from saved JSON file.

    Returns:
        Tuple of (triplets, quality_ratings) where:
        - triplets: list of (anchor_id, positive_id, negative_id) tuples
        - quality_ratings: list of quality ratings (or None if not available)
    """
    task_dir = Path(output_dir) / task_name
    triplets_file = task_dir / "triplets.json"

    if not triplets_file.exists():
        raise FileNotFoundError(f"Triplets file not found: {triplets_file}")

    with open(triplets_file) as f:
        triplet_records = json.load(f)

    # Extract triplet tuples
    triplets = [
        (r["anchor_id"], r["positive_id"], r["negative_id"]) for r in triplet_records
    ]

    # Extract quality ratings if available
    quality_ratings = None
    if triplet_records and "quality_rating" in triplet_records[0]:
        quality_ratings = [r["quality_rating"] for r in triplet_records]

    logger.info(f"Loaded {len(triplets)} triplets from {triplets_file}")

    # Generate viewer if it doesn't exist
    viewer_path = triplets_file.parent / "viewer.html"
    if not viewer_path.exists():
        try:
            from multiview.visualization.triplet_viewer import generate_triplet_viewer

            generate_triplet_viewer(triplets_file)
            logger.info(f"Generated viewer: {viewer_path}")
        except Exception as e:
            logger.warning(f"Failed to generate viewer: {e}")

    return triplets, quality_ratings


def can_use_cached_triplets(
    *,
    output_dir: str | Path,
    task_name: str,
    current_config: dict,
) -> bool:
    """Check if cached triplets exist and match config (lightweight check).

    This is a fast check that doesn't load the actual triplets - just validates
    that the cache exists and the config matches. Use this before expensive
    operations like annotation to decide if you need to generate new triplets.

    Args:
        output_dir: Directory where cached triplets might be saved
        task_name: Name of the task (e.g., "gsm8k__arithmetic")
        current_config: Current task configuration dict

    Returns:
        True if cached triplets are available and config matches, False otherwise
    """
    try:
        # Check if triplet file exists
        triplets_file = Path(output_dir) / task_name / "triplets.json"
        if not triplets_file.exists():
            return False

        # Check if cached config exists
        cached_config = load_triplet_config(output_dir=output_dir, task_name=task_name)
        if cached_config is None:
            return False

        # Check if configs match
        return triplet_config_matches(cached_config, current_config)

    except Exception:
        return False


def try_load_cached_triplets(
    *,
    output_dir: str | Path,
    task_name: str,
    current_config: dict,
) -> tuple[list[tuple[int, int, int]], list[int] | None] | None:
    """Try to load cached triplets if they match the current config.

    This is a utility function that handles the full cache validation workflow:
    1. Check if cached config exists
    2. Validate config matches
    3. Load triplets if match

    Args:
        output_dir: Directory where cached triplets might be saved
        task_name: Name of the task (e.g., "gsm8k__arithmetic")
        current_config: Current task configuration dict

    Returns:
        Tuple of (triplets, quality_ratings) if cache is valid, None otherwise
    """
    cache_path = Path(output_dir) / task_name
    logger.info(f"Checking for cached triplets in: {cache_path}")

    try:
        # Check if cached config exists
        cached_config = load_triplet_config(output_dir=output_dir, task_name=task_name)

        if cached_config is None:
            logger.info("  ✗ No cached triplet_config.json found")
            return None

        logger.info("  ✓ Found triplet_config.json")

        # Check if configs match
        current_extracted = _extract_triplet_generation_config(current_config)
        if not triplet_config_matches(cached_config, current_config):
            logger.info("  ✗ Config mismatch - cached triplets cannot be reused")

            # Show what differs
            all_keys = set(cached_config.keys()) | set(current_extracted.keys())
            diffs = []
            for key in sorted(all_keys):
                cached_val = cached_config.get(key)
                current_val = current_extracted.get(key)
                if cached_val != current_val:
                    diffs.append(f"    - {key}: {cached_val} → {current_val}")

            if diffs:
                logger.info("  Config differences:")
                for diff in diffs[:5]:  # Show first 5 differences
                    logger.info(diff)
                if len(diffs) > 5:
                    logger.info(f"    ... and {len(diffs) - 5} more")
            return None

        logger.info("  ✓ Config matches - cached triplets can be reused!")

        # Load the cached triplets
        triplets, quality_ratings = load_triplets_from_json(
            output_dir=output_dir, task_name=task_name
        )

        logger.info(f"  ✓ Successfully loaded {len(triplets)} cached triplets")
        if quality_ratings:
            logger.info(f"     (with {len(quality_ratings)} quality ratings)")

        # Generate viewer if it doesn't exist
        triplets_file = Path(output_dir) / task_name / "triplets.json"
        viewer_path = triplets_file.parent / "viewer.html"
        if not viewer_path.exists():
            try:
                from multiview.visualization.triplet_viewer import (
                    generate_triplet_viewer,
                )

                generate_triplet_viewer(triplets_file)
                logger.info(f"  ✓ Generated viewer for cached triplets: {viewer_path}")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to generate viewer: {e}")

        return triplets, quality_ratings

    except FileNotFoundError as e:
        logger.info(f"  ✗ Cached triplets file not found: {e.filename}")
        return None
    except Exception as e:
        logger.warning(f"  ✗ Failed to load cached triplets: {e}")
        return None


def _extract_triplet_generation_config(task_or_config: _TaskLike | dict) -> dict:
    """Extract config fields that affect triplet generation.

    These fields determine whether cached triplets can be reused.
    Fields that don't affect triplet generation (e.g., evaluation method configs)
    are excluded.
    """
    is_task_object = not isinstance(task_or_config, dict)
    if isinstance(task_or_config, dict):
        config = task_or_config
    else:
        # Extract from Task object
        config = task_or_config.config

    # Convert DictConfig to regular dict to ensure JSON serialization works
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Core fields that define triplet identity
    triplet_config = {
        "document_set": config.get("document_set"),
        "criterion": config.get("criterion"),
        "triplet_style": config.get("triplet_style"),
    }

    # Optional fields that affect triplet generation
    optional_fields = [
        "max_docs",
        "max_triplets",
        "candidate_strategy",
        "embedding_preset",
        "lm_judge_preset",
        "use_spurious_hard_negs",
        "prelabeled_selection",
        "max_num_candidates",
        "split",
        "config",  # docset-specific config (e.g., subset for nytclustering)
        "num_synthetic_docs",
        "seed",
        # LM annotation hints (affect annotations which affect triplet quality)
        "criterion_description",
        "category_schema_hint",
        "tag_schema_hint",
        "summary_hint",
        "triplet_example_hint",
        "n_schema_samples",
        # Quality filtering
        "rate_triplet_quality",
        "min_triplet_quality",
    ]

    for field in optional_fields:
        # For Task objects, prefer resolved attributes (e.g., criterion_description
        # inherited from dataset metadata even when omitted from YAML config).
        if is_task_object and hasattr(task_or_config, field):
            value = getattr(task_or_config, field)
            if value is not None:
                if isinstance(value, DictConfig):
                    value = OmegaConf.to_container(value, resolve=True)
                triplet_config[field] = value
                continue

        if field in config:
            value = config[field]
            # Convert any nested DictConfig objects to regular dicts
            if isinstance(value, DictConfig):
                value = OmegaConf.to_container(value, resolve=True)
            triplet_config[field] = value

    return triplet_config


def save_triplet_config(
    *,
    task: _TaskLike,
    output_dir: str | Path,
    task_name: str,
) -> Path:
    """Save triplet generation config to `{output_dir}/{task_name}/triplet_config.json`.

    This enables checking if cached triplets were generated with the same config.
    """
    task_dir = _task_dir(output_dir, task_name)
    output_file = task_dir / "triplet_config.json"

    config = _extract_triplet_generation_config(task)

    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Saved triplet generation config to {output_file}")
    return output_file


def load_triplet_config(
    *,
    output_dir: str | Path,
    task_name: str,
) -> dict | None:
    """Load triplet generation config from `{output_dir}/{task_name}/triplet_config.json`.

    Returns None if the config file doesn't exist.
    """
    task_dir = Path(output_dir) / task_name
    config_file = task_dir / "triplet_config.json"

    if not config_file.exists():
        return None

    with open(config_file) as f:
        return json.load(f)


def triplet_config_matches(
    config1: dict,
    config2: dict,
    *,
    ignore_fields: list[str] | None = None,
) -> bool:
    """Check if two triplet generation configs would produce the same triplets.

    Args:
        config1: First config (can be full task config or extracted triplet config)
        config2: Second config (can be full task config or extracted triplet config)
        ignore_fields: Optional list of fields to ignore in comparison
                      (e.g., ["seed"] if you want to match configs regardless of seed)

    Returns:
        True if the configs would produce the same triplets, False otherwise.
    """
    # Extract just the triplet-relevant fields
    c1 = _extract_triplet_generation_config(config1)
    c2 = _extract_triplet_generation_config(config2)

    # Remove ignored fields
    if ignore_fields:
        for field in ignore_fields:
            c1.pop(field, None)
            c2.pop(field, None)

    return c1 == c2


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
    """Save a Task's triplets to JSONL under `{output_dir}/{task_name}/triplets.jsonl`.

    Also saves the triplet generation config to triplet_config.json for cache validation.
    """
    if task.documents is None:
        raise RuntimeError("Must call load_documents() before saving triplets")
    if task.triplets is None:
        raise RuntimeError("Must call create_triplets() before saving triplets")

    # Save the triplet generation config alongside triplets
    save_triplet_config(
        task=task,
        output_dir=output_dir,
        task_name=task.get_task_name(),
    )

    return save_triplets_json(
        documents=task.documents,
        triplets=task.triplets,
        output_dir=output_dir,
        task_name=task.get_task_name(),
        triplet_quality_ratings=task.triplet_quality_ratings,
        triplet_quality_ratings_with_annotations=(
            task.triplet_quality_ratings_with_annotations
        ),
        triplet_quality_ratings_without_annotations=(
            task.triplet_quality_ratings_without_annotations
        ),
        triplet_quality_reasoning=task.triplet_quality_reasoning,
        triplet_quality_reasoning_with_annotations=(
            task.triplet_quality_reasoning_with_annotations
        ),
        triplet_quality_reasoning_without_annotations=(
            task.triplet_quality_reasoning_without_annotations
        ),
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
                f"| Tags Jaccard (Median) | {stats.get('hard_pos_tags_median', 0):.3f} |\n"
            )
            f.write(
                f"| Tags Jaccard (25th - 75th percentile) | {stats.get('hard_pos_tags_p25', 0):.3f} - {stats.get('hard_pos_tags_p75', 0):.3f} |\n"
            )
            f.write(
                f"| Tags Jaccard (Min - Max) | {stats.get('hard_pos_tags_min', 0):.3f} - {stats.get('hard_pos_tags_max', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Mean ± Std) | {stats.get('hard_pos_spurious_mean', 0):.3f} ± {stats.get('hard_pos_spurious_std', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Median) | {stats.get('hard_pos_spurious_median', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (25th - 75th percentile) | {stats.get('hard_pos_spurious_p25', 0):.3f} - {stats.get('hard_pos_spurious_p75', 0):.3f} |\n"
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
                f"| Tags Jaccard (Median) | {stats.get('hard_neg_tags_median', 0):.3f} |\n"
            )
            f.write(
                f"| Tags Jaccard (25th - 75th percentile) | {stats.get('hard_neg_tags_p25', 0):.3f} - {stats.get('hard_neg_tags_p75', 0):.3f} |\n"
            )
            f.write(
                f"| Tags Jaccard (Min - Max) | {stats.get('hard_neg_tags_min', 0):.3f} - {stats.get('hard_neg_tags_max', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Mean ± Std) | {stats.get('hard_neg_spurious_mean', 0):.3f} ± {stats.get('hard_neg_spurious_std', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (Median) | {stats.get('hard_neg_spurious_median', 0):.3f} |\n"
            )
            f.write(
                f"| Spurious Jaccard (25th - 75th percentile) | {stats.get('hard_neg_spurious_p25', 0):.3f} - {stats.get('hard_neg_spurious_p75', 0):.3f} |\n"
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

        # Triplet quality analysis section
        if "triplet_quality" in stats:
            tq = stats["triplet_quality"]
            f.write("## Triplet Quality Analysis\n\n")
            f.write(
                "Analysis of triplet quality ratings by whether triplets involve synthetic documents.\n\n"
            )

            # Summary counts
            f.write("### Summary\n\n")
            f.write("| Category | Count |\n")
            f.write("|----------|-------|\n")
            f.write(
                f"| Triplets with synthetic docs | {tq['total_with_synthetic']} |\n"
            )
            f.write(
                f"| Triplets without synthetic docs | {tq['total_without_synthetic']} |\n\n"
            )

            # Quality rating distributions
            f.write("### Quality Rating Distribution\n\n")

            if tq["with_synthetic"]["count"] > 0:
                f.write("#### Triplets WITH Synthetic Documents\n\n")
                ws = tq["with_synthetic"]
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Total | {ws['count']} |\n")
                f.write(f"| Mean Rating | {ws['mean']:.2f} |\n")
                f.write(f"| Median Rating | {ws['median']:.1f} |\n")
                f.write(
                    f"| Rating 1 (Invalid) | {ws['rating_1']} ({ws['rating_1_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 2 (Ambiguous) | {ws['rating_2']} ({ws['rating_2_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 3 (Trivial) | {ws['rating_3']} ({ws['rating_3_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 4 (Acceptable) | {ws['rating_4']} ({ws['rating_4_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 5 (Ideal) | {ws['rating_5']} ({ws['rating_5_pct']:.1f}%) |\n\n"
                )

            if tq["without_synthetic"]["count"] > 0:
                f.write("#### Triplets WITHOUT Synthetic Documents\n\n")
                wos = tq["without_synthetic"]
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Total | {wos['count']} |\n")
                f.write(f"| Mean Rating | {wos['mean']:.2f} |\n")
                f.write(f"| Median Rating | {wos['median']:.1f} |\n")
                f.write(
                    f"| Rating 1 (Invalid) | {wos['rating_1']} ({wos['rating_1_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 2 (Ambiguous) | {wos['rating_2']} ({wos['rating_2_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 3 (Trivial) | {wos['rating_3']} ({wos['rating_3_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 4 (Acceptable) | {wos['rating_4']} ({wos['rating_4_pct']:.1f}%) |\n"
                )
                f.write(
                    f"| Rating 5 (Ideal) | {wos['rating_5']} ({wos['rating_5_pct']:.1f}%) |\n\n"
                )

            # Interpretation
            f.write("**Quality Scale:**\n")
            f.write("- 1 (Invalid): Anchor not closer to positive than negative\n")
            f.write(
                "- 2 (Ambiguous): Arguably closer but very ambiguous or subjective\n"
            )
            f.write("- 3 (Trivial): Obviously closer, not challenging\n")
            f.write("- 4 (Acceptable): Reasonably challenging and usable triplet\n")
            f.write("- 5 (Ideal): Excellent hard negative triplet\n\n")

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
