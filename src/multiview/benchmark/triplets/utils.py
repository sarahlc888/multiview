"""Utility functions for triplet creation."""

import json


def build_triplet_dicts(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
) -> list[dict]:
    """Convert ID triplets into text triplet dicts (doc IDs NEVER in prompts!)."""
    return [
        {
            "anchor": documents[anchor_id],
            "positive": documents[positive_id],
            "negative": documents[negative_id],
            # Include IDs for annotation lookup / artifact writing
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
        }
        for anchor_id, positive_id, negative_id in triplet_ids
    ]


def coerce_to_text(value) -> str:
    """Convert a value into a reasonably-readable string for prompts/logging."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def annotation_final_summary(annotation: dict | None) -> str:
    """Extract ONLY summary.final_summary (never annotation traces), coerced to text."""
    if not isinstance(annotation, dict):
        return ""

    summary = annotation.get("summary", {})
    final_summary = ""

    if isinstance(summary, list):
        if summary and isinstance(summary[0], dict):
            final_summary = summary[0].get("final_summary", "")
        else:
            final_summary = ""
    elif isinstance(summary, dict):
        final_summary = summary.get("final_summary", "")
    else:
        # Unexpected legacy format: treat as the final summary value.
        final_summary = summary

    return coerce_to_text(final_summary)


def triplet_annotation_summary(
    *,
    triplets: list[dict],
    triplet_idx: int,
    triplet_key: str,  # "anchor" | "positive" | "negative"
    annotations: list[dict] | None,
) -> str:
    """Get a summary string for a doc in a triplet.

    Prefer attached `{key}_annotation` if present; otherwise resolve via `{key}_id`
    into the provided `annotations` list.
    """
    t = triplets[triplet_idx]

    attached_key = f"{triplet_key}_annotation"
    _MISSING = object()
    ann = t.get(attached_key, _MISSING)
    if ann is not _MISSING:
        return (
            annotation_final_summary(ann)
            if isinstance(ann, dict)
            else coerce_to_text(ann)
        )

    id_key = f"{triplet_key}_id"
    doc_id = t.get(id_key, _MISSING)
    return (
        annotation_final_summary(annotations[doc_id])
        if (
            annotations is not None
            and doc_id is not _MISSING
            and isinstance(doc_id, int)
            and 0 <= doc_id < len(annotations)
        )
        else ""
    )


def add_annotation_summaries_to_inputs(
    inputs: dict[str, list[str]],
    *,
    triplets: list[dict],
    annotations: list[dict] | None,
    triplet_keys_by_input_suffix: dict[str, str],
) -> dict[str, list[str]]:
    """Add `annotation_*` columns to an inference input dict.

    Intended for LM judge presets that take annotation fields like `annotation_a`,
    `annotation_b`, `annotation_c`.
    """
    if not annotations:
        return inputs

    for suffix, triplet_key in triplet_keys_by_input_suffix.items():
        inputs[f"annotation_{suffix}"] = [
            triplet_annotation_summary(
                triplets=triplets,
                triplet_idx=i,
                triplet_key=triplet_key,
                annotations=annotations,
            )
            for i in range(len(triplets))
        ]

    return inputs


def triplet_full_annotation(
    *,
    triplets: list[dict],
    triplet_idx: int,
    triplet_key: str,  # "anchor" | "positive" | "negative"
    annotations: list[dict] | None,
) -> dict | None:
    """Return the full annotation object for a triplet member, if resolvable."""
    if annotations is None:
        return None

    id_key = f"{triplet_key}_id"
    t = triplets[triplet_idx]
    if id_key not in t:
        return None

    doc_id = t[id_key]
    if not isinstance(doc_id, int) or not (0 <= doc_id < len(annotations)):
        return None

    ann = annotations[doc_id]
    return ann if isinstance(ann, dict) else None


def extract_active_tags(annotation: dict, tag_key: str = "tags") -> set[str]:
    """Extract active tag names from an annotation dict.

    Args:
        annotation: Annotation dict with tag key
        tag_key: Key to extract tags from (default: "tags", can be "spurious_tags")

    Returns:
        Set of active tag names (where value is True)
    """
    return {
        tag_name for tag_name, value in annotation.get(tag_key, {}).items() if value
    }


def jaccard_similarity(a: set, b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Args:
        a: First set
        b: Second set

    Returns:
        Jaccard similarity (intersection / union), or 0.0 if both sets are empty
    """
    if not a and not b:
        return 0.0
    union = a.union(b)
    if not union:
        return 0.0
    intersection = a.intersection(b)
    return len(intersection) / len(union)


def format_annotation_for_display(ann: dict, include_spurious: bool = False) -> str:
    """Format annotation dict for display in LM judge prompts.

    Args:
        ann: Annotation dict with category, tags, spurious_tags, summary
        include_spurious: If True, include spurious_tags in the output

    Returns:
        Formatted string with annotation details
    """
    parts = []

    # Category
    if ann.get("category"):
        parts.append(f"Category: {ann['category']}")

    # Tags (use helper to extract active tags)
    active_tags = extract_active_tags(ann, "tags")
    if active_tags:
        parts.append(f"Tags: {', '.join(sorted(active_tags))}")

    # Spurious tags (if requested)
    if include_spurious:
        spurious_tags = extract_active_tags(ann, "spurious_tags")
        if spurious_tags:
            parts.append(f"Spurious tags: {', '.join(sorted(spurious_tags))}")

    # Summary - ONLY use final_summary, never annotation_trace
    final_summary = annotation_final_summary(ann)
    if final_summary:
        parts.append(f"Summary: {final_summary}")

    return "\n".join(parts) if parts else "No annotation"
