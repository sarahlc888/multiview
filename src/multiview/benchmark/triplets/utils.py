"""Utility functions for triplet creation."""

import json


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


def format_annotation_for_display(ann: dict) -> str:
    """Format annotation dict for display in LM judge prompts.

    Args:
        ann: Annotation dict with category, tags, spurious_tags, summary

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

    # Summary (with fallback to annotation_trace if final_summary is empty)
    summary_dict = ann.get("summary", {})
    if isinstance(summary_dict, dict):
        final_summary = summary_dict.get("final_summary", "") or summary_dict.get(
            "annotation_trace", ""
        )
    else:
        final_summary = str(summary_dict)

    # Serialize structured summaries (list/dict) to JSON for display
    if isinstance(final_summary, (list | dict)):
        final_summary = json.dumps(final_summary)

    if final_summary:
        parts.append(f"Summary: {final_summary}")

    return "\n".join(parts) if parts else "No annotation"
