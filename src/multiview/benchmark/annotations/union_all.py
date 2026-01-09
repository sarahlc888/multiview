"""Union of all annotation types (lm_all).

This module orchestrates all annotation types together (mirrors clean_slate's lm_all.py):
- Generate schemas from sample documents
- Apply all annotation types to documents
- Combine into rich annotations with categories, tags, and summaries

Named "union_all" to reflect that it combines/unites all annotation types.
"""

from __future__ import annotations

import logging

from multiview.benchmark.annotations.class_schema import (
    classify_documents_batch,
    generate_category_schema,
)
from multiview.benchmark.annotations.open_ended import (
    generate_criteria_description,
    generate_summaries_batch,
    generate_summary_guidance,
)
from multiview.benchmark.annotations.tag_schema import (
    apply_tags_batch,
    generate_spurious_tag_schema,
    generate_tag_schema,
)

logger = logging.getLogger(__name__)


def annotate_with_lm_all(
    documents: list[str],
    criterion: str,
    criterion_description: str | None = None,
    n_schema_samples: int = 10,
    category_schema_hint: str | None = None,
    tag_schema_hint: str | None = None,
    summary_guidance_hint: str | None = None,
    summary_format_hint: str | None = None,
    include_debug: bool = False,
    cache_alias_prefix: str | None = None,
) -> list[dict]:
    """Annotate documents with rich multi-faceted annotations.

    This is the main entry point for multi-faceted annotation. It:
    1. Generates schemas from sample documents
    2. Applies schemas to all documents
    3. Returns rich annotations with categories, tags, spurious tags, and summaries

    Args:
        documents: List of document strings
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Description of what the criterion means
        n_schema_samples: Number of documents to sample for schema generation
        category_schema_hint: Optional hint for category schema generation
        tag_schema_hint: Optional hint for tag schema generation
        summary_guidance_hint: Optional hint for summary guidance
        summary_format_hint: Optional hint for summary format
        include_debug: If True, include debug/reasoning info
        cache_alias_prefix: Prefix for cache aliases

    Returns:
        List of rich annotation dicts, one per document. Each dict has the structure:
        {
            "criterion_value": None,  # Backward compatibility
            "category": "category_name",  # Single category classification
            "tags": {"tag1": True, "tag2": False, ...},  # Binary tag labels
            "spurious_tags": {"stag1": False, ...},  # Spurious/superficial properties
            "summary": {
                "annotation_trace": "...",  # Step-by-step reasoning
                "final_summary": "..."  # Final annotation summary
            },
            "category_schema": {...},  # Generated category schema
            "tag_schema": {...},  # Generated tag schema
            "spurious_tag_schema": {...},  # Generated spurious tag schema
            "summary_guidance": {...}  # Generated summary guidance
        }
    """
    logger.info(
        f"Starting multi-faceted annotation for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    # Generate cache aliases for schemas
    schema_cache_prefix = f"schema_{criterion}" if cache_alias_prefix else None

    # Step 1: Generate schemas from sample documents
    logger.info("Step 1/2: Generating annotation schemas...")

    category_schema = generate_category_schema(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description or "",
        n_samples=n_schema_samples,
        schema_hint=category_schema_hint,
        cache_alias=f"{schema_cache_prefix}_category" if schema_cache_prefix else None,
    )

    tag_schema = generate_tag_schema(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description or "",
        n_samples=n_schema_samples,
        schema_hint=tag_schema_hint,
        is_spurious=False,
        cache_alias=f"{schema_cache_prefix}_tags" if schema_cache_prefix else None,
    )

    spurious_tag_schema = generate_spurious_tag_schema(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description or "",
        n_samples=n_schema_samples,
        cache_alias=f"{schema_cache_prefix}_spurious" if schema_cache_prefix else None,
    )

    # Generate enhanced criteria description for summaries
    enhanced_criteria = generate_criteria_description(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description or "",
        n_samples=n_schema_samples,
        cache_alias=f"{schema_cache_prefix}_criteria_desc"
        if schema_cache_prefix
        else None,
    )
    summary_criteria_desc = (
        enhanced_criteria.get("description") or criterion_description or ""
    )

    summary_guidance = generate_summary_guidance(
        documents=documents,
        criterion=criterion,
        criterion_description=summary_criteria_desc,  # Use enhanced description
        n_samples=n_schema_samples,
        guidance_hint=summary_guidance_hint,
        format_hint=summary_format_hint,
        cache_alias=f"{schema_cache_prefix}_guidance" if schema_cache_prefix else None,
    )

    # Step 2: Apply schemas to all documents
    logger.info("Step 2/2: Applying schemas to all documents...")

    # Generate cache aliases
    cat_alias = f"{cache_alias_prefix}_category" if cache_alias_prefix else None
    tag_alias = f"{cache_alias_prefix}_tags" if cache_alias_prefix else None
    spur_alias = f"{cache_alias_prefix}_spurious" if cache_alias_prefix else None
    summ_alias = f"{cache_alias_prefix}_summary" if cache_alias_prefix else None

    # Run all annotation types
    category_annotations = classify_documents_batch(
        documents,
        criterion,
        criterion_description or "",
        category_schema,
        cache_alias=cat_alias,
    )

    tag_annotations = apply_tags_batch(
        documents,
        criterion,
        criterion_description or "",
        tag_schema,
        cache_alias=tag_alias,
    )

    spurious_annotations = apply_tags_batch(
        documents,
        criterion,
        "spurious/superficial properties",
        spurious_tag_schema,
        cache_alias=spur_alias,
    )

    summary_annotations = generate_summaries_batch(
        documents,
        criterion,
        summary_criteria_desc,  # Use enhanced description
        summary_guidance,
        cache_alias=summ_alias,
    )

    # Combine annotations
    rich_annotations = []
    for i in range(len(documents)):
        annotation = {
            "criterion_value": None,  # Backward compatibility
            "category": category_annotations[i].get("category"),
            "tags": tag_annotations[i].get("tags", {}),
            "spurious_tags": spurious_annotations[i].get("tags", {}),
            "summary": summary_annotations[i].get("summary", {}),
            # Include schemas for reproducibility
            "category_schema": category_schema,
            "tag_schema": tag_schema,
            "spurious_tag_schema": spurious_tag_schema,
            "summary_guidance": summary_guidance,
        }

        rich_annotations.append(annotation)

    logger.info(f"Completed multi-faceted annotation for {len(documents)} documents")
    return rich_annotations
