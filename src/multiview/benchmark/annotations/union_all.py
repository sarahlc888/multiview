"""Union of all annotation types (lm_all).

This module orchestrates all annotation types together (mirrors clean_slate's lm_all.py):
- Generate schemas from sample documents
- Apply all annotation types to documents
- Combine into rich annotations with categories, tags, and summaries

Named "union_all" to reflect that it combines/unites all annotation types.
"""

from __future__ import annotations

import logging

from multiview.benchmark.annotations.annotation_utils import extract_image, extract_text
from multiview.benchmark.annotations.class_schema import (
    classify_documents_batch,
    generate_category_schema,
)
from multiview.benchmark.annotations.open_ended import (
    generate_pairwise_sim_hint,
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
    documents: list[str | dict],
    criterion: str,
    document_type: str,
    criterion_description: str | None = None,
    n_schema_samples: int = 10,
    category_schema_hint: str | None = None,
    tag_schema_hint: str | None = None,
    summary_hint: str | None = None,
    include_debug: bool = False,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str | dict] | None = None,
    category_schema_preset: str | None = None,
    category_classify_preset: str | None = None,
    tag_schema_preset: str | None = None,
    spurious_tag_schema_preset: str | None = None,
    tag_apply_preset: str | None = None,
    summary_guidance_preset: str | None = None,
    summary_generate_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with rich multi-faceted annotations.

    This is the main entry point for multi-faceted annotation. It:
    1. Generates schemas from sample documents
    2. Applies schemas to all documents
    3. Returns rich annotations with categories, tags, spurious tags, and summaries

    Args:
        documents: List of documents (strings or dicts) to annotate
        criterion: Criterion name (e.g., "arithmetic_operations")
        document_type: Type of documents being annotated (e.g., "math word problem", "story", "sentence")
        criterion_description: Brief description of what the criterion means
        n_schema_samples: Number of documents to sample for schema generation
        category_schema_hint: Optional hint for category schema generation
        tag_schema_hint: Optional hint for tag schema generation
        summary_hint: Optional rich description/hint for summary guidance.
            If not provided, will be auto-generated from criterion_description and sample documents.
        include_debug: If True, include debug/reasoning info
        cache_alias_prefix: Prefix for cache aliases
        run_name: Optional experiment/run name for cache organization
        schema_documents: Optional separate list of documents to use for schema generation.
            If not provided, uses `documents` for both schema generation and annotation.
        category_schema_preset: Preset for category schema generation (default: "category_schema_generation_gemini")
        category_classify_preset: Preset for category classification (default: "category_classify_gemini")
        tag_schema_preset: Preset for tag schema generation (default: "tag_schema_generation_gemini")
        spurious_tag_schema_preset: Preset for spurious tag schema (default: "spurious_tag_schema_generation_gemini")
        tag_apply_preset: Preset for tag application (default: "tag_apply_gemini")
        summary_guidance_preset: Preset for summary guidance generation (default: "summary_guidance_generation_gemini")
        summary_generate_preset: Preset for summary generation (default: "summary_generate_gemini")

    Returns:
        List of rich annotation dicts, one per document. Each dict has the structure:
        {
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

    # Set defaults for presets
    category_schema_preset = (
        category_schema_preset or "category_schema_generation_gemini"
    )
    category_classify_preset = category_classify_preset or "category_classify_gemini"
    tag_schema_preset = tag_schema_preset or "tag_schema_generation_gemini"
    spurious_tag_schema_preset = (
        spurious_tag_schema_preset or "spurious_tag_schema_generation_gemini"
    )
    tag_apply_preset = tag_apply_preset or "tag_apply_gemini"
    summary_guidance_preset = (
        summary_guidance_preset or "summary_guidance_generation_gemini"
    )
    summary_generate_preset = summary_generate_preset or "summary_generate_gemini"

    # Use schema_documents for schema generation if provided, otherwise use documents
    schema_pool = schema_documents if schema_documents is not None else documents

    # Extract text strings and images
    schema_pool_texts = [extract_text(doc) for doc in schema_pool]
    schema_pool_images = [extract_image(doc) for doc in schema_pool]
    document_texts = [extract_text(doc) for doc in documents]
    document_images = [extract_image(doc) for doc in documents]

    # Only pass images if at least one is present
    has_schema_images = any(img is not None for img in schema_pool_images)
    has_doc_images = any(img is not None for img in document_images)

    # Generate cache aliases for schemas
    schema_cache_prefix = f"schema_{criterion}" if cache_alias_prefix else None

    # Step 1: Generate schemas from sample documents
    logger.info("Step 1/2: Generating annotation schemas...")

    category_schema = generate_category_schema(
        documents=schema_pool_texts,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        schema_hint=category_schema_hint,
        cache_alias=f"{schema_cache_prefix}_category" if schema_cache_prefix else None,
        run_name=run_name,
        config=category_schema_preset,
        images=schema_pool_images if has_schema_images else None,
    )

    tag_schema = generate_tag_schema(
        documents=schema_pool_texts,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        schema_hint=tag_schema_hint,
        is_spurious=False,
        cache_alias=f"{schema_cache_prefix}_tags" if schema_cache_prefix else None,
        run_name=run_name,
        config=tag_schema_preset,
        images=schema_pool_images if has_schema_images else None,
    )

    spurious_tag_schema = generate_spurious_tag_schema(
        documents=schema_pool_texts,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        cache_alias=f"{schema_cache_prefix}_spurious" if schema_cache_prefix else None,
        run_name=run_name,
        config=spurious_tag_schema_preset,
        images=schema_pool_images if has_schema_images else None,
    )

    # If summary_hint not provided, auto-generate it from samples
    enriched_summary_hint = summary_hint
    if not enriched_summary_hint:
        logger.info("No summary_hint provided, generating from samples...")
        generated_hint = generate_pairwise_sim_hint(
            documents=schema_pool_texts,
            criterion=criterion,
            criterion_description=criterion_description or "",
            document_type=document_type,
            n_samples=n_schema_samples,
            cache_alias=f"{schema_cache_prefix}_summary_hint"
            if schema_cache_prefix
            else None,
            run_name=run_name,
            images=schema_pool_images if has_schema_images else None,
        )
        enriched_summary_hint = generated_hint.get("summary_hint") or ""

    summary_guidance = generate_summary_guidance(
        documents=schema_pool_texts,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        summary_hint=enriched_summary_hint,
        cache_alias=f"{schema_cache_prefix}_guidance" if schema_cache_prefix else None,
        run_name=run_name,
        guidance_preset=summary_guidance_preset,
        images=schema_pool_images if has_schema_images else None,
    )

    # Step 2: Apply schemas to all documents
    logger.info("Step 2/2: Applying schemas to all documents...")

    # Generate cache aliases
    cat_alias = f"{cache_alias_prefix}_category" if cache_alias_prefix else None
    tag_alias = f"{cache_alias_prefix}_tags" if cache_alias_prefix else None
    spur_alias = f"{cache_alias_prefix}_spurious" if cache_alias_prefix else None
    summ_alias = f"{cache_alias_prefix}_summary" if cache_alias_prefix else None

    # Run all annotation types (using extracted text strings and images)
    category_annotations = classify_documents_batch(
        document_texts,
        criterion,
        criterion_description or "",
        category_schema,
        document_type=document_type,
        cache_alias=cat_alias,
        run_name=run_name,
        config=category_classify_preset,
        images=document_images if has_doc_images else None,
    )

    tag_annotations = apply_tags_batch(
        document_texts,
        criterion,
        criterion_description or "",
        tag_schema,
        cache_alias=tag_alias,
        run_name=run_name,
        config=tag_apply_preset,
        images=document_images if has_doc_images else None,
    )

    spurious_annotations = apply_tags_batch(
        document_texts,
        criterion,
        "spurious/superficial properties",
        spurious_tag_schema,
        cache_alias=spur_alias,
        run_name=run_name,
        config=tag_apply_preset,
        images=document_images if has_doc_images else None,
    )

    summary_annotations = generate_summaries_batch(
        document_texts,
        criterion,
        criterion_description or "",
        summary_guidance,
        cache_alias=summ_alias,
        run_name=run_name,
        generate_preset=summary_generate_preset,
        images=document_images if has_doc_images else None,
    )

    # Combine annotations
    rich_annotations = []
    for i in range(len(documents)):
        annotation = {
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
