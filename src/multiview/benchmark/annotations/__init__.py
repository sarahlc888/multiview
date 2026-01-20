"""Document annotation utilities.

This package provides functions for annotating documents with:
- Category classification (class_schema.py)
- Multi-label binary tags (tag_schema.py)
- Open-ended summaries (open_ended.py)
- All annotation types combined (union_all.py)

Module Organization:
    class_schema.py: Single-category classification
        - generate_category_schema(): Generate category schema from samples
        - classify_documents_batch(): Classify multiple documents

    tag_schema.py: Multi-label binary tag annotation
        - generate_tag_schema(): Generate tag schema from samples
        - generate_spurious_tag_schema(): Generate spurious tag schema
        - apply_tags_batch(): Apply tags to multiple documents

    open_ended.py: Open-ended summary generation
        - generate_pairwise_sim_hint(): Generate rich summary hint from samples (auto-enriches brief criterion descriptions)
        - generate_summary_guidance(): Generate summary guidance from samples
        - generate_summaries_batch(): Generate summaries for multiple documents

    union_all.py: Unified multi-faceted annotation (⭐ Main Entry Point)
        - annotate_with_lm_all(): Orchestrates all annotation types together
          * Generates schemas from sample documents
          * Applies all annotation types (categories, tags, summaries)
          * Returns rich annotations combining all information
"""

import logging

# Schema generation
# Batch annotation
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

# Main "all" annotation (from union_all.py, mirrors lm_all.py from clean_slate)
from multiview.benchmark.annotations.union_all import annotate_with_lm_all

logger = logging.getLogger(__name__)


def annotate_with_known_criterion(
    documents: list[str],
    document_set,
    criterion: str,
) -> list[dict]:
    """Annotate documents using known criterion extraction.

    Args:
        documents: List of document strings
        document_set: DocumentSet instance with extraction method
        criterion: Criterion name to extract

    Returns:
        List of annotation dicts with prelabel
    """
    annotations = []
    for doc in documents:
        value = document_set.get_known_criterion_value(doc, criterion)
        annotations.append({"prelabel": value})

    logger.debug(f"Extracted {criterion} for {len(documents)} documents")
    return annotations


def annotate_with_precomputed(
    documents: list[str],
    document_set,
    criterion: str,
) -> list[dict]:
    """Load pre-computed annotations from dataset.

    Args:
        documents: List of document strings or dicts
        document_set: DocumentSet instance with precomputed annotations
        criterion: Criterion name to load annotations for

    Returns:
        List of annotation dicts (format: {"prelabel": value})
    """
    precomputed = document_set.get_precomputed_annotations(criterion)

    annotations = []
    for doc in documents:
        # Extract text from document (handles both string and dict documents)
        doc_text = document_set.get_document_text(doc)

        if doc_text in precomputed:
            annotations.append(precomputed[doc_text])
        else:
            logger.warning(
                f"Document not found in precomputed annotations for {criterion}: {doc_text[:100]}..."
            )
            annotations.append({"prelabel": None})

    logger.info(
        f"Loaded precomputed annotations for {criterion}: {len(annotations)} documents"
    )
    return annotations


def annotate_with_lm_category(
    documents: list[str],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    category_schema_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str] | None = None,
    category_schema_preset: str | None = None,
    category_classify_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with category classification only.

    Args:
        documents: List of document strings to annotate
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Description of what the criterion means
        document_type: Type of documents (e.g., "news article", "math problem")
        n_schema_samples: Number of documents to sample for schema generation
        category_schema_hint: Optional hint for category schema generation
        cache_alias_prefix: Prefix for cache aliases
        run_name: Optional experiment/run name for cache organization
        schema_documents: Optional separate list of documents to use for schema generation.
            If not provided, uses `documents` for both schema generation and annotation.
        category_schema_preset: Preset to use for category schema generation.
            Defaults to "category_schema_generation_gemini".
        category_classify_preset: Preset to use for category classification.
            Defaults to "category_classify_gemini".

    Returns:
        List of annotation dicts, one per document. Each dict has the structure:
        {
            "category": "category_name",  # Single category classification
            "category_reasoning": "...",  # Reasoning for the classification
            "category_schema": {...}  # Generated category schema
        }
    """
    logger.info(
        f"Starting category annotation for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    # Use schema_documents for schema generation if provided, otherwise use documents
    schema_pool = schema_documents if schema_documents is not None else documents

    # Generate cache alias for schema
    schema_cache_prefix = f"schema_{criterion}" if cache_alias_prefix else None

    # Step 1: Generate category schema from sample documents
    logger.info("Step 1/2: Generating category schema...")
    category_schema = generate_category_schema(
        documents=schema_pool,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        schema_hint=category_schema_hint,
        cache_alias=f"{schema_cache_prefix}_category" if schema_cache_prefix else None,
        run_name=run_name,
        config=category_schema_preset or "category_schema_generation_gemini",
    )

    # Step 2: Apply schema to all documents
    logger.info("Step 2/2: Applying category schema to all documents...")
    cat_alias = f"{cache_alias_prefix}_category" if cache_alias_prefix else None

    category_annotations = classify_documents_batch(
        documents,
        criterion,
        criterion_description or "",
        category_schema,
        document_type=document_type,
        cache_alias=cat_alias,
        run_name=run_name,
        config=category_classify_preset or "category_classify_gemini",
    )

    # Combine annotations
    rich_annotations = []
    for i in range(len(documents)):
        annotation = {
            "category": category_annotations[i].get("category"),
            "category_reasoning": category_annotations[i].get("category_reasoning"),
            "category_schema": category_schema,
        }
        rich_annotations.append(annotation)

    logger.info(f"Completed category annotation for {len(documents)} documents")
    return rich_annotations


def annotate_with_lm_tags(
    documents: list[str],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    tag_schema_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str] | None = None,
    tag_schema_preset: str | None = None,
    spurious_tag_schema_preset: str | None = None,
    tag_apply_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with criterion-relevant and spurious tags.

    Args:
        documents: List of document strings to annotate
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Description of what the criterion means
        document_type: Type of documents (e.g., "news article", "math problem")
        n_schema_samples: Number of documents to sample for schema generation
        tag_schema_hint: Optional hint for tag schema generation
        cache_alias_prefix: Prefix for cache aliases
        run_name: Optional experiment/run name for cache organization
        schema_documents: Optional separate list of documents to use for schema generation.
            If not provided, uses `documents` for both schema generation and annotation.
        tag_schema_preset: Preset to use for tag schema generation.
            Defaults to "tag_schema_generation_gemini".
        spurious_tag_schema_preset: Preset to use for spurious tag schema generation.
            Defaults to "spurious_tag_schema_generation_gemini".
        tag_apply_preset: Preset to use for tag application.
            Defaults to "tag_apply_gemini".

    Returns:
        List of annotation dicts, one per document. Each dict has the structure:
        {
            "tags": {"tag1": True, "tag2": False, ...},  # Binary tag labels
            "spurious_tags": {"stag1": False, ...},  # Spurious/superficial properties
            "tag_schema": {...},  # Generated tag schema
            "spurious_tag_schema": {...}  # Generated spurious tag schema
        }
    """
    logger.info(
        f"Starting tag annotation for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    # Use schema_documents for schema generation if provided, otherwise use documents
    schema_pool = schema_documents if schema_documents is not None else documents

    # Generate cache alias for schemas
    schema_cache_prefix = f"schema_{criterion}" if cache_alias_prefix else None

    # Step 1: Generate tag schemas from sample documents
    logger.info("Step 1/2: Generating tag schemas...")

    tag_schema = generate_tag_schema(
        documents=schema_pool,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        schema_hint=tag_schema_hint,
        is_spurious=False,
        cache_alias=f"{schema_cache_prefix}_tags" if schema_cache_prefix else None,
        run_name=run_name,
        config=tag_schema_preset,
    )

    spurious_tag_schema = generate_spurious_tag_schema(
        documents=schema_pool,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        cache_alias=f"{schema_cache_prefix}_spurious" if schema_cache_prefix else None,
        run_name=run_name,
        config=spurious_tag_schema_preset,
    )

    # Step 2: Apply schemas to all documents
    logger.info("Step 2/2: Applying tag schemas to all documents...")

    tag_alias = f"{cache_alias_prefix}_tags" if cache_alias_prefix else None
    spur_alias = f"{cache_alias_prefix}_spurious" if cache_alias_prefix else None

    tag_annotations = apply_tags_batch(
        documents,
        criterion,
        criterion_description or "",
        tag_schema,
        cache_alias=tag_alias,
        run_name=run_name,
        config=tag_apply_preset or "tag_apply_gemini",
    )

    spurious_annotations = apply_tags_batch(
        documents,
        criterion,
        "spurious/superficial properties",
        spurious_tag_schema,
        cache_alias=spur_alias,
        run_name=run_name,
        config=tag_apply_preset or "tag_apply_gemini",
    )

    # Combine annotations
    rich_annotations = []
    for i in range(len(documents)):
        annotation = {
            "tags": tag_annotations[i].get("tags", {}),
            "spurious_tags": spurious_annotations[i].get("tags", {}),
            "tag_schema": tag_schema,
            "spurious_tag_schema": spurious_tag_schema,
        }
        rich_annotations.append(annotation)

    logger.info(f"Completed tag annotation for {len(documents)} documents")
    return rich_annotations


def annotate_with_lm_summary_dict(
    documents: list[str],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    summary_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str] | None = None,
    summary_guidance_preset: str | None = None,
    summary_generate_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with structured/dictionary summaries.

    Args:
        documents: List of document strings to annotate
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Brief description of what the criterion means
        document_type: Type of documents (e.g., "haiku", "math problem")
        n_schema_samples: Number of documents to sample for schema generation
        summary_hint: Optional rich description/hint for summary guidance.
            If not provided, will be auto-generated from criterion_description and sample documents.
        cache_alias_prefix: Prefix for cache aliases
        run_name: Optional experiment/run name for cache organization
        schema_documents: Optional separate list of documents to use for schema generation.
            If not provided, uses `documents` for both schema generation and annotation.
        summary_guidance_preset: Preset to use for summary guidance generation.
            Defaults to "summary_guidance_generation_dict_gemini".
        summary_generate_preset: Preset to use for summary generation.
            Defaults to "summary_generate_dict_gemini".

    Returns:
        List of annotation dicts, one per document. Each dict has the structure:
        {
            "summary": {
                "annotation_trace": "...",  # Step-by-step reasoning
                "final_summary": "..."  # Final annotation summary
            },
            "summary_guidance": {...}  # Generated summary guidance
        }
    """
    logger.info(
        f"Starting summary annotation for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    # Use schema_documents for schema generation if provided, otherwise use documents
    schema_pool = schema_documents if schema_documents is not None else documents

    # Generate cache alias for schemas
    schema_cache_prefix = f"schema_{criterion}" if cache_alias_prefix else None

    # Step 1: Generate summary guidance from sample documents
    logger.info("Step 1/2: Generating summary guidance...")

    # If summary_hint not provided, auto-generate it from samples
    enriched_summary_hint = summary_hint
    if not enriched_summary_hint:
        logger.info("No summary_hint provided, generating from samples...")
        generated_hint = generate_pairwise_sim_hint(
            documents=schema_pool,
            criterion=criterion,
            criterion_description=criterion_description or "",
            document_type=document_type,
            n_samples=n_schema_samples,
            cache_alias=f"{schema_cache_prefix}_summary_hint"
            if schema_cache_prefix
            else None,
            run_name=run_name,
        )
        enriched_summary_hint = generated_hint.get("summary_hint") or ""

    summary_guidance = generate_summary_guidance(
        documents=schema_pool,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        summary_hint=enriched_summary_hint,
        cache_alias=f"{schema_cache_prefix}_guidance" if schema_cache_prefix else None,
        run_name=run_name,
        guidance_preset=summary_guidance_preset
        or "summary_guidance_generation_dict_gemini",
    )

    # Step 2: Apply guidance to all documents
    logger.info("Step 2/2: Applying summary guidance to all documents...")
    summ_alias = f"{cache_alias_prefix}_summary" if cache_alias_prefix else None

    summary_annotations = generate_summaries_batch(
        documents,
        criterion,
        criterion_description or "",
        summary_guidance,
        cache_alias=summ_alias,
        run_name=run_name,
        generate_preset=summary_generate_preset or "summary_generate_dict_gemini",
    )

    # Combine annotations
    rich_annotations = []
    for i in range(len(documents)):
        annotation = {
            "summary": summary_annotations[i].get("summary", {}),
            "summary_guidance": summary_guidance,
        }
        rich_annotations.append(annotation)

    logger.info(f"Completed summary annotation for {len(documents)} documents")
    return rich_annotations


def annotate_with_lm_summary_sentence(
    documents: list[str],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    summary_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str] | None = None,
    summary_guidance_preset: str | None = None,
    summary_generate_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with one-sentence prose summaries.

    Uses sentence-specific prompts that encourage natural language prose output.

    Args:
        documents: List of document strings to annotate
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Brief description of what the criterion means
        document_type: Type of documents (e.g., "haiku", "math problem")
        n_schema_samples: Number of documents to sample for schema generation
        summary_hint: Optional rich description/hint for summary guidance.
            If not provided, will be auto-generated from criterion_description and sample documents.
        cache_alias_prefix: Prefix for cache aliases
        run_name: Optional experiment/run name for cache organization
        schema_documents: Optional separate list of documents to use for schema generation.
            If not provided, uses `documents` for both schema generation and annotation.
        summary_guidance_preset: Preset to use for summary guidance generation.
            Defaults to "summary_guidance_generation_sentence_gemini".
        summary_generate_preset: Preset to use for summary generation.
            Defaults to "summary_generate_sentence_gemini".

    Returns:
        List of annotation dicts, one per document. Each dict has the structure:
        {
            "summary": {
                "annotation_trace": "...",  # Step-by-step reasoning
                "final_summary": "..."  # Final annotation summary (single sentence)
            },
            "summary_guidance": {...}  # Generated summary guidance
        }
    """
    logger.info(
        f"Starting sentence-mode summary annotation for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    # Use schema_documents for schema generation if provided, otherwise use documents
    schema_pool = schema_documents if schema_documents is not None else documents

    # Generate cache alias for schemas
    schema_cache_prefix = f"schema_{criterion}" if cache_alias_prefix else None

    # Step 1: Generate summary guidance from sample documents
    logger.info("Step 1/2: Generating summary guidance (sentence mode)...")

    # If summary_hint not provided, auto-generate it from samples
    enriched_summary_hint = summary_hint
    if not enriched_summary_hint:
        logger.info("No summary_hint provided, generating from samples...")
        generated_hint = generate_pairwise_sim_hint(
            documents=schema_pool,
            criterion=criterion,
            criterion_description=criterion_description or "",
            document_type=document_type,
            n_samples=n_schema_samples,
            cache_alias=f"{schema_cache_prefix}_summary_hint"
            if schema_cache_prefix
            else None,
            run_name=run_name,
        )
        enriched_summary_hint = generated_hint.get("summary_hint") or ""

    summary_guidance = generate_summary_guidance(
        documents=schema_pool,
        criterion=criterion,
        criterion_description=criterion_description or "",
        document_type=document_type,
        n_samples=n_schema_samples,
        summary_hint=enriched_summary_hint,
        cache_alias=f"{schema_cache_prefix}_guidance" if schema_cache_prefix else None,
        run_name=run_name,
        guidance_preset=summary_guidance_preset
        or "summary_guidance_generation_sentence_gemini",
    )

    # Step 2: Apply guidance to all documents
    logger.info(
        "Step 2/2: Applying summary guidance to all documents (sentence mode)..."
    )
    summ_alias = f"{cache_alias_prefix}_summary" if cache_alias_prefix else None

    summary_annotations = generate_summaries_batch(
        documents,
        criterion,
        criterion_description or "",
        summary_guidance,
        cache_alias=summ_alias,
        run_name=run_name,
        generate_preset=summary_generate_preset or "summary_generate_sentence_gemini",
    )

    # Combine annotations
    rich_annotations = []
    for i in range(len(documents)):
        annotation = {
            "summary": summary_annotations[i].get("summary", {}),
            "summary_guidance": summary_guidance,
        }
        rich_annotations.append(annotation)

    logger.info(
        f"Completed sentence-mode summary annotation for {len(documents)} documents"
    )
    return rich_annotations


__all__ = [
    # Main entry points
    "annotate_with_known_criterion",
    "annotate_with_precomputed",
    "annotate_with_lm_all",
    "annotate_with_lm_category",
    "annotate_with_lm_tags",
    "annotate_with_lm_summary_dict",
    "annotate_with_lm_summary_sentence",
    # Schema generation
    "generate_category_schema",
    "generate_tag_schema",
    "generate_spurious_tag_schema",
    "generate_pairwise_sim_hint",
    "generate_summary_guidance",
    # Batch annotation
    "classify_documents_batch",
    "apply_tags_batch",
    "generate_summaries_batch",
]
