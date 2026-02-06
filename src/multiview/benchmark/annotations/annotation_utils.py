"""Shared annotation utilities for LM-based annotation workflows."""

import logging

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


# Document normalization utilities
def normalize_document(doc: str | dict) -> dict:
    """Normalize a document to dict format.

    Args:
        doc: Either a string (text-only) or dict (with text and optional fields)

    Returns:
        Dict with at least 'text' field
    """
    if isinstance(doc, str):
        return {"text": doc}
    return doc


def extract_lean_document(doc: str | dict) -> dict:
    """Extract only essential fields (text, image_path) to avoid metadata bloat.

    Args:
        doc: Document (string or dict)

    Returns:
        Dict with only 'text' and optionally 'image_path'
    """
    normalized = normalize_document(doc)
    lean = {"text": normalized["text"]}

    # Include image_path if present
    if "image_path" in normalized:
        lean["image_path"] = normalized["image_path"]

    return lean


def extract_text(doc: str | dict) -> str:
    """Extract text string from document.

    Args:
        doc: Document (string or dict)

    Returns:
        Text string (may be empty if document has only images)
    """
    if isinstance(doc, str):
        return doc
    return doc.get("text", "")


def extract_image(doc: str | dict) -> str | None:
    """Extract image path/URL from document.

    Args:
        doc: Document (string or dict)

    Returns:
        Image path/URL or None
    """
    if isinstance(doc, str):
        return None
    return doc.get("image_path")


def extract_texts_and_images(
    documents: list[str | dict],
) -> tuple[list[str], list[str | None]]:
    """Extract texts and images from a list of documents.

    Args:
        documents: List of documents (strings or dicts)

    Returns:
        Tuple of (texts, images) where:
        - texts: List of text strings (with <image> markers where images exist without text)
        - images: List of image paths/URLs (None for text-only documents)
    """
    texts = []
    images = []

    for doc in documents:
        text = extract_text(doc)
        image = extract_image(doc)

        # If document has image but no text, use <image> placeholder
        if image and not text:
            text = "<image>"

        texts.append(text)
        images.append(image)

    return texts, images


def add_document_inputs_with_images(
    inputs: dict,
    documents_by_position: dict[str, list[str | dict]],
) -> None:
    """Add document inputs with proper text/image extraction and structuring.

    CRITICAL: This is the SINGLE source of truth for converting documents to inference inputs.
    All code that passes documents to LLM inference should use this function.

    Handles:
    - Extracting text and images separately
    - Adding <image> markers for image-only documents
    - Structuring images as list-of-lists for proper cache key generation
    - Ensuring image signatures are included in packed prompts

    Args:
        inputs: Dict to modify (adds document_* and optionally images keys)
        documents_by_position: Dict mapping position names to document lists
            Example: {"a": [doc1_anchor, doc2_anchor, ...],
                     "b": [doc1_positive, doc2_positive, ...],
                     "c": [doc1_negative, doc2_negative, ...]}
            Position names become "document_{position}" in inputs

    Example:
        inputs = {"criterion": ["topic"] * 5}
        add_document_inputs_with_images(inputs, {
            "a": triplet_anchors,
            "b": triplet_positives,
            "c": triplet_negatives,
        })
        # Now inputs has: document_a, document_b, document_c, images (if any docs have images)
    """
    # Extract text and images for each position
    position_texts = {}
    position_images = {}

    for position, docs in documents_by_position.items():
        texts, images = extract_texts_and_images(docs)
        position_texts[position] = texts
        position_images[position] = images

    # Add texts to inputs
    for position, texts in position_texts.items():
        inputs[f"document_{position}"] = texts

    # Collect images across all positions for each item
    n_items = len(next(iter(documents_by_position.values())))
    all_images = []

    for i in range(n_items):
        item_images = []
        # Collect images from all positions for this item
        for position in sorted(documents_by_position.keys()):  # Sort for consistency
            img = position_images[position][i]
            if img:
                item_images.append(img)
        all_images.append(item_images if item_images else None)

    # Add images if any item has images
    if any(imgs for imgs in all_images):
        inputs["images"] = all_images


def _schema_pool(
    documents: list[str | dict], schema_documents: list[str | dict] | None
) -> list[str | dict]:
    return schema_documents if schema_documents is not None else documents


def _schema_cache_prefix(cache_alias_prefix: str | None, criterion: str) -> str | None:
    return f"schema_{criterion}" if cache_alias_prefix else None


def annotate_with_known_criterion(
    documents: list[str | dict],
    document_set,
    criterion: str,
) -> list[dict]:
    """Annotate documents using known criterion extraction.

    Note: Accepts full document dicts to preserve metadata needed for criterion extraction.
    """
    annotations = []
    for doc in documents:
        # Keep full document for metadata-based extraction
        value = document_set.get_known_criterion_value(doc, criterion)
        annotations.append({"prelabel": value})

    logger.debug(f"Extracted {criterion} for {len(documents)} documents")
    return annotations


def annotate_with_precomputed(
    documents: list[str | dict],
    document_set,
    criterion: str,
) -> list[dict]:
    """Load pre-computed annotations from dataset."""
    precomputed = document_set.get_precomputed_annotations(criterion)

    annotations = []
    for doc in documents:
        # Extract text for lookup (handles both str and dict)
        doc_text = extract_text(doc)

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
    documents: list[str | dict],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    category_schema_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str | dict] | None = None,
    category_schema_preset: str | None = None,
    category_classify_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with category classification only."""
    logger.info(
        f"Starting category annotation for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    # Set defaults for presets
    category_schema_preset = (
        category_schema_preset or "category_schema_generation_gemini"
    )
    category_classify_preset = category_classify_preset or "category_classify_gemini"

    schema_pool = _schema_pool(documents, schema_documents)
    schema_cache_prefix = _schema_cache_prefix(cache_alias_prefix, criterion)

    # Extract text strings and images
    schema_pool_texts = [extract_text(doc) for doc in schema_pool]
    schema_pool_images = [extract_image(doc) for doc in schema_pool]
    document_texts = [extract_text(doc) for doc in documents]
    document_images = [extract_image(doc) for doc in documents]

    # Only pass images if at least one is present
    has_schema_images = any(img is not None for img in schema_pool_images)
    has_doc_images = any(img is not None for img in document_images)

    logger.info("Step 1/2: Generating category schema...")
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

    logger.info("Step 2/2: Applying category schema to all documents...")
    cat_alias = f"{cache_alias_prefix}_category" if cache_alias_prefix else None

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
    documents: list[str | dict],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    tag_schema_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str | dict] | None = None,
    tag_schema_preset: str | None = None,
    spurious_tag_schema_preset: str | None = None,
    tag_apply_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with criterion-relevant and spurious tags."""
    logger.info(
        f"Starting tag annotation for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    # Set defaults for presets
    tag_schema_preset = tag_schema_preset or "tag_schema_generation_gemini"
    spurious_tag_schema_preset = (
        spurious_tag_schema_preset or "spurious_tag_schema_generation_gemini"
    )
    tag_apply_preset = tag_apply_preset or "tag_apply_gemini"

    schema_pool = _schema_pool(documents, schema_documents)
    schema_cache_prefix = _schema_cache_prefix(cache_alias_prefix, criterion)

    # Extract text strings and images
    schema_pool_texts = [extract_text(doc) for doc in schema_pool]
    schema_pool_images = [extract_image(doc) for doc in schema_pool]
    document_texts = [extract_text(doc) for doc in documents]
    document_images = [extract_image(doc) for doc in documents]

    # Only pass images if at least one is present
    has_schema_images = any(img is not None for img in schema_pool_images)
    has_doc_images = any(img is not None for img in document_images)

    logger.info(f"Step 1/2: Generating tag schemas with {tag_schema_preset=}...")
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

    logger.info("Step 2/2: Applying tag schemas to all documents...")
    tag_alias = f"{cache_alias_prefix}_tags" if cache_alias_prefix else None
    spur_alias = f"{cache_alias_prefix}_spurious" if cache_alias_prefix else None

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


def _summary_log_messages(mode: str) -> dict[str, str]:
    if mode == "sentence":
        return {
            "start": "Starting sentence-mode summary annotation",
            "step1": "Step 1/2: Generating summary guidance (sentence mode)...",
            "step2": "Step 2/2: Applying summary guidance to all documents (sentence mode)...",
            "done": "Completed sentence-mode summary annotation",
        }
    return {
        "start": "Starting summary annotation",
        "step1": "Step 1/2: Generating summary guidance...",
        "step2": "Step 2/2: Applying summary guidance to all documents...",
        "done": "Completed summary annotation",
    }


def _annotate_with_lm_summary(
    documents: list[str | dict],
    criterion: str,
    criterion_description: str | None,
    document_type: str | None,
    n_schema_samples: int,
    summary_hint: str | None,
    cache_alias_prefix: str | None,
    run_name: str | None,
    schema_documents: list[str | dict] | None,
    summary_guidance_preset: str | None,
    summary_generate_preset: str | None,
    *,
    mode: str,
    guidance_default: str,
    generate_default: str,
) -> list[dict]:
    messages = _summary_log_messages(mode)
    logger.info(
        f"{messages['start']} for {len(documents)} documents "
        f"with criterion '{criterion}'"
    )

    schema_pool = _schema_pool(documents, schema_documents)
    schema_cache_prefix = _schema_cache_prefix(cache_alias_prefix, criterion)

    # Extract text strings and images
    schema_pool_texts = [extract_text(doc) for doc in schema_pool]
    schema_pool_images = [extract_image(doc) for doc in schema_pool]
    document_texts = [extract_text(doc) for doc in documents]
    document_images = [extract_image(doc) for doc in documents]

    # Only pass images if at least one is present
    has_schema_images = any(img is not None for img in schema_pool_images)
    has_doc_images = any(img is not None for img in document_images)

    logger.info(messages["step1"])
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
        guidance_preset=summary_guidance_preset or guidance_default,
        images=schema_pool_images if has_schema_images else None,
    )

    logger.info(messages["step2"])
    summ_alias = f"{cache_alias_prefix}_summary" if cache_alias_prefix else None

    summary_annotations = generate_summaries_batch(
        document_texts,
        criterion,
        criterion_description or "",
        summary_guidance,
        cache_alias=summ_alias,
        run_name=run_name,
        generate_preset=summary_generate_preset or generate_default,
        images=document_images if has_doc_images else None,
    )

    rich_annotations = []
    for i in range(len(documents)):
        annotation = {
            "summary": summary_annotations[i].get("summary", {}),
            "summary_guidance": summary_guidance,
        }
        rich_annotations.append(annotation)

    logger.info(f"{messages['done']} for {len(documents)} documents")
    return rich_annotations


def annotate_with_lm_summary_dict(
    documents: list[str | dict],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    summary_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str | dict] | None = None,
    summary_guidance_preset: str | None = None,
    summary_generate_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with structured/dictionary summaries."""
    return _annotate_with_lm_summary(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        document_type=document_type,
        n_schema_samples=n_schema_samples,
        summary_hint=summary_hint,
        cache_alias_prefix=cache_alias_prefix,
        run_name=run_name,
        schema_documents=schema_documents,
        summary_guidance_preset=summary_guidance_preset,
        summary_generate_preset=summary_generate_preset,
        mode="dict",
        guidance_default="summary_guidance_generation_dict_gemini",
        generate_default="summary_generate_dict_gemini",
    )


def annotate_with_lm_summary_sentence(
    documents: list[str | dict],
    criterion: str,
    criterion_description: str | None = None,
    document_type: str | None = None,
    n_schema_samples: int = 10,
    summary_hint: str | None = None,
    cache_alias_prefix: str | None = None,
    run_name: str | None = None,
    schema_documents: list[str | dict] | None = None,
    summary_guidance_preset: str | None = None,
    summary_generate_preset: str | None = None,
) -> list[dict]:
    """Annotate documents with one-sentence prose summaries."""
    return _annotate_with_lm_summary(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        document_type=document_type,
        n_schema_samples=n_schema_samples,
        summary_hint=summary_hint,
        cache_alias_prefix=cache_alias_prefix,
        run_name=run_name,
        schema_documents=schema_documents,
        summary_guidance_preset=summary_guidance_preset,
        summary_generate_preset=summary_generate_preset,
        mode="sentence",
        guidance_default="summary_guidance_generation_sentence_gemini",
        generate_default="summary_generate_sentence_gemini",
    )
