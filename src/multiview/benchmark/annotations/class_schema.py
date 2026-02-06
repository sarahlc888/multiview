"""Category schema generation and classification.

This module handles discrete category annotation (similar to lm_discrete.py):
- Generate category schema from sample documents
- Classify documents into categories
"""

from __future__ import annotations

import logging

from multiview.inference.inference import run_inference
from multiview.utils.sampling_utils import deterministic_sample

logger = logging.getLogger(__name__)


def generate_category_schema(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    document_type: str,
    n_samples: int = 10,
    schema_hint: str | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
    config: str = "category_schema_generation_gemini",
    images: list[str | None] | None = None,
) -> dict:
    """Generate a category schema from sample documents.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Description of what the criterion means
        document_type: Type of documents being classified (e.g., "math word problem", "story", "sentence")
        n_samples: Number of documents to sample for schema generation
        schema_hint: Optional hint about what categories to create
        cache_alias: Optional cache alias for inference calls
        run_name: Optional experiment/run name for cache organization
        config: Inference config name to use (default: "category_schema_generation_gemini")
        images: Optional list of image paths/URLs corresponding to documents

    Returns:
        Category schema dict with structure:
        {
            "categories": [
                {"name": "...", "description": "..."},
                ...
            ]
        }
    """
    # Sample documents deterministically based on criterion
    sample_docs = deterministic_sample(documents, n_samples, criterion)

    # Sample images if provided (using same indices)
    sample_images = None
    if images is not None:
        # Get the indices used for sampling
        indices = deterministic_sample(
            list(range(len(documents))), n_samples, criterion
        )
        sample_images = [images[i] for i in indices]

    # Format sample documents with <image> markers if images are provided
    if sample_images is not None:
        sample_docs_str = "\n\n".join(
            f"[Document {i+1}]\n<image>" for i in range(len(sample_docs))
        )
    else:
        sample_docs_str = "\n\n".join(
            f"[Document {i+1}]\n{doc}" for i, doc in enumerate(sample_docs)
        )

    # Format schema_hint with heading if provided
    schema_hint_formatted = (
        f"\nSCHEMA HINT (optional):\n{schema_hint}\n" if schema_hint else ""
    )

    # Prepare inputs with template variables
    if not document_type:
        raise ValueError("document_type is required for category schema generation")

    inputs = {
        "document_type": [document_type],
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "schema_hint": [schema_hint_formatted],
        "sample_documents": [sample_docs_str],
    }

    # Add images if available
    # Wrap in list for multi-image single prompt (needed for proper interleaving)
    if sample_images is not None:
        inputs["images"] = [sample_images]

    # Generate schema using inference
    results = run_inference(
        inputs=inputs,
        config=config,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=True,
    )

    schema = results[0]
    if schema is None:
        raise ValueError("Failed to generate category schema")

    # Some JSON parsers return wrapped outputs like: [[{...schema...}]]
    # Unwrap common single-element list wrappers.
    while isinstance(schema, list) and len(schema) == 1:
        schema = schema[0]

    if not isinstance(schema, dict):
        raise ValueError(f"Expected schema to be a dict, got {type(schema)}: {schema}")

    logger.info(
        f"Generated category schema with {len(schema.get('categories', []))} categories"
    )
    return schema


def classify_documents_batch(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    category_schema: dict,
    document_type: str,
    cache_alias: str | None = None,
    run_name: str | None = None,
    config: str = "category_classify_gemini",
    images: list[str | None] | None = None,
) -> list[dict]:
    """Classify multiple documents into categories.

    Args:
        documents: List of document strings
        criterion: Criterion name
        criterion_description: Criterion description
        category_schema: Category schema dict
        document_type: Type of documents being classified (e.g., "math word problem", "story", "sentence")
        cache_alias: Optional cache alias for inference calls
        run_name: Optional experiment/run name for cache organization
        config: Inference config name to use (default: "category_classify_gemini")
        images: Optional list of image paths/URLs corresponding to documents

    Returns:
        List of annotation dicts:
        [
            {"category": "category_name"},
            ...
        ]
    """
    # Format schema for prompt
    categories = category_schema.get("categories", [])
    categories_text = "\n".join(
        [f"- {cat['name']}: {cat['description']}" for cat in categories]
    )

    # Prepare inputs
    if not document_type:
        raise ValueError("document_type is required for document classification")

    inputs = {
        "document": documents,
        "document_type": [document_type] * len(documents),
        "criterion": [criterion] * len(documents),
        "criterion_description": [criterion_description or ""] * len(documents),
        "category_schema": [categories_text] * len(documents),
    }

    # Add images if available
    if images is not None:
        inputs["images"] = images

    # Run inference
    results = run_inference(
        inputs=inputs,
        config=config,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )

    # Extract annotations
    annotations = []
    for result in results:
        if result is None:
            annotations.append({"category": None, "category_reasoning": None})
        else:
            # Result is now JSON with reasoning and category
            if isinstance(result, dict):
                category_name = result.get("category")
                reasoning = result.get("reasoning", "")
            else:
                # Fallback for old format (plain text)
                logger.warning(
                    f"⚠️  FALLBACK TRIGGERED: Received non-dict result from LM (type={type(result)}). "
                    f"Expected dict with 'category' and 'reasoning' keys. Treating as plain text category name."
                )
                category_name = result.strip() if isinstance(result, str) else None
                reasoning = ""

            # Match to valid category names
            matched = None
            if category_name:
                response_lower = category_name.lower()
                for cat in categories:
                    if (
                        cat["name"].lower() in response_lower
                        or response_lower in cat["name"].lower()
                    ):
                        matched = cat["name"]
                        break

            # Fallback to first category
            if matched is None and categories:
                logger.error(
                    f"⚠️  FALLBACK TRIGGERED: Could not match category '{category_name}' to any valid category. "
                    f"Defaulting to '{categories[0]['name']}'. Valid categories: {[c['name'] for c in categories]}"
                )
                matched = categories[0]["name"]

            annotations.append({"category": matched, "category_reasoning": reasoning})

    logger.info(f"Classified {len(documents)} documents into categories")
    return annotations
