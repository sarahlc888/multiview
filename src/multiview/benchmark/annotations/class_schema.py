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
    n_samples: int = 10,
    schema_hint: str | None = None,
    cache_alias: str | None = None,
) -> dict:
    """Generate a category schema from sample documents.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Description of what the criterion means
        n_samples: Number of documents to sample for schema generation
        schema_hint: Optional hint about what categories to create
        cache_alias: Optional cache alias for inference calls

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
    sample_docs_str = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(sample_docs))

    # Prepare inputs with template variables
    # Pass empty strings for optional fields - presets handle this gracefully
    inputs = {
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "schema_hint": [schema_hint or ""],
        "sample_documents": [sample_docs_str],
    }

    # Generate schema using inference
    results = run_inference(
        inputs=inputs,
        config="category_schema_generation_gemini",
        cache_alias=cache_alias,
        verbose=True,
    )

    schema = results[0]
    if schema is None:
        raise ValueError("Failed to generate category schema")

    # Handle JSON parser wrapping dict in list
    if isinstance(schema, list) and len(schema) == 1:
        schema = schema[0]

    if not isinstance(schema, dict):
        raise ValueError(f"Expected schema to be a dict, got {type(schema)}: {schema}")

    logger.info(
        f"Generated category schema with {len(schema.get('categories', []))} categories"
    )
    return schema


def classify_document(
    document: str,
    criterion: str,
    criterion_description: str,
    category_schema: dict,
    cache_alias: str | None = None,
) -> str:
    """Classify a single document into a category.

    Args:
        document: Document string
        criterion: Criterion name
        criterion_description: Criterion description
        category_schema: Category schema dict
        cache_alias: Optional cache alias for inference calls

    Returns:
        Category name as string
    """
    # Format schema for prompt
    categories = category_schema.get("categories", [])
    categories_text = "\n".join(
        [f"- {cat['name']}: {cat['description']}" for cat in categories]
    )

    # Prepare inputs
    inputs = {
        "document": [document],
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "category_schema": [categories_text],
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config="category_classify_gemini",
        cache_alias=cache_alias,
        verbose=False,
    )

    # Result is plain text (category name)
    category_name = results[0].strip() if results[0] else None

    # Match to valid category names
    if category_name:
        response_lower = category_name.lower()
        for cat in categories:
            if (
                cat["name"].lower() in response_lower
                or response_lower in cat["name"].lower()
            ):
                return cat["name"]

    # Fallback to first category
    return categories[0]["name"] if categories else "unknown"


def classify_documents_batch(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    category_schema: dict,
    cache_alias: str | None = None,
) -> list[dict]:
    """Classify multiple documents into categories.

    Args:
        documents: List of document strings
        criterion: Criterion name
        criterion_description: Criterion description
        category_schema: Category schema dict
        cache_alias: Optional cache alias for inference calls

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
    inputs = {
        "document": documents,
        "criterion": [criterion] * len(documents),
        "criterion_description": [criterion_description or ""] * len(documents),
        "category_schema": [categories_text] * len(documents),
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config="category_classify_gemini",
        cache_alias=cache_alias,
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
                matched = categories[0]["name"]

            annotations.append({"category": matched, "category_reasoning": reasoning})

    logger.info(f"Classified {len(documents)} documents into categories")
    return annotations
