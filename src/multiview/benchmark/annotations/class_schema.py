"""Category schema generation and classification.

This module handles discrete category annotation (similar to lm_discrete.py):
- Generate category schema from sample documents
- Classify documents into categories
"""

from __future__ import annotations

import logging
import random

from multiview.inference.inference import run_inference

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
    # Sample documents
    sample_docs = random.sample(documents, min(n_samples, len(documents)))
    sample_docs_str = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(sample_docs))

    # Build prompt conditionally
    prompt_parts = [
        "You have a set of unlabeled documents and a criterion of interest. "
        "Your task is to design a category schema that captures how the documents vary with respect to the criterion.",
        "",
        f"CRITERION: {criterion}",
    ]

    if criterion_description:
        prompt_parts.extend(
            [
                "",
                "CRITERION DESCRIPTION:",
                criterion_description,
            ]
        )

    if schema_hint:
        prompt_parts.extend(
            [
                "",
                "SCHEMA HINT:",
                schema_hint,
            ]
        )

    prompt_parts.extend(
        [
            "",
            "SAMPLE DOCUMENTS:",
            sample_docs_str,
            "",
            "Think about what kind of candidate schemas are possible.",
            "- The ideal candidate schema partitions the output space into discrete categories.",
            "- If there is a way to enumerate a closed set of categories, that would be best, but if not, it's OK to include an 'other' category.",
            "- Try to choose a number of categories that reflects the range of variation within the sample documents.",
            "",
            "Choose the single best schema strategy and return it in valid JSON with reasoning:",
            "{",
            '  "reasoning": "Explain your schema choice: what alternatives you considered and why this schema best captures variation along the criteria",',
            '  "categories": [{"name": "...", "description": "..."}, ...]',
            "}",
        ]
    )

    prompt = "\n".join(prompt_parts)

    # Prepare inputs with pre-built prompt
    inputs = {
        "prompt": [prompt],
    }

    # Generate schema using inference with a custom config
    from multiview.inference.presets import InferenceConfig

    custom_config = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="{prompt}",
        parser="json",
        temperature=0.0,
        max_tokens=8192,
    )

    results = run_inference(
        inputs=inputs,
        config=custom_config,
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
