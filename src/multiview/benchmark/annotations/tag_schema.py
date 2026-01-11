"""Tag schema generation and application.

This module handles multi-label binary tag annotation (similar to lm_tags.py):
- Generate tag schemas (criterion-relevant and spurious)
- Apply tags to documents
"""

from __future__ import annotations

import logging

from multiview.inference.inference import run_inference
from multiview.utils.sampling_utils import deterministic_sample

logger = logging.getLogger(__name__)


def generate_tag_schema(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    n_samples: int = 10,
    schema_hint: str | None = None,
    is_spurious: bool = False,
    cache_alias: str | None = None,
) -> dict:
    """Generate a tag schema from sample documents.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name
        criterion_description: Description of what the criterion means
        n_samples: Number of documents to sample for schema generation
        schema_hint: Optional hint about what tags to create
        is_spurious: If True, generate spurious tags (surface-level properties)
        cache_alias: Optional cache alias for inference calls

    Returns:
        Tag schema dict with structure:
        {
            "tags": [
                {"name": "...", "description": "..."},
                ...
            ]
        }
    """
    # Sample documents deterministically based on criterion (and spurious flag for uniqueness)
    seed_base = f"{criterion}_spurious" if is_spurious else criterion
    sample_docs = deterministic_sample(documents, n_samples, seed_base)
    sample_docs_str = "\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(sample_docs)
    )

    # Format schema_hint with heading if provided
    schema_hint_formatted = (
        f"\nTAG SCHEMA HINT (optional):\n{schema_hint}\n" if schema_hint else ""
    )

    # Prepare inputs with template variables
    inputs = {
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "schema_hint": [schema_hint_formatted],
        "sample_documents": [sample_docs_str],
        "n_samples": [str(len(sample_docs))],
    }

    # Use preset based on tag type
    preset_name = (
        "spurious_tag_schema_generation_gemini"
        if is_spurious
        else "tag_schema_generation_gemini"
    )

    # Generate schema using inference
    results = run_inference(
        inputs=inputs,
        config=preset_name,
        cache_alias=cache_alias,
        verbose=True,
    )

    schema = results[0]

    if schema is None:
        raise ValueError(
            f"Failed to generate {'spurious ' if is_spurious else ''}tag schema"
        )

    logger.info(
        f"Generated {'spurious ' if is_spurious else ''}tag schema with {len(schema.get('tags', []))} tags"
    )
    return schema


def generate_spurious_tag_schema(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    n_samples: int = 10,
    cache_alias: str | None = None,
) -> dict:
    """Generate spurious tag schema (surface-level properties).

    Convenience wrapper around generate_tag_schema with is_spurious=True.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name
        criterion_description: Description of what the criterion means
        n_samples: Number of documents to sample for schema generation
        cache_alias: Optional cache alias for inference calls

    Returns:
        Spurious tag schema dict
    """
    return generate_tag_schema(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        n_samples=n_samples,
        schema_hint="Focus on surface-level properties independent of the criterion",
        is_spurious=True,
        cache_alias=cache_alias,
    )


def apply_tags_batch(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    tag_schema: dict,
    cache_alias: str | None = None,
) -> list[dict]:
    """Apply binary tags to multiple documents.

    Args:
        documents: List of document strings
        criterion: Criterion name
        criterion_description: Criterion description
        tag_schema: Tag schema dict
        cache_alias: Optional cache alias for inference calls

    Returns:
        List of annotation dicts:
        [
            {"tags": {"tag1": true, "tag2": false, ...}},
            ...
        ]
    """
    # Format schema for prompt
    tags = tag_schema.get("tags", [])
    tags_text = "\n".join([f"- {tag['name']}: {tag['description']}" for tag in tags])

    # Prepare inputs
    inputs = {
        "document": documents,
        "criterion": [criterion] * len(documents),
        "criterion_description": [criterion_description or ""] * len(documents),
        "tag_schema": [tags_text] * len(documents),
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config="tag_apply_gemini",
        cache_alias=cache_alias,
        verbose=False,
    )

    # Extract annotations (convert 0/1 to boolean)
    annotations = []
    for result in results:
        if result is None:
            annotations.append({"tags": {}, "tag_reasoning": None})
        else:
            # Result now has reasoning and tags fields
            if isinstance(result, dict) and "tags" in result:
                tags_dict = result.get("tags", {})
                reasoning = result.get("reasoning", "")
            else:
                # Fallback for old format (plain dict of tags)
                tags_dict = result if isinstance(result, dict) else {}
                reasoning = ""

            tags_bool = {k: bool(v) for k, v in tags_dict.items()}
            annotations.append({"tags": tags_bool, "tag_reasoning": reasoning})

    logger.info(f"Applied tags to {len(documents)} documents")
    return annotations
