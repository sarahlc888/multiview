"""Utilities for generating text variations (queries, summaries) from documents."""

from __future__ import annotations

import logging

from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def generate_text_variations_from_documents(
    documents: list[str],
    criterion: str,
    criterion_description: str | None,
    num_variations: int,
    generation_preset: str,
    cache_alias: str | None,
    run_name: str | None,
    cache_suffix: str = "generation",
) -> list[str]:
    """Generate k text variations (queries/summaries) FOR EACH document.

    This is a general-purpose function for generating multiple text variations
    from documents based on a criterion. Can be used for:
    - Query generation (for retrieval evaluation)
    - Summary generation (for multisummary evaluation)
    - Any other criterion-focused text generation task

    Args:
        documents: List of documents to generate variations from
        criterion: Criterion name
        criterion_description: Optional criterion description
        num_variations: Number of variations to generate per document (k)
        generation_preset: Inference preset for generation (e.g., "document_to_summaries_gemini")
        cache_alias: Optional cache identifier
        run_name: Optional run name
        cache_suffix: Suffix for cache alias (default: "generation")

    Returns:
        List of all text variation strings (len = num_documents × k)
    """
    # Prepare inputs for batch inference
    # Each document gets k variations generated
    inputs = {
        "criterion": [criterion] * len(documents),
        "criterion_description": [criterion_description or ""] * len(documents),
        "document": documents,
        "num_expansions": [num_variations] * len(documents),
    }

    # Use cache alias with suffix
    generation_cache_alias = f"{cache_alias}_{cache_suffix}" if cache_alias else None

    logger.info(
        f"Generating {num_variations} variations for each of {len(documents)} documents "
        f"with preset: {generation_preset}"
    )
    if generation_cache_alias:
        logger.info(f"Using cache alias: {generation_cache_alias}")

    # Run inference to generate variations for all documents
    results = run_inference(
        inputs=inputs,
        config=generation_preset,
        cache_alias=generation_cache_alias,
        run_name=run_name,
        verbose=False,
    )

    # Collect all variations into a flat list
    all_variations = []
    for i, result in enumerate(results):
        # Result should be a list of strings (from json parser)
        if isinstance(result, list) and all(isinstance(v, str) for v in result):
            variations = result
        else:
            logger.warning(
                f"Unexpected generation result format for document {i}: {result}. "
                f"Expected list of strings. Using criterion as fallback."
            )
            variations = [criterion] * num_variations

        # Validate we got the right number of variations
        if len(variations) != num_variations:
            logger.warning(
                f"Expected {num_variations} variations for document {i}, got {len(variations)}. "
                f"Adjusting to match expected count."
            )
            if len(variations) < num_variations:
                # Pad with criterion
                variations.extend([criterion] * (num_variations - len(variations)))
            else:
                # Truncate
                variations = variations[:num_variations]

        # Add to flat list
        all_variations.extend(variations)

    logger.info(f"Total variations generated: {len(all_variations)}")
    return all_variations
