"""In-one-word evaluation method using hidden state extraction.

This module implements evaluation using hidden states from causal language models
after prompting them to categorize documents "in one word" based on category schemas
from annotations. The hidden states serve as embeddings for triplet evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def get_category_list(annotation: dict) -> list[str]:
    """Extract list of category names from annotation's category schema.

    Args:
        annotation: Annotation dict with category_schema field

    Returns:
        List of category names (e.g., ["addition", "multiplication", ...])

    Raises:
        ValueError: If category_schema is missing or malformed
    """
    if "category_schema" not in annotation:
        raise ValueError("Annotation missing 'category_schema' field")

    schema = annotation["category_schema"]
    if not schema:
        raise ValueError("Annotation has empty category_schema")

    if "categories" not in schema:
        raise ValueError("category_schema missing 'categories' field")

    categories = schema["categories"]
    if not isinstance(categories, list):
        raise ValueError(
            f"category_schema['categories'] must be list, got {type(categories)}"
        )

    # Extract category names
    category_names = []
    for cat in categories:
        if isinstance(cat, dict) and "name" in cat:
            category_names.append(cat["name"])
        else:
            raise ValueError(f"Malformed category entry: {cat}")

    if not category_names:
        raise ValueError("category_schema has no categories")

    return category_names


def validate_annotations(annotations: list[dict]) -> None:
    """Validate that annotations have required category schema information.

    Args:
        annotations: List of annotation dicts

    Raises:
        ValueError: If annotations are missing or lack category schemas
    """
    if not annotations:
        raise ValueError(
            "in_one_word evaluation requires annotations with category schemas"
        )

    if not isinstance(annotations, list):
        raise ValueError(f"annotations must be a list, got {type(annotations)}")

    # Check first annotation has category schema
    try:
        get_category_list(annotations[0])
    except ValueError as e:
        raise ValueError(
            f"in_one_word evaluation requires annotations with category schemas: {e}"
        ) from e


def format_category_context(categories: list[str]) -> str:
    """Format category list for prompt context.

    Args:
        categories: List of category names

    Returns:
        Formatted string with categories for prompt
    """
    category_str = ", ".join(categories)
    return (
        f"Categories: {category_str}\n" f"Question: Categorize this text in one word."
    )


def evaluate_with_in_one_word(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    category_context: str | None = None,
    annotations: list[dict] | None = None,
    preset: str = "inoneword_hf_qwen3_8b",
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using in-one-word hidden state extraction.

    This method:
    1. Gets category context from parameter or annotations
    2. Builds prompts with category context + document text
    3. Runs inference to get hidden state embeddings
    4. Evaluates triplets using cosine similarity

    Args:
        documents: List of document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        category_context: Freetext category context string for prompts.
                         If not provided, will be extracted/formatted from annotations.
        annotations: Optional list of annotation dicts. If category_context not provided,
                    will look for 'category_context' field or fall back to extracting
                    from 'category_schema' field for backwards compatibility.
        preset: Inference preset to use (default: "inoneword_hf_qwen3_8b")
        cache_alias: Optional cache identifier
        run_name: Optional experiment/run name for cache organization
        preset_overrides: Optional preset configuration overrides

    Returns:
        Dict with evaluation metrics:
        - positive_scores: Cosine similarities for (anchor, positive) pairs
        - negative_scores: Cosine similarities for (anchor, negative) pairs
        - outcomes: List of outcomes (1=correct, -1=incorrect, 0=tie)
        - avg_positive_score: Mean positive similarity
        - avg_negative_score: Mean negative similarity
        - triplet_logs: Detailed per-triplet information

    Raises:
        ValueError: If neither category_context nor annotations are provided
    """
    # Determine category_context source: parameter or annotations
    if category_context is None:
        # Must extract from annotations
        if annotations is None:
            raise ValueError(
                "Either 'category_context' parameter or 'annotations' must be provided"
            )

        # Try to get category_context directly from annotation
        if "category_context" in annotations[0]:
            category_context = annotations[0]["category_context"]
            logger.info("Using category_context from annotations")
        else:
            # Fall back to extracting from category_schema for backwards compatibility
            logger.info(
                "No 'category_context' in annotations, falling back to category_schema"
            )
            validate_annotations(annotations)
            categories = get_category_list(annotations[0])
            category_context = format_category_context(categories)
            logger.info(f"Formatted category context from {len(categories)} categories")
    else:
        logger.info("Using category_context from parameter")

    if not triplet_ids:
        logger.warning("No triplets provided for evaluation")
        return {
            "positive_scores": [],
            "negative_scores": [],
            "outcomes": [],
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
            "triplet_logs": [],
        }

    logger.info(f"Evaluating {len(triplet_ids)} triplets with in-one-word method")
    logger.info(f"Using preset: {preset}")
    logger.debug(f"Category context:\n{category_context}")

    # Build prompts: category context + document text
    prompts = [f"{category_context}\nText: {doc}" for doc in documents]

    logger.info(f"Computing hidden state embeddings for {len(documents)} documents")

    # Run inference to get hidden state embeddings
    inference_kwargs = {"verbose": False}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)

    inputs = {"document": prompts}

    embeddings = run_inference(
        inputs=inputs,
        config=preset,
        cache_alias=cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )

    # Evaluate triplets using cosine similarity
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    outcomes: list[int] = []
    triplet_logs: list[dict[str, Any]] = []

    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        anchor_emb = embeddings[anchor_id]
        positive_emb = embeddings[positive_id]
        negative_emb = embeddings[negative_id]

        pos_score = cosine_similarity(anchor_emb, positive_emb)
        neg_score = cosine_similarity(anchor_emb, negative_emb)

        positive_scores.append(pos_score)
        negative_scores.append(neg_score)

        # Determine outcome
        if pos_score > neg_score:
            outcome = 1  # Correct
        elif neg_score > pos_score:
            outcome = -1  # Incorrect
        else:
            outcome = 0  # Tie

        outcomes.append(outcome)

        # Log detailed triplet information
        triplet_logs.append(
            {
                "triplet_idx": i,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "positive_score": pos_score,
                "negative_score": neg_score,
                "outcome": outcome,
                "correct": outcome == 1,
                "is_tie": outcome == 0,
            }
        )

    # Compute aggregate statistics
    avg_pos = float(np.mean(positive_scores)) if positive_scores else 0.0
    avg_neg = float(np.mean(negative_scores)) if negative_scores else 0.0

    logger.info(f"Average positive score: {avg_pos:.4f}")
    logger.info(f"Average negative score: {avg_neg:.4f}")
    logger.info(
        f"Accuracy: {sum(1 for o in outcomes if o == 1)}/{len(outcomes)} "
        f"({sum(1 for o in outcomes if o == 1) / len(outcomes):.2%})"
    )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "outcomes": outcomes,
        "avg_positive_score": avg_pos,
        "avg_negative_score": avg_neg,
        "triplet_logs": triplet_logs,
    }


__all__ = ["evaluate_with_in_one_word"]
