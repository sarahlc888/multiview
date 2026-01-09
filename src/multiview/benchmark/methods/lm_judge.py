"""LM judge evaluation method.

This module implements evaluation methods that use language models to judge
similarity between documents in triplets.
"""

import logging
from typing import Any

from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def evaluate_with_lm_judge_triplet(
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    lm_judge_preset: str = "lmjudge_triplet_plaintext_binaryhard_gemini",
    cache_alias: str | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using an LM judge.

    This method uses a language model to judge which of two documents (positive or negative)
    is more similar to an anchor document based on a similarity criterion.

    Args:
        triplets: List of triplet dicts with keys:
            - "anchor": anchor document text
            - "positive": positive document text
            - "negative": negative document text
        criterion: Similarity criterion name
        criterion_description: Detailed description of the criterion (optional)
        lm_judge_preset: Preset name for LM judge (default: simple triplet comparison)
        cache_alias: Cache alias for inference caching

    Returns:
        Dict with evaluation metrics:
            - accuracy: Float between 0 and 1
            - n_correct: Number of triplets where positive ranked higher than negative
            - n_incorrect: Number of triplets where negative ranked higher than positive
            - n_ties: Number of triplets judged as ties
            - n_total: Total number of triplets evaluated

    Example:
        >>> triplets = [
        ...     {"anchor": "doc1", "positive": "doc2", "negative": "doc3"},
        ...     {"anchor": "doc4", "positive": "doc5", "negative": "doc6"},
        ... ]
        >>> results = evaluate_with_lm_judge_triplet(
        ...     triplets,
        ...     criterion="mathematical_operations",
        ...     criterion_description="What math operations are used in the problem"
        ... )
        >>> print(f"Accuracy: {results['accuracy']:.2f}")
        Accuracy: 0.85
    """
    if not triplets:
        logger.warning("No triplets provided for evaluation")
        return {
            "accuracy": 0.0,
            "n_correct": 0,
            "n_incorrect": 0,
            "n_ties": 0,
            "n_total": 0,
        }

    logger.info(f"Evaluating {len(triplets)} triplets with LM judge")
    logger.info(f"Using preset: {lm_judge_preset}")

    # Prepare inputs for batch inference
    # For the triplet judge preset, we need:
    # - similarity_criteria: the criterion description
    # - document_a: anchor
    # - document_b: positive
    # - document_c: negative
    inputs = {
        "similarity_criteria": [criterion_description or criterion] * len(triplets),
        "document_a": [t["anchor"] for t in triplets],
        "document_b": [t["positive"] for t in triplets],
        "document_c": [t["negative"] for t in triplets],
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        verbose=False,
    )

    # Count results
    # The triplet judge returns:
    #   1 if (b) is more similar to (a) - i.e., positive wins (correct)
    #  -1 if (c) is more similar to (a) - i.e., negative wins (incorrect)
    #   0 if tie/draw
    n_correct = sum(1 for r in results if r == 1)
    n_incorrect = sum(1 for r in results if r == -1)
    n_ties = sum(1 for r in results if r == 0)
    n_total = len(results)

    # Calculate accuracy (excluding ties)
    n_judged = n_correct + n_incorrect
    accuracy = n_correct / n_judged if n_judged > 0 else 0.0

    logger.info(f"Evaluation complete: {n_correct}/{n_total} correct ({accuracy:.2%})")
    if n_ties > 0:
        logger.info(f"Note: {n_ties} triplets judged as ties (excluded from accuracy)")

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_ties": n_ties,
        "n_total": n_total,
    }
