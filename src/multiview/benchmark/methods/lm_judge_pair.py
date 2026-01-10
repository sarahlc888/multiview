"""Pairwise LM judge evaluation method.

This module implements evaluation methods that use language models to judge
pairwise similarity between documents, then compute triplet accuracy.
"""

import logging
from typing import Any

from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def evaluate_with_lm_judge_pair(
    triplets: list[dict],
    criterion: str,
    criterion_description: str | None = None,
    lm_judge_preset: str = "lmjudge_pair_plaintext_likerthard_gemini",
    cache_alias: str | None = None,
    annotations: list[dict] | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using pairwise LM judge scoring.

    This method uses a language model to score pairwise similarity between documents,
    then evaluates triplets by comparing the similarity scores of anchor-positive
    vs anchor-negative pairs.

    Args:
        triplets: List of triplet dicts with keys:
            - "anchor": anchor document text (or document dict)
            - "positive": positive document text (or document dict)
            - "negative": negative document text (or document dict)
        criterion: Similarity criterion name
        criterion_description: Detailed description of the criterion (optional)
        lm_judge_preset: Preset name for pairwise LM judge (default: Likert scale)
        cache_alias: Cache alias for inference caching
        annotations: List of annotation dicts, one per document, with at least "summary" key (optional).
            If provided, will use annotation-aware preset and include summaries in the prompt.

    Returns:
        Dict with evaluation metrics:
            - accuracy: Float between 0 and 1
            - n_correct: Number of triplets where positive scored higher than negative
            - n_incorrect: Number of triplets where negative scored higher than positive
            - n_ties: Number of triplets with identical scores
            - n_total: Total number of triplets evaluated
            - avg_positive_score: Average similarity score for anchor-positive pairs
            - avg_negative_score: Average similarity score for anchor-negative pairs

    Example:
        >>> triplets = [
        ...     {"anchor": "doc1", "positive": "doc2", "negative": "doc3"},
        ...     {"anchor": "doc4", "positive": "doc5", "negative": "doc6"},
        ... ]
        >>> results = evaluate_with_lm_judge_pair(
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
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
        }

    logger.info(f"Evaluating {len(triplets)} triplets with pairwise LM judge")
    logger.info(f"Using preset: {lm_judge_preset}")

    # Check if we have annotations
    has_annotations = annotations is not None and len(annotations) > 0

    criterion_text = criterion_description or criterion

    # Helper function to get annotation summary
    def get_annotation_summary(triplet_key: str, triplet_idx: int) -> str:
        """Get annotation summary for a document in the triplet."""
        # First try to get from triplet itself (if pre-attached)
        annotation_key = f"{triplet_key}_annotation"
        if annotation_key in triplets[triplet_idx]:
            ann = triplets[triplet_idx][annotation_key]
            return ann.get("summary", "") if isinstance(ann, dict) else str(ann)

        # Otherwise try to map using document indices
        id_key = f"{triplet_key}_id"
        if id_key in triplets[triplet_idx] and annotations:
            doc_id = triplets[triplet_idx][id_key]
            if 0 <= doc_id < len(annotations):
                ann = annotations[doc_id]
                return ann.get("summary", "") if isinstance(ann, dict) else ""

        return ""

    # Prepare inputs for batch inference - score anchor-positive pairs
    positive_pairs_inputs = {
        "similarity_criteria": [criterion_text] * len(triplets),
        "document_a": [t["anchor"] for t in triplets],
        "document_b": [t["positive"] for t in triplets],
    }

    # Add annotations if provided
    if has_annotations:
        logger.info("Using annotations in evaluation")
        positive_pairs_inputs["annotation_a"] = [
            get_annotation_summary("anchor", i) for i in range(len(triplets))
        ]
        positive_pairs_inputs["annotation_b"] = [
            get_annotation_summary("positive", i) for i in range(len(triplets))
        ]

    # Score anchor-positive pairs
    logger.debug("Scoring anchor-positive pairs")
    positive_scores = run_inference(
        inputs=positive_pairs_inputs,
        config=lm_judge_preset,
        cache_alias=f"{cache_alias}_positive" if cache_alias else None,
        verbose=False,
    )

    # Prepare inputs for batch inference - score anchor-negative pairs
    negative_pairs_inputs = {
        "similarity_criteria": [criterion_text] * len(triplets),
        "document_a": [t["anchor"] for t in triplets],
        "document_b": [t["negative"] for t in triplets],
    }

    # Add annotations if provided
    if has_annotations:
        negative_pairs_inputs["annotation_a"] = [
            get_annotation_summary("anchor", i) for i in range(len(triplets))
        ]
        negative_pairs_inputs["annotation_b"] = [
            get_annotation_summary("negative", i) for i in range(len(triplets))
        ]

    # Score anchor-negative pairs
    logger.debug("Scoring anchor-negative pairs")
    negative_scores = run_inference(
        inputs=negative_pairs_inputs,
        config=lm_judge_preset,
        cache_alias=f"{cache_alias}_negative" if cache_alias else None,
        verbose=False,
    )

    # Count results by comparing scores
    n_correct = 0
    n_incorrect = 0
    n_ties = 0

    for pos_score, neg_score in zip(positive_scores, negative_scores, strict=False):
        if pos_score > neg_score:
            n_correct += 1
        elif neg_score > pos_score:
            n_incorrect += 1
        else:
            n_ties += 1

    n_total = len(triplets)

    # Calculate accuracy (excluding ties)
    n_judged = n_correct + n_incorrect
    accuracy = n_correct / n_judged if n_judged > 0 else 0.0

    # Calculate average scores
    avg_positive_score = (
        sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    )
    avg_negative_score = (
        sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    )

    logger.info(f"Evaluation complete: {n_correct}/{n_total} correct ({accuracy:.2%})")
    logger.info(
        f"Average positive score: {avg_positive_score:.3f}, Average negative score: {avg_negative_score:.3f}"
    )
    if n_ties > 0:
        logger.info(
            f"Note: {n_ties} triplets had identical scores (excluded from accuracy)"
        )

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_ties": n_ties,
        "n_total": n_total,
        "avg_positive_score": avg_positive_score,
        "avg_negative_score": avg_negative_score,
    }
