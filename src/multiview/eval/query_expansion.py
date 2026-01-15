"""Query expansion evaluation method.

This module implements evaluation using query expansion: generating summaries
for all documents at eval time, then computing similarity over summaries using
BM25 or embeddings.

Key differences from other evaluation methods:
- Generates summaries fresh at eval time (not from annotations)
- Expands ALL documents (anchor, positive, negative)
- Supports both BM25 and embedding-based retrieval
- Configurable summary generation model (any LM preset)
- Independent of data generation annotations
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from multiview.inference.inference import run_inference
from multiview.utils.bm25_utils import compute_bm25_matrix

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


def evaluate_with_query_expansion(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    criterion: str,
    criterion_description: str | None = None,
    retrieval_method: str = "bm25",
    summary_preset: str = "query_expansion_summary_gemini",
    embedding_preset: str = "openai_embedding_small",
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using query expansion with configurable retrieval method.

    This method:
    1. Generates criterion-aware summaries for all documents using a configurable LM
    2. Computes similarity over summaries using BM25 or embeddings
    3. Evaluates triplet accuracy based on similarity scores

    Args:
        documents: List of document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        criterion: Criterion name (e.g., "arithmetic")
        criterion_description: Optional detailed description of criterion
        retrieval_method: "bm25" or "embeddings"
        summary_preset: Inference preset for summary generation (any LM model)
        embedding_preset: Inference preset for embeddings (if retrieval_method="embeddings")
        cache_alias: Optional cache identifier for the entire evaluation
        run_name: Optional experiment/run name for cache organization
        preset_overrides: Optional preset configuration overrides

    Returns:
        Dict with structure:
        {
            "positive_scores": [...],
            "negative_scores": [...],
            "triplet_logs": [
                {
                    "triplet_idx": 0,
                    "method_type": "query_expansion",
                    "retrieval_method": "bm25",
                    "summary_preset": "...",
                    "embedding_preset": "..." (if applicable),
                    "anchor_id": 0,
                    "positive_id": 1,
                    "negative_id": 2,
                    "anchor": "original doc",
                    "positive": "original doc",
                    "negative": "original doc",
                    "anchor_summary": "generated summary",
                    "positive_summary": "generated summary",
                    "negative_summary": "generated summary",
                    "positive_score": 0.85,
                    "negative_score": 0.42,
                    "outcome": 1,
                },
                ...
            ]
        }
    """
    if not triplet_ids:
        logger.warning("No triplets provided for evaluation")
        return {
            "positive_scores": [],
            "negative_scores": [],
            "triplet_logs": [],
        }

    if retrieval_method not in ("bm25", "embeddings"):
        raise ValueError(
            f"Invalid retrieval_method: {retrieval_method}. Must be 'bm25' or 'embeddings'"
        )

    logger.info(
        f"Evaluating {len(triplet_ids)} triplets with query expansion ({retrieval_method})"
    )
    logger.info(f"Generating summaries for {len(documents)} documents")

    # Step 1: Generate summaries for all documents
    summaries = _generate_summaries(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        summary_preset=summary_preset,
        cache_alias=cache_alias,
        run_name=run_name,
    )

    # Step 2: Compute similarity matrix over summaries
    if retrieval_method == "bm25":
        logger.info(f"Computing BM25 similarity matrix over {len(summaries)} summaries")
        similarity_matrix = _compute_bm25_similarity_matrix(summaries)
    else:  # embeddings
        logger.info(
            f"Computing embedding similarity matrix over {len(summaries)} summaries"
        )
        logger.info(f"Using embedding preset: {embedding_preset}")
        similarity_matrix = _compute_embedding_similarity_matrix(
            summaries=summaries,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            run_name=run_name,
            preset_overrides=preset_overrides,
            criterion=criterion,
        )

    # Step 3: Evaluate triplets
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        pos_score = similarity_matrix[anchor_id][positive_id]
        neg_score = similarity_matrix[anchor_id][negative_id]
        positive_scores.append(float(pos_score))
        negative_scores.append(float(neg_score))

        if pos_score > neg_score:
            outcome = 1
        elif neg_score > pos_score:
            outcome = -1
        else:
            outcome = 0

        log_entry = {
            "triplet_idx": i,
            "method_type": "query_expansion",
            "retrieval_method": retrieval_method,
            "summary_preset": summary_preset,
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "anchor": documents[anchor_id],
            "positive": documents[positive_id],
            "negative": documents[negative_id],
            "anchor_summary": summaries[anchor_id],
            "positive_summary": summaries[positive_id],
            "negative_summary": summaries[negative_id],
            "positive_score": float(pos_score),
            "negative_score": float(neg_score),
            "outcome": outcome,
        }

        # Add embedding preset to logs if using embeddings
        if retrieval_method == "embeddings":
            log_entry["embedding_preset"] = embedding_preset

        triplet_logs.append(log_entry)

    logger.info(
        f"Query expansion ({retrieval_method}) evaluation complete: "
        f"{len(positive_scores)} triplets evaluated"
    )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "triplet_logs": triplet_logs,
    }


# Backwards compatibility alias
def evaluate_with_query_expansion_bm25(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    criterion: str,
    criterion_description: str | None = None,
    summary_preset: str = "query_expansion_summary_gemini",
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Backwards compatibility wrapper for BM25 query expansion.

    Deprecated: Use evaluate_with_query_expansion with retrieval_method='bm25' instead.
    """
    return evaluate_with_query_expansion(
        documents=documents,
        triplet_ids=triplet_ids,
        criterion=criterion,
        criterion_description=criterion_description,
        retrieval_method="bm25",
        summary_preset=summary_preset,
        cache_alias=cache_alias,
        run_name=run_name,
    )


def _generate_summaries(
    documents: list[str],
    criterion: str,
    criterion_description: str | None,
    summary_preset: str,
    cache_alias: str | None,
    run_name: str | None,
) -> list[str]:
    """Generate criterion-aware summaries for documents.

    Args:
        documents: List of document texts
        criterion: Criterion name
        criterion_description: Optional criterion description
        summary_preset: Inference preset for summary generation
        cache_alias: Optional cache identifier
        run_name: Optional run name

    Returns:
        List of summary strings (one per document)
    """
    # Prepare inputs for batch inference
    inputs = {
        "criterion": criterion,
        "criterion_description": criterion_description or "",
        "document": documents,
    }

    # Use cache alias with _summary suffix to distinguish from other caches
    summary_cache_alias = f"{cache_alias}_summary" if cache_alias else None

    logger.info(f"Generating summaries with preset: {summary_preset}")
    if summary_cache_alias:
        logger.info(f"Using cache alias: {summary_cache_alias}")

    # Run inference to generate summaries
    results = run_inference(
        inputs=inputs,
        config=summary_preset,
        cache_alias=summary_cache_alias,
        run_name=run_name,
        verbose=False,
    )

    # Extract final_summary from each result
    summaries: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, dict) and "final_summary" in result:
            summary = result["final_summary"]
            if isinstance(summary, str) and summary.strip():
                summaries.append(summary)
            else:
                # Fallback to original document if summary is empty
                logger.warning(
                    f"Empty summary for document {i}, using original document"
                )
                summaries.append(documents[i])
        else:
            # Fallback to original document if result is malformed
            logger.warning(
                f"Malformed summary result for document {i}: {result}. "
                f"Using original document."
            )
            summaries.append(documents[i])

    return summaries


def _compute_bm25_similarity_matrix(summaries: list[str]) -> np.ndarray:
    """Compute BM25 similarity matrix over summaries.

    Args:
        summaries: List of summary texts

    Returns:
        N×N similarity matrix where matrix[i][j] is BM25 score from summary i to summary j
    """
    return compute_bm25_matrix(summaries)


def _compute_embedding_similarity_matrix(
    summaries: list[str],
    embedding_preset: str,
    cache_alias: str | None,
    run_name: str | None,
    preset_overrides: dict | None,
    criterion: str | None = None,
) -> np.ndarray:
    """Compute embedding-based cosine similarity matrix over summaries.

    Args:
        summaries: List of summary texts
        embedding_preset: Inference preset for embeddings
        cache_alias: Optional cache identifier
        run_name: Optional run name
        preset_overrides: Optional preset overrides
        criterion: Criterion name (required for instruction-tuned embeddings)

    Returns:
        N×N similarity matrix where matrix[i][j] is cosine similarity between summary i and j
    """
    # Use cache alias with _embeddings suffix
    embedding_cache_alias = f"{cache_alias}_embeddings" if cache_alias else None

    inference_kwargs = {"verbose": False}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)

    # Build inputs - include criterion if provided (needed for instruction-tuned embeddings)
    inputs = {"document": summaries}
    if criterion is not None:
        inputs["criterion"] = criterion

    # Generate embeddings for all summaries
    embeddings = run_inference(
        inputs=inputs,
        config=embedding_preset,
        cache_alias=embedding_cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )

    # Compute pairwise cosine similarity matrix
    n = len(embeddings)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])

    return similarity_matrix
