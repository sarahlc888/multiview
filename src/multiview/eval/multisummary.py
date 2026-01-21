"""Multisummary evaluation method.

This module implements evaluation using multiple summaries per document: generating
k criterion-focused summaries for each document, then computing max-similarity pooling
over embeddings for triplet evaluation.

Key features:
- Generates k summaries per document at eval time
- Embeds all summaries using any embedding preset
- Uses symmetric max-similarity pooling for robust comparison
- Supports configurable number of summaries (k)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from multiview.eval.generation_utils import generate_text_variations_from_documents
from multiview.eval.similarity import cosine_similarity
from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def evaluate_with_multisummary(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    criterion: str,
    criterion_description: str,
    num_summaries: int = 5,
    summary_preset: str = "document_to_summaries_gemini",
    embedding_preset: str = "openai_embedding_small",
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using multiple summaries per document with max-similarity pooling.

    This method:
    1. Generates k summaries for each document using a configurable LM
    2. Embeds all summaries using any embedding model
    3. Groups embeddings by document (each doc has k embeddings)
    4. Computes triplet scores using symmetric max-similarity pooling

    Max-similarity pooling:
    - For each anchor summary, find max similarity with any positive/negative summary
    - Average across all anchor summaries for final score
    - Symmetric: treats anchor and target symmetrically

    Args:
        documents: List of document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        criterion: Criterion name (e.g., "arithmetic")
        criterion_description: Detailed description of criterion (required)
        num_summaries: Number of summaries to generate per document (default: 5)
        summary_preset: Inference preset for summary generation (default: "document_to_summaries_gemini")
        embedding_preset: Embedding preset (e.g., "openai_embedding_small")
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
                    "method_type": "multisummary",
                    "embedding_preset": "...",
                    "summary_preset": "...",
                    "num_summaries": 5,
                    "anchor_id": 0,
                    "positive_id": 1,
                    "negative_id": 2,
                    "anchor": "original doc",
                    "positive": "original doc",
                    "negative": "original doc",
                    "anchor_summaries": ["summary 1", "summary 2", ...],
                    "positive_summaries": ["summary 1", "summary 2", ...],
                    "negative_summaries": ["summary 1", "summary 2", ...],
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

    logger.info(
        f"Evaluating {len(triplet_ids)} triplets with multisummary (k={num_summaries})"
    )
    logger.info(f"Generating {num_summaries} summaries for {len(documents)} documents")

    # Step 1: Generate k summaries for each document
    all_summaries_flat = _generate_multisummaries(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        num_summaries=num_summaries,
        summary_preset=summary_preset,
        cache_alias=cache_alias,
        run_name=run_name,
    )

    # Step 2: Embed all summaries
    logger.info(
        f"Embedding {len(all_summaries_flat)} summaries with preset: {embedding_preset}"
    )
    all_embeddings_flat = _embed_summaries(
        summaries=all_summaries_flat,
        embedding_preset=embedding_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        preset_overrides=preset_overrides,
        criterion=criterion,
    )

    # Step 3: Group embeddings by document
    summaries_by_doc = _group_by_document(
        all_summaries_flat, num_documents=len(documents), k=num_summaries
    )
    embeddings_by_doc = _group_by_document(
        all_embeddings_flat, num_documents=len(documents), k=num_summaries
    )

    # Step 4: Evaluate triplets with max-similarity
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        # Get embeddings for this triplet
        anchor_embs = embeddings_by_doc[anchor_id]
        positive_embs = embeddings_by_doc[positive_id]
        negative_embs = embeddings_by_doc[negative_id]

        logger.debug(f"Anchor embeddings: {np.array(anchor_embs).shape=}")
        logger.debug(f"Positive embeddings: {np.array(positive_embs).shape=}")
        logger.debug(f"Negative embeddings: {np.array(negative_embs).shape=}")

        # Compute max-similarity scores
        pos_score = _max_similarity(anchor_embs, positive_embs)
        neg_score = _max_similarity(anchor_embs, negative_embs)

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
            "method_type": "multisummary",
            "embedding_preset": embedding_preset,
            "summary_preset": summary_preset,
            "num_summaries": num_summaries,
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "anchor": documents[anchor_id],
            "positive": documents[positive_id],
            "negative": documents[negative_id],
            "anchor_summaries": summaries_by_doc[anchor_id],
            "positive_summaries": summaries_by_doc[positive_id],
            "negative_summaries": summaries_by_doc[negative_id],
            "positive_score": float(pos_score),
            "negative_score": float(neg_score),
            "outcome": outcome,
            "correct": outcome == 1,
            "is_tie": outcome == 0,
        }

        triplet_logs.append(log_entry)

    logger.info(
        f"Multisummary evaluation complete: {len(positive_scores)} triplets evaluated"
    )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "triplet_logs": triplet_logs,
    }


def _generate_multisummaries(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    num_summaries: int,
    summary_preset: str,
    cache_alias: str | None,
    run_name: str | None,
) -> list[str]:
    """Generate k summaries for each document.

    Args:
        documents: List of document texts
        criterion: Criterion name
        criterion_description: Criterion description (required)
        num_summaries: Number of summaries per document (k)
        summary_preset: Inference preset for summary generation
        cache_alias: Optional cache identifier
        run_name: Optional run name

    Returns:
        Flat list of all summaries (len = num_documents × k)
    """
    return generate_text_variations_from_documents(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        num_variations=num_summaries,
        generation_preset=summary_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        cache_suffix="summaries",
    )


def _embed_summaries(
    summaries: list[str],
    embedding_preset: str,
    cache_alias: str | None,
    run_name: str | None,
    preset_overrides: dict | None,
    criterion: str | None,
) -> list[list[float]]:
    """Embed all summaries using the specified embedding preset.

    Args:
        summaries: List of summary texts
        embedding_preset: Inference preset for embeddings
        cache_alias: Optional cache identifier
        run_name: Optional run name
        preset_overrides: Optional preset overrides
        criterion: Criterion name (for instruction-tuned embeddings)

    Returns:
        List of embedding vectors
    """
    # Use cache alias with _embeddings suffix
    embedding_cache_alias = f"{cache_alias}_embeddings" if cache_alias else None

    inference_kwargs = {"verbose": False}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)

    # Build inputs - include criterion if provided
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

    return embeddings


def _group_by_document(
    flat_list: list[Any], num_documents: int, k: int
) -> list[list[Any]]:
    """Reshape flat list into nested structure grouped by document.

    Args:
        flat_list: Flat list of items (len = num_documents × k)
        num_documents: Number of documents
        k: Number of items per document

    Returns:
        Nested list where result[doc_id][item_idx] = flat_list[doc_id * k + item_idx]
    """
    if len(flat_list) != num_documents * k:
        raise ValueError(
            f"Expected {num_documents * k} items, got {len(flat_list)}. "
            f"Cannot group into {num_documents} documents with k={k}."
        )

    grouped = []
    for doc_id in range(num_documents):
        start_idx = doc_id * k
        end_idx = start_idx + k
        grouped.append(flat_list[start_idx:end_idx])

    return grouped


def _max_similarity(
    embeddings_a: list[list[float]], embeddings_b: list[list[float]]
) -> float:
    """Compute symmetric max-similarity between two sets of embeddings.

    For each embedding in set A, finds the maximum similarity with any embedding
    in set B, then averages these maximum similarities.

    This is symmetric: max_similarity(A, B) = max_similarity(B, A)

    Args:
        embeddings_a: List of k embeddings for first document
        embeddings_b: List of k embeddings for second document

    Returns:
        Average of maximum similarities (float in [0, 1])
    """
    if not embeddings_a or not embeddings_b:
        return 0.0

    max_scores = []
    for emb_a in embeddings_a:
        # Find max similarity of this anchor embedding with all target embeddings
        scores = [cosine_similarity(emb_a, emb_b) for emb_b in embeddings_b]
        max_scores.append(max(scores))

    # Average the max similarities
    return sum(max_scores) / len(max_scores)
