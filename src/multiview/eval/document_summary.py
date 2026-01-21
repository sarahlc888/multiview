"""Document summarization evaluation method.

This module implements evaluation using document summarization: generating
criterion-focused summaries for documents at eval time, then computing
similarity over summaries using BM25 or embeddings.

Key differences from other evaluation methods:
- Generates summaries fresh at eval time (not from annotations)
- Optimized to only summarize documents that appear in triplets
- For embeddings: computes only the specific pairwise similarities needed
- For BM25: builds a matrix only over documents in triplets
- Supports both BM25 and embedding-based retrieval
- Configurable summary generation model (any LM preset)
- Independent of data generation annotations
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from multiview.eval.similarity import cosine_similarity
from multiview.inference.inference import run_inference
from multiview.utils.bm25_utils import compute_bm25_matrix

logger = logging.getLogger(__name__)


def evaluate_with_document_rewrite(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    criterion: str,
    criterion_description: str,
    summary_preset: str = "document_summary_gemini",
    embedding_preset: str = "bm25_lexical",
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using document rewriting with BM25 or embeddings.

    This method:
    1. Extracts unique document IDs from triplets (optimization)
    2. Generates criterion-aware summaries only for those documents using a configurable LM
    3. For embeddings: computes only the pairwise similarities needed for triplets
    4. For BM25: builds a similarity matrix over the unique documents
    5. Evaluates triplet accuracy based on similarity scores

    Args:
        documents: List of document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        criterion: Criterion name (e.g., "arithmetic")
        criterion_description: Detailed description of criterion (required)
        summary_preset: Inference preset for summary generation (any LM model)
        embedding_preset: BM25 preset (e.g., "bm25_lexical") or any embedding preset (e.g., "openai_embedding_small")
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
                    "method_type": "document_rewrite",
                    "embedding_preset": "bm25" or embedding model name,
                    "summary_preset": "...",
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

    logger.info(
        f"Evaluating {len(triplet_ids)} triplets with document rewrite (preset={embedding_preset})"
    )

    # Step 1: Extract unique document IDs from triplets
    unique_doc_ids = set()
    for anchor_id, positive_id, negative_id in triplet_ids:
        unique_doc_ids.add(anchor_id)
        unique_doc_ids.add(positive_id)
        unique_doc_ids.add(negative_id)
    unique_doc_ids = sorted(unique_doc_ids)

    logger.info(
        f"Generating summaries for {len(unique_doc_ids)} unique documents "
        f"(out of {len(documents)} total)"
    )

    # Step 2: Generate summaries only for unique documents
    unique_documents = [documents[i] for i in unique_doc_ids]
    unique_summaries = _generate_summaries(
        documents=unique_documents,
        criterion=criterion,
        criterion_description=criterion_description,
        summary_preset=summary_preset,
        cache_alias=cache_alias,
        run_name=run_name,
    )

    # Create mapping from original doc ID to summary
    doc_id_to_summary = dict(zip(unique_doc_ids, unique_summaries, strict=False))

    # Step 3: Compute similarities only for triplets (no full matrix)
    # Check if using BM25 (backward compatible with "bm25" string or proper presets)
    is_bm25 = embedding_preset == "bm25" or embedding_preset.startswith("bm25_")

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    if is_bm25:
        # For BM25, we still need a matrix over unique documents
        logger.info(
            f"Computing BM25 similarity matrix over {len(unique_summaries)} unique summaries"
        )
        similarity_matrix = _compute_bm25_similarity_matrix(unique_summaries)
        # Create mapping from original doc ID to matrix index
        doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(unique_doc_ids)}

        for _i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
            anchor_idx = doc_id_to_idx[anchor_id]
            positive_idx = doc_id_to_idx[positive_id]
            negative_idx = doc_id_to_idx[negative_id]

            pos_score = similarity_matrix[anchor_idx][positive_idx]
            neg_score = similarity_matrix[anchor_idx][negative_idx]
            positive_scores.append(float(pos_score))
            negative_scores.append(float(neg_score))
    else:
        # For embeddings, compute only the similarities we need
        logger.info(
            f"Computing embeddings for {len(unique_summaries)} unique summaries"
        )
        logger.info(f"Using embedding preset: {embedding_preset}")

        # Get embeddings for unique documents
        doc_id_to_embedding = _compute_embeddings_for_documents(
            summaries=unique_summaries,
            doc_ids=unique_doc_ids,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            run_name=run_name,
            preset_overrides=preset_overrides,
            criterion=criterion,
        )

        # Compute only the specific similarities needed for triplets
        logger.info(
            f"Computing {len(triplet_ids) * 2} pairwise similarities for triplets"
        )
        for _i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
            anchor_emb = doc_id_to_embedding[anchor_id]
            positive_emb = doc_id_to_embedding[positive_id]
            negative_emb = doc_id_to_embedding[negative_id]

            pos_score = cosine_similarity(anchor_emb, positive_emb)
            neg_score = cosine_similarity(anchor_emb, negative_emb)
            positive_scores.append(float(pos_score))
            negative_scores.append(float(neg_score))

    # Step 4: Build triplet logs
    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        pos_score = positive_scores[i]
        neg_score = negative_scores[i]

        if pos_score > neg_score:
            outcome = 1
        elif neg_score > pos_score:
            outcome = -1
        else:
            outcome = 0

        log_entry = {
            "triplet_idx": i,
            "method_type": "document_rewrite",
            "embedding_preset": embedding_preset,
            "summary_preset": summary_preset,
            "anchor_id": anchor_id,
            "positive_id": positive_id,
            "negative_id": negative_id,
            "anchor": documents[anchor_id],
            "positive": documents[positive_id],
            "negative": documents[negative_id],
            "anchor_summary": doc_id_to_summary[anchor_id],
            "positive_summary": doc_id_to_summary[positive_id],
            "negative_summary": doc_id_to_summary[negative_id],
            "positive_score": float(pos_score),
            "negative_score": float(neg_score),
            "outcome": outcome,
            "correct": outcome == 1,
            "is_tie": outcome == 0,
        }

        triplet_logs.append(log_entry)

    logger.info(
        f"Document rewrite (preset={embedding_preset}) evaluation complete: "
        f"{len(positive_scores)} triplets evaluated"
    )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "triplet_logs": triplet_logs,
    }


def _generate_summaries(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    summary_preset: str,
    cache_alias: str | None,
    run_name: str | None,
) -> list[str]:
    """Generate criterion-aware summaries for documents.

    Args:
        documents: List of document texts
        criterion: Criterion name
        criterion_description: Criterion description (required)
        summary_preset: Inference preset for summary generation
        cache_alias: Optional cache identifier
        run_name: Optional run name

    Returns:
        List of summary strings (one per document)
    """
    # Prepare inputs for batch inference
    inputs = {
        "criterion": criterion,
        "criterion_description": criterion_description,
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


def _compute_embeddings_for_documents(
    summaries: list[str],
    doc_ids: list[int],
    embedding_preset: str,
    cache_alias: str | None,
    run_name: str | None,
    preset_overrides: dict | None,
    criterion: str | None = None,
) -> dict[int, list[float]]:
    """Compute embeddings for document summaries.

    Args:
        summaries: List of summary texts
        doc_ids: List of document IDs corresponding to summaries
        embedding_preset: Inference preset for embeddings
        cache_alias: Optional cache identifier
        run_name: Optional run name
        preset_overrides: Optional preset overrides
        criterion: Criterion name (required for instruction-tuned embeddings)

    Returns:
        Dictionary mapping doc_id -> embedding vector
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

    # Return mapping from doc_id to embedding
    return dict(zip(doc_ids, embeddings, strict=False))
