"""Candidate selection strategies for triplet creation.

This module provides multiple strategies for selecting candidate positives/negatives:
- BM25 similarity over summaries
- Embedding similarity (using HF or OpenAI embeddings)
- Jaccard similarity over tags
- Spurious hard negatives (high spurious similarity, low true similarity)
"""

import logging

import numpy as np

from multiview.benchmark.triplets.utils import (
    annotation_final_summary,
    extract_active_tags,
    jaccard_similarity,
)
from multiview.inference.inference import run_inference
from multiview.utils.bm25_utils import compute_bm25_scores

logger = logging.getLogger(__name__)


def _texts_for_similarity(
    *,
    documents: list[str | dict],
    annotations: list[dict],
    use_summary: bool,
) -> list[str | dict]:
    """Return a corpus for similarity scoring (summaries or raw documents)."""
    if not use_summary:
        # Return documents as-is (BM25 will handle text extraction)
        return documents
    return [annotation_final_summary(ann) for ann in annotations]


def select_candidates_bm25(
    documents: list[str | dict],
    annotations: list[dict],
    anchor_idx: int,
    k: int = 10,
    use_summary: bool = True,
) -> list[tuple[int, float]]:
    """Select candidates using BM25 similarity.

    Args:
        documents: List of documents (strings or dicts)
        annotations: List of annotation dicts (with summaries if use_summary=True)
        anchor_idx: Index of anchor document
        k: Number of candidates to return
        use_summary: If True, use summaries; otherwise use raw documents

    Returns:
        List of (index, score) tuples, sorted by score descending
    """
    corpus = _texts_for_similarity(
        documents=documents, annotations=annotations, use_summary=use_summary
    )

    # Compute BM25 scores using advanced tokenization
    scores = compute_bm25_scores(corpus, anchor_idx)

    # Exclude anchor itself
    scores[anchor_idx] = -np.inf

    # Get top k indices
    top_k_indices = np.argsort(scores, kind="stable")[::-1][:k]

    # Return (index, score) tuples
    candidates = [(int(idx), float(scores[idx])) for idx in top_k_indices]

    return candidates


def select_candidates_embedding(
    documents: list[str | dict],
    annotations: list[dict],
    anchor_idx: int,
    k: int = 10,
    embedding_preset: str = "hf_qwen3_embedding_8b",
    use_summary: bool = True,
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> list[tuple[int, float]]:
    """Select candidates using embedding similarity.

    Args:
        documents: List of document strings
        annotations: List of annotation dicts (with summaries if use_summary=True)
        anchor_idx: Index of anchor document
        k: Number of candidates to return
        embedding_preset: Preset name for embedding model
        use_summary: If True, use summaries; otherwise use raw documents
        cache_alias: Optional cache alias for inference calls
        run_name: Optional experiment/run name for cache organization

    Returns:
        List of (index, score) tuples, sorted by score descending
    """
    logger.warning(
        "Introducing bias into the triplet selection process by using embeddings for candidate selection, but also planning to evaluate on embeddings later."
    )
    texts = _texts_for_similarity(
        documents=documents, annotations=annotations, use_summary=use_summary
    )

    # Get embeddings
    inputs = {"document": texts}
    embeddings = run_inference(
        inputs=inputs,
        config=embedding_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Compute cosine similarities to anchor
    anchor_embedding = embeddings[anchor_idx]
    similarities = np.dot(embeddings, anchor_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(anchor_embedding)
    )

    # Exclude anchor itself
    similarities[anchor_idx] = -np.inf

    # Get top k indices
    top_k_indices = np.argsort(similarities, kind="stable")[::-1][:k]

    # Return (index, score) tuples
    candidates = [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    return candidates


def select_candidates_jaccard(
    annotations: list[dict],
    anchor_idx: int,
    k: int = 10,
    use_spurious: bool = False,
) -> list[tuple[int, float]]:
    """Select candidates using Jaccard similarity over tags.

    Args:
        annotations: List of annotation dicts (with tags)
        anchor_idx: Index of anchor document
        k: Number of candidates to return
        use_spurious: If True, use spurious_tags; otherwise use tags

    Returns:
        List of (index, score) tuples, sorted by score descending
    """
    # Get tag key
    tag_key = "spurious_tags" if use_spurious else "tags"

    # Get anchor tags
    anchor_tags = extract_active_tags(annotations[anchor_idx], tag_key)

    # Compute Jaccard similarity for all documents
    similarities = []
    for i, ann in enumerate(annotations):
        if i == anchor_idx:
            similarities.append(-np.inf)
            continue

        doc_tags = extract_active_tags(ann, tag_key)

        # Jaccard similarity using helper
        similarity = jaccard_similarity(anchor_tags, doc_tags)

        similarities.append(similarity)

    # Get top k indices
    similarities = np.array(similarities)
    top_k_indices = np.argsort(similarities, kind="stable")[::-1][:k]
    # Filter out anchor in case corpus size < k (anchor has -np.inf score)
    top_k_indices = [idx for idx in top_k_indices if idx != anchor_idx]

    # Return (index, score) tuples
    candidates = [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    return candidates


def select_spurious_hard_negatives(
    annotations: list[dict],
    anchor_idx: int,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Select spurious hard negatives (high spurious similarity, low true similarity).

    Args:
        annotations: List of annotation dicts (with tags and spurious_tags)
        anchor_idx: Index of anchor document
        k: Number of candidates to return

    Returns:
        List of (index, score) tuples, sorted by spurious similarity descending
    """
    # Get anchor tags
    anchor_tags = extract_active_tags(annotations[anchor_idx], "tags")
    anchor_spurious = extract_active_tags(annotations[anchor_idx], "spurious_tags")

    # Compute similarities
    candidates_with_scores = []
    for i, ann in enumerate(annotations):
        if i == anchor_idx:
            continue

        doc_tags = extract_active_tags(ann, "tags")
        doc_spurious = extract_active_tags(ann, "spurious_tags")

        # True similarity (should be low) using helper
        true_sim = jaccard_similarity(anchor_tags, doc_tags)

        # Spurious similarity (should be high) using helper
        spurious_sim = jaccard_similarity(anchor_spurious, doc_spurious)

        # Score: prioritize high spurious sim and low true sim
        # Hard negative score = spurious_sim - true_sim
        score = spurious_sim - true_sim

        candidates_with_scores.append((i, score, spurious_sim, true_sim))

    # Sort by score descending
    candidates_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top k (index, score) tuples
    candidates = [(idx, score) for idx, score, _, _ in candidates_with_scores[:k]]

    return candidates


def merge_candidate_pools(
    *candidate_lists: list[tuple[int, float]],
    deduplicate: bool = True,
    use_rrf: bool = True,
    rrf_k: int = 60,
) -> list[int]:
    """Merge multiple candidate pools using Reciprocal Rank Fusion or simple concatenation.

    Reciprocal Rank Fusion (RRF) is a robust method for combining rankings that:
    - Treats all retrieval strategies equally (no bias toward first list)
    - Rewards consensus (documents appearing in multiple lists score higher)
    - Uses reciprocal rank scoring: score = sum(1/(k + rank)) across all lists

    Example with k=60:
        - Doc 42 appears at rank 1 in BM25, rank 3 in embedding → score = 1/61 + 1/63 = 0.032
        - Doc 17 appears at rank 2 in BM25 only → score = 1/62 = 0.016
        - Doc 42 ranks higher due to consensus across strategies

    Args:
        *candidate_lists: Variable number of candidate lists (each is list of (index, score))
        deduplicate: If True, remove duplicates (only used for simple merge, not RRF)
        use_rrf: If True, use Reciprocal Rank Fusion; otherwise simple concatenation
        rrf_k: Constant for RRF scoring (default 60 is standard from literature)

    Returns:
        List of candidate indices sorted by RRF score (or concatenated if use_rrf=False)
    """
    if use_rrf:
        # Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across all lists
        rrf_scores = {}

        for candidates in candidate_lists:
            for rank, (idx, _) in enumerate(candidates, start=1):
                if idx not in rrf_scores:
                    rrf_scores[idx] = 0.0
                rrf_scores[idx] += 1.0 / (rrf_k + rank)

        # Sort by RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_candidates]

    else:
        # Simple concatenation (old behavior)
        merged = []

        for candidates in candidate_lists:
            for idx, _ in candidates:
                merged.append(idx)

        if deduplicate:
            # Preserve order, remove duplicates
            seen = set()
            deduplicated = []
            for idx in merged:
                if idx not in seen:
                    deduplicated.append(idx)
                    seen.add(idx)
            return deduplicated

        return merged
