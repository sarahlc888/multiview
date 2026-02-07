"""Query relevance vectors evaluation method.

This module implements evaluation using query relevance vectors: generating queries
from dev set documents, then creating score-vector representations for all documents
based on their relevance to the expanded queries.

Workflow:
1. Sample a dev set of documents
2. Generate k queries FOR EACH dev set document using LLM
3. Take global union of all queries (dev_set_size × k total queries)
4. Create score vectors for all documents (relevance to each query in union)
5. Evaluate triplet accuracy using L2 distance on score vectors
"""

from __future__ import annotations

import hashlib
import logging
import random
from typing import Any

import numpy as np

from multiview.eval.generation_utils import generate_text_variations_from_documents
from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def l2_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.linalg.norm(a - b))


def evaluate_with_query_relevance_vectors(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
    criterion: str,
    criterion_description: str | None = None,
    expansion_preset: str = "query_relevance_scores_gemini",
    embedding_preset: str = "openai_embedding_small",
    num_expansions: int = 10,
    dev_set_size: int = 25,
    random_seed: int | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using query relevance vectors.

    Workflow:
    1. Samples a dev set of documents
    2. Generates k queries FOR EACH dev set document using LLM
    3. Takes global union of all queries (dev_set_size × k total queries)
    4. Creates score vectors for all documents (relevance to each query in union)
    5. Evaluates triplet accuracy using cosine similarity on score vectors

    Args:
        documents: List of document texts
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        criterion: Criterion name (e.g., "arithmetic complexity")
        criterion_description: Optional detailed description of criterion
        expansion_preset: Inference preset for query expansion (LLM model)
        embedding_preset: Inference preset for embeddings (relevance scoring)
        num_expansions: Number of query variations to generate per document (k value)
        dev_set_size: Number of documents to use for query expansion
        random_seed: Seed for dev set sampling (if None, derive from task)
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
                    "method_type": "query_relevance_vectors",
                    "expansion_preset": "...",
                    "embedding_preset": "...",
                    "num_expansions": 10,
                    "anchor_id": 0,
                    "positive_id": 1,
                    "negative_id": 2,
                    "anchor": "original doc",
                    "positive": "original doc",
                    "negative": "original doc",
                    "expanded_queries": ["query1", "query2", ...],
                    "anchor_vector": [0.1, 0.2, ...],
                    "positive_vector": [0.3, 0.4, ...],
                    "negative_vector": [0.05, 0.1, ...],
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

    if dev_set_size > len(documents):
        logger.warning(
            f"dev_set_size ({dev_set_size}) > len(documents) ({len(documents)}), "
            f"using all documents for dev set"
        )
        dev_set_size = len(documents)

    if dev_set_size < 5:
        logger.warning(
            f"dev_set_size ({dev_set_size}) is very small, "
            f"query expansion may be less effective"
        )

    logger.info(f"Evaluating {len(triplet_ids)} triplets with query relevance vectors")
    logger.info(
        f"Parameters: k={num_expansions}, dev_set_size={dev_set_size}, "
        f"expansion_preset={expansion_preset}, embedding_preset={embedding_preset}"
    )

    # Step 0: Sample dev set for query expansion
    dev_set_docs, dev_set_seed = _sample_dev_set(
        documents=documents,
        dev_set_size=dev_set_size,
        criterion=criterion,
        random_seed=random_seed,
    )
    logger.info(
        f"Sampled {len(dev_set_docs)} documents for dev set (seed={dev_set_seed})"
    )

    # Step 1: Generate k queries FOR EACH dev set document, then union
    expanded_queries = generate_text_variations_from_documents(
        documents=dev_set_docs,
        criterion=criterion,
        criterion_description=criterion_description,
        num_variations=num_expansions,
        generation_preset=expansion_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        cache_suffix="expansion",
    )
    logger.info(
        f"Generated {len(expanded_queries)} total queries from {len(dev_set_docs)} "
        f"dev set documents ({num_expansions} queries per document)"
    )
    logger.debug(f"Example expanded queries: {expanded_queries=}")

    # Step 2: Create score vectors for all documents
    score_vectors, document_embeddings = _create_score_vectors(
        documents=documents,
        expanded_queries=expanded_queries,
        embedding_preset=embedding_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        preset_overrides=preset_overrides,
        criterion=criterion,
    )
    logger.info(f"Created {len(score_vectors)} score vectors")

    # Step 3: Evaluate triplets
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        anchor_vec = score_vectors[anchor_id]
        positive_vec = score_vectors[positive_id]
        negative_vec = score_vectors[negative_id]

        pos_score = l2_distance(anchor_vec, positive_vec)
        neg_score = l2_distance(anchor_vec, negative_vec)

        positive_scores.append(pos_score)
        negative_scores.append(neg_score)

        # With L2 distance, smaller is better (closer), so flip comparison
        if pos_score < neg_score:
            outcome = 1
        elif neg_score < pos_score:
            outcome = -1
        else:
            outcome = 0

        triplet_logs.append(
            {
                "triplet_idx": i,
                "method_type": "query_relevance_vectors",
                "criterion": criterion,
                "criterion_description": criterion_description,
                "expansion_preset": expansion_preset,
                "embedding_preset": embedding_preset,
                "num_expansions": num_expansions,
                "dev_set_size": dev_set_size,
                "dev_set_seed": dev_set_seed,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "anchor": documents[anchor_id],
                "positive": documents[positive_id],
                "negative": documents[negative_id],
                "expanded_queries": expanded_queries,
                "anchor_vector": anchor_vec,
                "positive_vector": positive_vec,
                "negative_vector": negative_vec,
                "positive_score": pos_score,
                "negative_score": neg_score,
                "outcome": outcome,
                "correct": outcome == 1,
                "is_tie": outcome == 0,
            }
        )

    logger.info(
        f"Query relevance vectors evaluation complete: {len(positive_scores)} triplets evaluated"
    )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "triplet_logs": triplet_logs,
        "embeddings": document_embeddings,  # Add document embeddings to results
    }


def _sample_dev_set(
    documents: list[str],
    dev_set_size: int,
    criterion: str,
    random_seed: int | None = None,
) -> tuple[list[str], int]:
    """Sample a dev set of documents for query expansion.

    Uses deterministic sampling based on criterion hash for reproducibility.

    Args:
        documents: List of all document texts
        dev_set_size: Number of documents to sample
        criterion: Criterion name (used to derive seed if random_seed is None)
        random_seed: Optional explicit seed

    Returns:
        Tuple of (dev_set_documents, seed_used)
    """
    if random_seed is None:
        # Derive seed from criterion for reproducibility
        criterion_hash = hashlib.md5(criterion.encode()).hexdigest()
        random_seed = int(criterion_hash[:8], 16) % 10000

    # Create a new Random instance to avoid affecting global state
    rng = random.Random(random_seed)

    # Sample indices without replacement
    n_docs = len(documents)
    sample_size = min(dev_set_size, n_docs)
    sampled_indices = rng.sample(range(n_docs), sample_size)

    # Extract sampled documents
    dev_set_docs = [documents[i] for i in sampled_indices]

    return dev_set_docs, random_seed


def _create_score_vectors(
    documents: list[str],
    expanded_queries: list[str],
    embedding_preset: str,
    cache_alias: str | None,
    run_name: str | None,
    preset_overrides: dict | None,
    criterion: str | None = None,
) -> tuple[list[list[float]], list]:
    """Create k-dimensional score vectors for all documents.

    For each document, computes embedding-based relevance scores against all
    expanded queries, creating a k-dimensional vector representation.

    Args:
        documents: List of document texts
        expanded_queries: List of k expanded query variations
        embedding_preset: Inference preset for embeddings
        cache_alias: Optional cache identifier
        run_name: Optional run name
        preset_overrides: Optional preset overrides
        criterion: Criterion name (for instruction-tuned embeddings)

    Returns:
        Tuple of (score_vectors, document_embeddings):
        - score_vectors: List of k-dimensional score vectors (one per document)
        - document_embeddings: List of embedding vectors for documents
    """
    k = len(expanded_queries)
    n_docs = len(documents)

    logger.info(
        f"Computing embeddings for {n_docs} documents and {k} queries "
        f"using preset: {embedding_preset}"
    )

    # Use cache alias with _scores suffix
    scores_cache_alias = f"{cache_alias}_scores" if cache_alias else None

    inference_kwargs = {"verbose": False}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)

    # Step 1: Embed all documents
    doc_inputs = {"document": documents}
    if criterion is not None:
        doc_inputs["criterion"] = criterion

    doc_cache_alias = f"{scores_cache_alias}_docs" if scores_cache_alias else None
    document_embeddings = run_inference(
        inputs=doc_inputs,
        config=embedding_preset,
        cache_alias=doc_cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )

    # Step 2: Embed all queries (only once, reused for all documents)
    query_inputs = {"document": expanded_queries}
    if criterion is not None:
        query_inputs["criterion"] = criterion

    query_cache_alias = f"{scores_cache_alias}_queries" if scores_cache_alias else None
    query_embeddings = run_inference(
        inputs=query_inputs,
        config=embedding_preset,
        cache_alias=query_cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )

    # Step 3: Compute score vectors
    # For each document, compute L2 distance with each query embedding
    score_vectors = []
    for _doc_idx, doc_emb in enumerate(document_embeddings):
        scores = []
        for query_emb in query_embeddings:
            score = l2_distance(doc_emb, query_emb)
            scores.append(score)
        score_vectors.append(scores)

    logger.info(f"Created {len(score_vectors)} score vectors of dimension {k}")

    return score_vectors, document_embeddings
