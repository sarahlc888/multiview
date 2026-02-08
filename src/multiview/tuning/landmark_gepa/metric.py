"""Triplet-agreement metric for the landmark GEPA workflow.

The metric evaluates query quality by:
1. Generating queries from the DSPy module output
2. Embedding queries + documents via ``run_inference``
3. Building per-document *score vectors* (distance to each query)
4. Comparing score vectors via L2 distance to evaluate triplet order
5. Returning agreement rate (optionally with GEPA feedback)

This mirrors the approach in ``multiview.eval.query_relevance_vectors``:
queries act as a learned coordinate system, and documents are compared
in that query-defined space rather than in raw embedding space.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from multiview.inference.inference import run_inference
from multiview.tuning.landmark_gepa.module import parse_queries

logger = logging.getLogger(__name__)


def _embed_texts(
    texts: list[str],
    embedding_preset: str,
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> list[Any]:
    """Embed a list of texts using a multiview inference preset."""
    return run_inference(
        inputs={"document": texts},
        config=embedding_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )


def _l2_distance(
    vec_a: list[float] | np.ndarray, vec_b: list[float] | np.ndarray
) -> float:
    """L2 (Euclidean) distance between two vectors."""
    return float(np.linalg.norm(np.array(vec_a) - np.array(vec_b)))


def _build_score_vectors(
    doc_embeddings: list[Any],
    query_embeddings: list[Any],
) -> list[list[float]]:
    """Build a k-dim score vector per document (L2 distance to each query).

    Returns a list of length ``len(doc_embeddings)``, each element being a
    list of ``len(query_embeddings)`` distances.
    """
    score_vectors: list[list[float]] = []
    for doc_emb in doc_embeddings:
        scores = [_l2_distance(doc_emb, q_emb) for q_emb in query_embeddings]
        score_vectors.append(scores)
    return score_vectors


def triplet_agreement(
    queries: list[str],
    documents: list[str | dict],
    triplet_ids: list[tuple[int, int, int]],
    embedding_preset: str = "openai_embedding_small",
    doc_embeddings: list[Any] | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Score queries by triplet agreement in query-relevance-vector space.

    For each document a *score vector* is computed — its L2 distance to every
    query embedding.  Triplets are then evaluated by comparing documents in
    this query-defined space (L2 on score vectors): the positive document
    should be closer to the anchor than the negative.

    Args:
        queries: Generated query strings.
        documents: Corpus documents (text or dict with ``text`` key).
        triplet_ids: ``(anchor_id, positive_id, negative_id)`` index tuples.
        embedding_preset: Preset for ``run_inference``.
        doc_embeddings: Pre-computed document embeddings (avoids re-computation).
        cache_alias: Passed through to ``run_inference``.
        run_name: Passed through to ``run_inference``.

    Returns:
        Dict with ``agreement`` (0–1 score), ``correct``, ``total``, and
        per-triplet ``details``.
    """
    if not queries:
        return {
            "agreement": 0.0,
            "correct": 0,
            "total": len(triplet_ids),
            "details": [],
        }

    # Embed queries
    query_embeddings = _embed_texts(
        queries,
        embedding_preset,
        cache_alias=f"{cache_alias}_queries" if cache_alias else None,
        run_name=run_name,
    )

    # Embed documents if not pre-computed
    if doc_embeddings is None:
        doc_texts = [
            d.get("text", str(d)) if isinstance(d, dict) else str(d) for d in documents
        ]
        doc_embeddings = _embed_texts(
            doc_texts,
            embedding_preset,
            cache_alias=f"{cache_alias}_docs" if cache_alias else None,
            run_name=run_name,
        )

    # Build per-document score vectors in query space
    score_vectors = _build_score_vectors(doc_embeddings, query_embeddings)

    # Evaluate triplets using L2 distance on score vectors
    correct = 0
    details: list[dict[str, Any]] = []
    for anchor_id, positive_id, negative_id in triplet_ids:
        anchor_vec = score_vectors[anchor_id]
        positive_vec = score_vectors[positive_id]
        negative_vec = score_vectors[negative_id]

        # L2 distance in query space (lower = closer)
        pos_dist = _l2_distance(anchor_vec, positive_vec)
        neg_dist = _l2_distance(anchor_vec, negative_vec)

        is_correct = pos_dist < neg_dist
        if is_correct:
            correct += 1

        details.append(
            {
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "pos_dist": float(pos_dist),
                "neg_dist": float(neg_dist),
                "correct": is_correct,
            }
        )

    agreement = correct / len(triplet_ids) if triplet_ids else 0.0
    return {
        "agreement": agreement,
        "correct": correct,
        "total": len(triplet_ids),
        "details": details,
    }


# ---------------------------------------------------------------------------
# DSPy-compatible metric wrapper
# ---------------------------------------------------------------------------


def gepa_metric(
    example: Any,
    pred: Any,
    trace: Any = None,
    **kwargs: Any,
) -> Any:
    """GEPA-friendly metric with margin feedback for landmark query tuning.

    ``example`` should carry ``queries_raw`` (the raw output text) and metadata
    such as ``criteria``, ``documents``, ``triplet_ids``, ``embedding_preset``,
    and optionally ``doc_embeddings``.

    ``pred`` is the DSPy module prediction with a ``queries`` field.
    """
    queries_text = getattr(pred, "queries", "")
    queries = parse_queries(queries_text)
    n_queries = len(queries)

    # Retrieve evaluation context from the example
    documents = example.get("documents", [])
    triplet_ids = example.get("triplet_ids", [])
    embedding_preset = example.get("embedding_preset", "openai_embedding_small")
    doc_embeddings = example.get("doc_embeddings", None)

    if not triplet_ids or not documents:
        score = 0.0
        feedback = "No triplet data available for evaluation."
    elif n_queries == 0:
        score = 0.0
        feedback = "No valid queries were parsed from the output."
    else:
        result = triplet_agreement(
            queries=queries,
            documents=documents,
            triplet_ids=triplet_ids,
            embedding_preset=embedding_preset,
            doc_embeddings=doc_embeddings,
        )
        score = result["agreement"]
        correct = result["correct"]
        total = result["total"]

        criteria_name = example.get("criteria", "unknown")
        if score >= 0.7:
            feedback = (
                f"Good agreement ({correct}/{total}={score:.2f}) for '{criteria_name}' "
                f"with {n_queries} queries. Try to improve coverage of edge cases."
            )
        else:
            feedback = (
                f"Low agreement ({correct}/{total}={score:.2f}) for '{criteria_name}' "
                f"with {n_queries} queries. Generate more diverse, criterion-specific queries."
            )

    try:
        from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

        return ScoreWithFeedback(score=score, feedback=feedback)
    except Exception:
        return {"score": score, "feedback": feedback}
