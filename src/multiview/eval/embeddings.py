"""Embedding-based evaluation method.

This module implements evaluation methods that use embedding models to compute
similarity scores via cosine similarity, then evaluate triplet accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.eval.generation_utils import validate_criterion_description
from multiview.eval.similarity import compute_similarity
from multiview.inference.inference import run_inference
from multiview.inference.presets import get_preset

logger = logging.getLogger(__name__)

# Providers in this project that only support text embeddings in the current codepath.
TEXT_ONLY_EMBEDDING_PROVIDERS = {
    "openai_embedding",
    "hf_embedding",
    "voyage_embedding",
    "gemini_embedding",
    "hf_local_colbert",
    "hf_local_hidden_state",
}


def _is_image_only_document(text: str, image: str | None) -> bool:
    """Return True when a document has no usable text for text embeddings."""
    normalized = text.strip().lower()
    is_placeholder = normalized == "<image>"
    return bool(image) and (not normalized or is_placeholder)


def _has_no_usable_text(text: str) -> bool:
    """Return True when text is empty or only the image placeholder marker."""
    normalized = text.strip().lower()
    return not normalized or normalized == "<image>"


def evaluate_with_embeddings(
    documents: list[str | dict],
    triplet_ids: list[tuple[int, int, int]],
    embedding_preset: str = "openai_embedding_small",
    cache_alias: str | None = None,
    run_name: str | None = None,
    preset_overrides: dict | None = None,
    criterion: str | None = None,
    criterion_description: str | None = None,
) -> dict[str, Any]:
    """Evaluate triplets using embedding-based cosine similarity.

    Args:
        documents: List of documents (text strings or dicts with optional image_path)
        triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
        embedding_preset: Inference preset to use
        cache_alias: Optional cache identifier
        run_name: Optional experiment/run name for cache organization
        preset_overrides: Optional preset configuration overrides
        criterion: Criterion name (required for instruction-tuned embeddings like instr_hf_qwen3_embedding_8b)
        criterion_description: Criterion description (used in embedding instructions for better context)
    """
    if not triplet_ids:
        logger.warning("No triplets provided for evaluation")
        return {
            "positive_scores": [],
            "negative_scores": [],
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
            "triplet_logs": [],
        }

    logger.info(f"Evaluating {len(triplet_ids)} triplets with embeddings")
    logger.info(f"Using preset: {embedding_preset}")
    logger.info(f"Computing embeddings for {len(documents)} documents")

    preset_config = get_preset(embedding_preset)
    is_text_only_provider = preset_config.provider in TEXT_ONLY_EMBEDDING_PROVIDERS

    # Normalize docs into text/image channels.
    # For multimodal providers, image-only docs are represented as "<image>".
    # For text-only providers, image-only docs are rejected early.
    texts: list[str] = []
    images: list[str | None] = []
    for doc in documents:
        if isinstance(doc, dict):
            text = doc.get("text", "")
            image = doc.get("image_path")
        else:
            text = doc
            image = None
        texts.append(text)
        images.append(image)

    if is_text_only_provider:
        image_only_indices = [
            i
            for i, (text, image) in enumerate(zip(texts, images, strict=False))
            if _has_no_usable_text(text)
        ]
        if image_only_indices:
            preview = image_only_indices[:10]
            raise ValueError(
                f"Embedding preset '{embedding_preset}' uses text-only provider "
                f"'{preset_config.provider}' and cannot embed image-only documents. "
                f"Found {len(image_only_indices)} image-only document(s), "
                f"sample indices: {preview}."
            )
    else:
        for i, (text, image) in enumerate(zip(texts, images, strict=False)):
            if _is_image_only_document(text, image):
                texts[i] = "<image>"

    inference_kwargs = {"verbose": False}
    if preset_overrides:
        inference_kwargs.update(preset_overrides)

    # Build inputs - include criterion and required description for criterion-aware prompts.
    inputs = {"document": texts}
    if any(img is not None for img in images) and not is_text_only_provider:
        inputs["images"] = images
    if criterion is not None:
        criterion_description = validate_criterion_description(
            criterion=criterion,
            criterion_description=criterion_description,
            context=f"embedding preset '{embedding_preset}'",
        )
        inputs["criterion"] = criterion
        inputs["criterion_description"] = criterion_description

    embeddings = run_inference(
        inputs=inputs,
        config=embedding_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )

    positive_scores: list[float] = []
    negative_scores: list[float] = []
    triplet_logs: list[dict[str, Any]] = []

    for i, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        anchor_emb = embeddings[anchor_id]
        positive_emb = embeddings[positive_id]
        negative_emb = embeddings[negative_id]

        pos_score = compute_similarity(anchor_emb, positive_emb)
        neg_score = compute_similarity(anchor_emb, negative_emb)

        positive_scores.append(pos_score)
        negative_scores.append(neg_score)

        if pos_score > neg_score:
            outcome = 1
        elif neg_score > pos_score:
            outcome = -1
        else:
            outcome = 0

        triplet_logs.append(
            {
                "triplet_idx": i,
                "method_type": "embeddings",
                "embedding_preset": embedding_preset,
                "cache_alias": cache_alias,
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
                "anchor": texts[anchor_id],
                "positive": texts[positive_id],
                "negative": texts[negative_id],
                "positive_score": pos_score,
                "negative_score": neg_score,
                "outcome": outcome,
                "correct": outcome == 1,
                "is_tie": outcome == 0,
            }
        )

    avg_positive_score = (
        sum(positive_scores) / len(positive_scores) if positive_scores else 0.0
    )
    avg_negative_score = (
        sum(negative_scores) / len(negative_scores) if negative_scores else 0.0
    )

    logger.info(
        f"Average positive score: {avg_positive_score:.3f}, Average negative score: {avg_negative_score:.3f}"
    )

    return {
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "avg_positive_score": avg_positive_score,
        "avg_negative_score": avg_negative_score,
        "triplet_logs": triplet_logs,
        "embeddings": embeddings,  # Add embeddings array to results
    }
