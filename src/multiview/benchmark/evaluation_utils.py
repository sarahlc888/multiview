"""Benchmark evaluation helpers.

This module keeps `Benchmark` lean by encapsulating method dispatch and shared
evaluation utilities (triplet dict construction, annotation gating, etc.).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from multiview.benchmark.triplets.utils import build_triplet_dicts
from multiview.eval import (
    evaluate_with_bm25,
    evaluate_with_embeddings,
    evaluate_with_in_one_word,
    evaluate_with_lm_judge_pair,
    evaluate_with_lm_judge_triplet,
    evaluate_with_query_expansion,
    evaluate_with_query_expansion_bm25,
    evaluate_with_reranker,
)

logger = logging.getLogger(__name__)


def _resolved_criterion_description(task: Any) -> str | None:
    """Single source of truth for criterion description used in evaluation.

    Prefer Task's metadata-aware resolver when available.
    """
    resolver = getattr(task, "_resolved_criterion_description", None)
    if callable(resolver):
        return resolver()
    return getattr(task, "config", {}).get("criterion_description")


def preset_requires_annotations(preset: str) -> bool:
    return "with_annotation" in preset.lower()


def has_rich_annotations(annotations: list[dict] | None) -> bool:
    """Check if annotations contain rich information (summaries, tags, etc.).

    Returns:
        True if annotations have rich fields beyond just criterion_value
        False if annotations only have criterion_value (simple precomputed annotations)
    """
    if not annotations or len(annotations) == 0:
        return False

    # Check first annotation for rich fields
    first_ann = annotations[0]

    # Rich annotations have at least one of: summary, tags, category
    has_summary = "summary" in first_ann and first_ann.get("summary")
    has_tags = "tags" in first_ann and first_ann.get("tags")
    has_category = "category" in first_ann and first_ann.get("category")

    return has_summary or has_tags or has_category


def has_category_schemas(annotations: list[dict] | None) -> bool:
    """Check if annotations contain category schema information.

    Returns:
        True if annotations have category_schema field
        False otherwise
    """
    if not annotations or len(annotations) == 0:
        return False

    # Check first annotation for category_schema
    first_ann = annotations[0]
    return "category_schema" in first_ann and first_ann.get("category_schema")


def get_annotations_if_required(*, preset: str, task: Any) -> dict | None:
    """Return task annotations only if the preset expects them (else None).

    Also checks if annotations are rich enough for the preset - simple precomputed
    annotations (only criterion_value) are not sufficient for presets that need
    summaries/tags.
    """
    if not preset_requires_annotations(preset):
        return None

    if task.document_annotations is None:
        logger.warning(
            f"Preset '{preset}' requires annotations but task has none. "
            "Skipping this method."
        )
        return None

    # Check if annotations are rich enough
    if not has_rich_annotations(task.document_annotations):
        logger.warning(
            f"Preset '{preset}' requires rich annotations (summaries/tags), "
            f"but task only has simple criterion values. "
            f"Skipping this method. Use presets without 'with_annotation' for simple annotations."
        )
        return None

    logger.info("Using annotations for evaluation (preset requires them)")
    return task.document_annotations


def make_cache_alias(*, task: Any, method_config: dict, default_name: str) -> str:
    return f"{task.get_task_name()}_eval_{method_config.get('name', default_name)}"


def evaluate_method(
    *, method_type: str, task: Any, method_config: dict
) -> dict[str, Any]:
    """Dispatch evaluation for a single method config on a single task."""
    if task.triplets is None:
        raise ValueError("Task has no triplets.")

    if method_type == "lm_judge_triplet":
        preset = method_config.get(
            "preset", "lmjudge_triplet_plaintext_binaryhard_gemini"
        )
        annotations = get_annotations_if_required(preset=preset, task=task)

        # Skip if preset requires annotations but they're not available/suitable
        if preset_requires_annotations(preset) and annotations is None:
            logger.warning("Skipping method due to missing/insufficient annotations")
            return {"skipped": True, "reason": "Missing or insufficient annotations"}

        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="lmjudge"
        )

        # Extract text from documents (handle both string and dict formats)
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_lm_judge_triplet(
            triplets=build_triplet_dicts(document_texts, task.triplets),
            criterion=task.criterion_name,
            criterion_description=_resolved_criterion_description(task),
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            annotations=annotations,
        )
        return finalize_method_results(raw)

    if method_type == "lm_judge_pair":
        preset = method_config.get("preset", "lmjudge_pair_plaintext_likerthard_gemini")
        annotations = get_annotations_if_required(preset=preset, task=task)

        # Skip if preset requires annotations but they're not available/suitable
        if preset_requires_annotations(preset) and annotations is None:
            logger.warning("Skipping method due to missing/insufficient annotations")
            return {"skipped": True, "reason": "Missing or insufficient annotations"}

        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="lmjudge_pair"
        )

        # Extract text from documents (handle both string and dict formats)
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_lm_judge_pair(
            triplets=build_triplet_dicts(document_texts, task.triplets),
            criterion=task.criterion_name,
            criterion_description=_resolved_criterion_description(task),
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            annotations=annotations,
        )
        return finalize_method_results(raw)

    if method_type == "embeddings":
        preset = method_config.get("preset", "openai_embedding_small")
        preset_overrides = method_config.get("preset_overrides", {})

        # Get criterion metadata for embed_instr
        criterion_metadata = (
            task.document_set.get_criterion_metadata(task.criterion_name) or {}
        )

        # Support embed_instr from criterion metadata
        # For symmetric retrieval (doc-to-doc), use embed_doc_instr_template
        # Clear embed_query_instr_template to avoid double-prepending
        # Priority: method_config > criterion_metadata
        if "embed_instr" not in method_config and "embed_instr" in criterion_metadata:
            if not isinstance(preset_overrides, dict):
                preset_overrides = {}
            # For symmetric document comparison, use embed_doc_instr_template
            # Clear any query instruction from preset to prevent double-prepending
            preset_overrides["embed_doc_instr_template"] = criterion_metadata[
                "embed_instr"
            ]
            preset_overrides["embed_query_instr_template"] = None

        # Support convenient embed_instr shorthand in method config (always applies if specified)
        if "embed_instr" in method_config:
            if not isinstance(preset_overrides, dict):
                preset_overrides = {}
            # For symmetric document comparison, use embed_doc_instr_template
            # Clear any query instruction from preset to prevent double-prepending
            preset_overrides["embed_doc_instr_template"] = method_config["embed_instr"]
            preset_overrides["embed_query_instr_template"] = None

        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="embeddings"
        )

        # Extract text from documents (handle both string and dict formats)
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_embeddings(
            documents=document_texts,
            triplet_ids=task.triplets,
            embedding_preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides if preset_overrides else None,
            criterion=task.criterion_name,
        )
        return finalize_method_results(raw)

    if method_type == "reranker":
        preset = method_config.get("preset", "qwen3_reranker_8b")
        instruction = method_config.get("instruction")
        preset_overrides = method_config.get("preset_overrides")

        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="reranker"
        )

        # Extract text from documents (handle both string and dict formats)
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_reranker(
            documents=document_texts,
            triplet_ids=task.triplets,
            reranker_preset=preset,
            instruction=instruction,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides,
        )
        return finalize_method_results(raw)

    if method_type == "bm25":
        if method_config.get("use_annotations"):
            logger.warning("bm25: use_annotations is not supported; ignoring.")

        # Extract text from documents (handle both string and dict formats)
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_bm25(
            documents=document_texts,
            triplet_ids=task.triplets,
        )
        return finalize_method_results(raw)

    if method_type == "query_expansion":
        retrieval_method = method_config.get("retrieval_method", "bm25")
        summary_preset = method_config.get(
            "summary_preset", "query_expansion_summary_gemini"
        )
        embedding_preset = method_config.get(
            "embedding_preset", "openai_embedding_small"
        )
        preset_overrides = method_config.get("preset_overrides")

        cache_alias = make_cache_alias(
            task=task,
            method_config=method_config,
            default_name=f"qe_{retrieval_method}",
        )

        # Extract text from documents
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_query_expansion(
            documents=document_texts,
            triplet_ids=task.triplets,
            criterion=task.criterion_name,
            criterion_description=_resolved_criterion_description(task),
            retrieval_method=retrieval_method,
            summary_preset=summary_preset,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides,
        )
        return finalize_method_results(raw)

    # Backwards compatibility for query_expansion_bm25
    if method_type == "query_expansion_bm25":
        summary_preset = method_config.get(
            "summary_preset", "query_expansion_summary_gemini"
        )

        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="qe_bm25"
        )

        # Extract text from documents
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_query_expansion_bm25(
            documents=document_texts,
            triplet_ids=task.triplets,
            criterion=task.criterion_name,
            criterion_description=_resolved_criterion_description(task),
            summary_preset=summary_preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
        )
        return finalize_method_results(raw)

    if method_type == "in_one_word":
        preset = method_config.get("preset", "inoneword_hf_qwen3_8b")

        # Validate annotations exist
        if task.document_annotations is None:
            logger.error("in_one_word method requires annotations but task has none")
            return {"skipped": True, "reason": "Missing annotations"}

        # Validate annotations have category schemas
        if not has_category_schemas(task.document_annotations):
            logger.error("in_one_word method requires category schemas in annotations")
            return {"skipped": True, "reason": "Missing category schemas"}

        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="inoneword"
        )

        # Extract text from documents
        document_texts = [
            task.document_set.get_document_text(doc) for doc in task.documents
        ]

        raw = evaluate_with_in_one_word(
            documents=document_texts,
            triplet_ids=task.triplets,
            annotations=task.document_annotations,
            preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=method_config.get("preset_overrides"),
        )
        return finalize_method_results(raw)

    raise ValueError(f"Unknown method type: {method_type}")


#
# Metrics + normalization helpers (merged here to keep `Benchmark` lean).
#


def metrics_from_correctness(
    correct: Sequence[bool],
    *,
    is_tie: Sequence[bool] | None = None,
    exclude_ties: bool = True,
) -> dict[str, Any]:
    """Aggregate correctness booleans into the canonical metrics dict."""
    n_total = len(correct)

    if is_tie is not None and len(is_tie) != n_total:
        raise ValueError(
            "correct and is_tie must have the same length: "
            f"{n_total} != {len(is_tie)}"
        )

    if is_tie is None:
        n_correct = sum(1 for c in correct if c)
        n_incorrect = n_total - n_correct
        n_ties = 0
        accuracy = (n_correct / n_total) if n_total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_ties": n_ties,
            "n_total": n_total,
        }

    n_ties = sum(1 for t in is_tie if t)

    if exclude_ties:
        n_correct = sum(1 for c, t in zip(correct, is_tie, strict=False) if c and not t)
        n_incorrect = sum(
            1 for c, t in zip(correct, is_tie, strict=False) if (not c) and (not t)
        )
        n_judged = n_correct + n_incorrect
        accuracy = (n_correct / n_judged) if n_judged > 0 else 0.0
    else:
        n_correct = sum(1 for c in correct if c)
        n_incorrect = n_total - n_correct - n_ties
        accuracy = (n_correct / n_total) if n_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_ties": n_ties,
        "n_total": n_total,
    }


def outcomes_from_pair_scores(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    tie_tol: float = 0.0,
) -> list[int]:
    """Convert pairwise scores into standardized triplet outcomes."""
    if len(positive_scores) != len(negative_scores):
        raise ValueError(
            "positive_scores and negative_scores must have the same length: "
            f"{len(positive_scores)} != {len(negative_scores)}"
        )
    if tie_tol < 0:
        raise ValueError(f"tie_tol must be >= 0, got {tie_tol}")

    outcomes: list[int] = []
    for pos, neg in zip(positive_scores, negative_scores, strict=False):
        if pos > neg + tie_tol:
            outcomes.append(1)
        elif neg > pos + tie_tol:
            outcomes.append(-1)
        else:
            outcomes.append(0)
    return outcomes


def metrics_from_outcomes(outcomes: Sequence[int]) -> dict[str, Any]:
    """Aggregate triplet outcomes into the canonical metrics dict."""
    n_total = len(outcomes)
    correct = [o == 1 for o in outcomes]
    is_tie = [o == 0 for o in outcomes]
    metrics = metrics_from_correctness(correct, is_tie=is_tie, exclude_ties=True)
    metrics["n_total"] = n_total
    return metrics


def finalize_method_results(method_results: dict[str, Any]) -> dict[str, Any]:
    """Normalize method results to include standard metrics + correctness vectors."""
    if "accuracy" in method_results and "n_total" in method_results:
        return method_results

    if "correct" in method_results:
        correct = method_results["correct"]
        is_tie = method_results.get("is_tie")
        metrics = metrics_from_correctness(correct, is_tie=is_tie, exclude_ties=True)
        return {**metrics, **method_results}

    if "outcomes" in method_results:
        outcomes = method_results["outcomes"]
        correct = [o == 1 for o in outcomes]
        is_tie = [o == 0 for o in outcomes]
        metrics = metrics_from_correctness(correct, is_tie=is_tie, exclude_ties=True)
        return {**metrics, "correct": correct, "is_tie": is_tie, **method_results}

    if "positive_scores" in method_results and "negative_scores" in method_results:
        outcomes = outcomes_from_pair_scores(
            method_results["positive_scores"],
            method_results["negative_scores"],
        )
        correct = [o == 1 for o in outcomes]
        is_tie = [o == 0 for o in outcomes]
        metrics = metrics_from_correctness(correct, is_tie=is_tie, exclude_ties=True)
        return {**metrics, "correct": correct, "is_tie": is_tie, **method_results}

    return method_results


__all__ = [
    "evaluate_method",
    "finalize_method_results",
    "get_annotations_if_required",
    "make_cache_alias",
    "metrics_from_correctness",
    "metrics_from_outcomes",
    "outcomes_from_pair_scores",
    "preset_requires_annotations",
]
