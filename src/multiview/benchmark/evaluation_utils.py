"""Benchmark evaluation helpers.

This module keeps `Benchmark` lean by encapsulating method dispatch and shared
evaluation utilities (triplet dict construction, annotation gating, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.benchmark.methods import (
    evaluate_with_bm25,
    evaluate_with_embeddings,
    evaluate_with_lm_judge_pair,
    evaluate_with_lm_judge_triplet,
)
from multiview.benchmark.result_utils import finalize_method_results
from multiview.benchmark.triplets.utils import build_triplet_dicts

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


def get_annotations_if_required(*, preset: str, task: Any) -> dict | None:
    """Return task annotations only if the preset expects them (else None)."""
    if not preset_requires_annotations(preset):
        return None

    if task.document_annotations is not None:
        logger.info("Using annotations for evaluation (preset requires them)")
        return task.document_annotations

    logger.warning(
        f"Preset '{preset}' requires annotations but task has none. "
        "Evaluation may fail or produce incorrect results."
    )
    return None


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
        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="lmjudge"
        )

        raw = evaluate_with_lm_judge_triplet(
            triplets=build_triplet_dicts(task.documents, task.triplets),
            criterion=task.criterion_name,
            criterion_description=_resolved_criterion_description(task),
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            annotations=annotations,
        )
        return finalize_method_results(raw)

    if method_type == "lm_judge_pair":
        preset = method_config.get("preset", "lmjudge_pair_plaintext_likerthard_gemini")
        annotations = get_annotations_if_required(preset=preset, task=task)
        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="lmjudge_pair"
        )

        raw = evaluate_with_lm_judge_pair(
            triplets=build_triplet_dicts(task.documents, task.triplets),
            criterion=task.criterion_name,
            criterion_description=_resolved_criterion_description(task),
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            annotations=annotations,
        )
        return finalize_method_results(raw)

    if method_type == "embeddings":
        preset = method_config.get("preset", "openai_embedding_small")
        preset_overrides = method_config.get("preset_overrides")
        cache_alias = make_cache_alias(
            task=task, method_config=method_config, default_name="embeddings"
        )

        raw = evaluate_with_embeddings(
            documents=task.documents,
            triplet_ids=task.triplets,
            embedding_preset=preset,
            cache_alias=cache_alias,
            preset_overrides=preset_overrides,
        )
        return finalize_method_results(raw)

    if method_type == "bm25":
        if method_config.get("use_annotations"):
            logger.warning("bm25: use_annotations is not supported; ignoring.")

        raw = evaluate_with_bm25(
            documents=task.documents,
            triplet_ids=task.triplets,
        )
        return finalize_method_results(raw)

    raise ValueError(f"Unknown method type: {method_type}")
