"""Helpers for normalizing per-method evaluation outputs.

Methods should return *raw per-triplet signals* (e.g., outcomes or score pairs).
This module converts those into the standard metrics + correctness vectors.
"""

from __future__ import annotations

from typing import Any

from multiview.benchmark.metrics_utils import (
    metrics_from_correctness,
    outcomes_from_pair_scores,
)


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
