"""Shared helpers for triplet evaluation metrics.

We standardize triplet outcomes as ints:
  -  1: correct (positive > negative)
  - -1: incorrect (negative > positive)
  -  0: tie (positive == negative within tolerance)

This matches LM triplet-judge outputs and lets methods share aggregation logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


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
        n_correct = sum(1 for c, t in zip(correct, is_tie, strict=True) if c and not t)
        n_incorrect = sum(
            1 for c, t in zip(correct, is_tie, strict=True) if (not c) and (not t)
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
    for pos, neg in zip(positive_scores, negative_scores, strict=True):
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
