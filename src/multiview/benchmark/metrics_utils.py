"""Compatibility shim for metrics helpers.

These helpers were merged into `multiview.benchmark.evaluation_utils` to keep
evaluation logic colocated. This module remains as a thin re-export to avoid
churn in call sites/tests.
"""

from __future__ import annotations

from multiview.benchmark.evaluation_utils import (
    metrics_from_correctness,
    metrics_from_outcomes,
    outcomes_from_pair_scores,
)

__all__ = [
    "metrics_from_correctness",
    "metrics_from_outcomes",
    "outcomes_from_pair_scores",
]
