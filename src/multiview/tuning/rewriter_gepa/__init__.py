"""GEPA-based rewriter prompt tuning.

Optimises a document-rewriting prompt so that triplet agreement
(anchor closer to positive than negative) improves after rewriting.
"""

from __future__ import annotations

from multiview.tuning.rewriter_gepa.metric import gepa_metric, metric
from multiview.tuning.rewriter_gepa.module import CheckTripletSimple, RewriteQuery

__all__ = [
    "CheckTripletSimple",
    "RewriteQuery",
    "gepa_metric",
    "metric",
]
