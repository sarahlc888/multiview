"""GEPA-based landmark / query-generation prompt tuning.

Optimises a query generation prompt so that the produced queries yield
embeddings with better triplet agreement.
"""

from __future__ import annotations

from multiview.tuning.landmark_gepa.module import (
    DSPyQueryGenerator,
    QueryGeneratorModule,
    parse_queries,
)

__all__ = [
    "DSPyQueryGenerator",
    "QueryGeneratorModule",
    "parse_queries",
]
