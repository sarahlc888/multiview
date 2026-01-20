"""Compatibility shim for method result normalization.

`finalize_method_results` is now defined in `multiview.benchmark.evaluation_utils`.
This module remains as a thin re-export for backwards compatibility.
"""

from __future__ import annotations

from multiview.benchmark.evaluation_utils import finalize_method_results

__all__ = ["finalize_method_results"]
