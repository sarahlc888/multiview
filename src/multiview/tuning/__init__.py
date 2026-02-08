"""Prompt tuning for multiview models.

Subpackages:
- rewriter_gepa: GEPA-based rewriter prompt optimization
- landmark_gepa: GEPA-based landmark/query prompt optimization
- embedding_finetune: (future) standard embedding model finetuning
"""

from __future__ import annotations

from multiview.tuning.data_loading import (
    load_triplets_as_dspy_examples,
    load_triplets_from_benchmark,
)
from multiview.tuning.utils import (
    LearningCurveTracker,
    extract_score,
    save_detailed_results,
    save_prompt_details,
    save_readout_file,
    setup_logging,
    setup_proposal_prompt_logging,
)

__all__ = [
    "LearningCurveTracker",
    "extract_score",
    "load_triplets_as_dspy_examples",
    "load_triplets_from_benchmark",
    "save_detailed_results",
    "save_prompt_details",
    "save_readout_file",
    "setup_logging",
    "setup_proposal_prompt_logging",
]
