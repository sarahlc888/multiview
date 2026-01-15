"""Corpus visualization module for multiview.

This module provides tools for visualizing document corpora in 2D space
using various dimensionality reduction techniques.
"""

from multiview.visualization.corpus_viz import CorpusVisualizer
from multiview.visualization.markers import (
    create_gsm8k_marker_images,
    generate_class_colors,
    load_annotation_classes,
)
from multiview.visualization.reducers import (
    DimensionalityReducer,
    PCAReducer,
    SOMReducer,
    TSNEReducer,
    UMAPReducer,
)

__all__ = [
    "CorpusVisualizer",
    "DimensionalityReducer",
    "TSNEReducer",
    "PCAReducer",
    "UMAPReducer",
    "SOMReducer",
    "generate_class_colors",
    "create_gsm8k_marker_images",
    "load_annotation_classes",
]
