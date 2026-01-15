"""Marker generation utilities for corpus visualization.

This module provides utilities for generating visual markers for corpus
visualization, including color mappings and image-based markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Optional dependencies
try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import GSM8K graph utilities
try:
    from multiview.visualization.gsm8k_graph import (
        GSM8KComputationalGraph,
        parse_gsm8k_document,
    )

    HAS_GSM8K_GRAPH = True
except ImportError:
    HAS_GSM8K_GRAPH = False


def generate_class_colors(
    classes: list[str],
    cmap_name: str = "tab20",
    unlabeled_color: str = "#9aa0a6",
    unlabeled_label: str = "Unlabeled",
) -> tuple[list[str], dict[str, str]]:
    """Generate color mapping for class labels.

    Args:
        classes: List of class labels (one per data point)
        cmap_name: Name of matplotlib colormap to use
        unlabeled_color: Color for unlabeled items
        unlabeled_label: Label to treat as unlabeled

    Returns:
        Tuple of (colors_per_point, legend_dict) where:
        - colors_per_point: List of color strings (one per data point)
        - legend_dict: Dict mapping class name to color
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for color generation. "
            "Install it with: pip install -e '.[viz]'"
        )

    # Get unique classes
    unique_classes = sorted(set(classes))

    # Move unlabeled to front if present
    if unlabeled_label in unique_classes:
        unique_classes = [unlabeled_label] + [
            c for c in unique_classes if c != unlabeled_label
        ]

    # Create colormap
    if len(unique_classes) <= 10:
        cmap = plt.get_cmap("tab10", len(unique_classes))
    elif len(unique_classes) <= 20:
        cmap = plt.get_cmap("tab20", len(unique_classes))
    else:
        # For many classes, use a continuous colormap
        cmap = plt.get_cmap("hsv", len(unique_classes))

    # Build legend dict
    legend_dict = {}
    for idx, class_name in enumerate(unique_classes):
        if class_name == unlabeled_label:
            legend_dict[class_name] = unlabeled_color
        else:
            color_rgba = cmap(idx)
            legend_dict[class_name] = mcolors.rgb2hex(color_rgba[:3])

    # Map each data point to its color
    colors_per_point = [legend_dict[c] for c in classes]

    return colors_per_point, legend_dict


def create_gsm8k_marker_images(
    documents: list[Any],
    output_dir: str,
    show_question: bool = False,
    figsize: tuple[float, float] = (3, 2),
    dpi: int = 100,
    format: str = "png",
    minimal: bool = True,
) -> list[str]:
    """Generate computational graph images for GSM8K documents.

    Args:
        documents: List of GSM8K document dicts or strings
        output_dir: Directory where marker images will be saved
        show_question: Whether to include question text in graphs
        figsize: Size for each graph image (width, height in inches)
        dpi: Resolution for graph images
        format: Image format (png, svg, pdf, jpg)
        minimal: If True, generate graphs without any text labels

    Returns:
        List of paths to generated images (same order as documents)

    Raises:
        ImportError: If GSM8K graph utilities are not available
        RuntimeError: If graph generation fails
    """
    if not HAS_GSM8K_GRAPH:
        raise ImportError(
            "GSM8K graph utilities are required. "
            "Make sure multiview.visualization.gsm8k_graph is available."
        )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = []

    for i, doc in enumerate(documents):
        try:
            # Parse document
            question, answer = parse_gsm8k_document(doc)

            # Create graph
            graph = GSM8KComputationalGraph(question, answer)
            graph.parse()

            # Render graph (with or without text based on minimal flag)
            output_file = output_path / f"marker_{i}"
            path = graph.render(
                str(output_file),
                show_question=show_question,
                format=format,
                figsize=figsize,
                dpi=dpi,
                minimal=minimal,
            )

            image_paths.append(path)

        except Exception as e:
            # If graph generation fails, create a placeholder or skip
            print(f"Warning: Failed to generate graph for document {i}: {e}")
            # Create a simple placeholder image
            if HAS_MATPLOTLIB:
                fig, ax = plt.subplots(figsize=figsize)
                ax.text(
                    0.5,
                    0.5,
                    f"Doc {i}\n(graph error)",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax.axis("off")
                placeholder_path = output_path / f"marker_{i}.{format}"
                fig.savefig(placeholder_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                image_paths.append(str(placeholder_path))
            else:
                # If matplotlib not available, use empty string
                image_paths.append("")

    return image_paths


def load_annotation_classes(
    annotations_file: str,
    class_field: str = "category",
    default_class: str = "Unknown",
) -> list[str]:
    """Load class labels from an annotations file.

    Args:
        annotations_file: Path to JSONL annotations file
        class_field: Field name containing the class label
        default_class: Default class for missing values

    Returns:
        List of class labels (one per document)
    """
    import json

    classes = []

    with open(annotations_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                annotation = json.loads(line)

                # Try to extract class from various possible fields
                class_value = (
                    annotation.get(class_field)
                    or annotation.get("label")
                    or annotation.get("class")
                    or annotation.get("category")
                    or default_class
                )

                # Handle list values (take first element)
                if isinstance(class_value, list):
                    class_value = class_value[0] if class_value else default_class

                classes.append(str(class_value))

            except json.JSONDecodeError:
                classes.append(default_class)

    return classes
