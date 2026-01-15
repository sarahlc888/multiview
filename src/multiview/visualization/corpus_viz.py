"""Core corpus visualization functionality.

This module provides the main CorpusVisualizer class for creating 2D
visualizations of document corpora using various dimensionality reduction
techniques.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Optional dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from multiview.visualization.markers import generate_class_colors
from multiview.visualization.reducers import DimensionalityReducer, TSNEReducer


class CorpusVisualizer:
    """Main visualization class for corpus embeddings."""

    def __init__(self, reducer: DimensionalityReducer | None = None):
        """Initialize corpus visualizer.

        Args:
            reducer: Dimensionality reduction method. If None, uses TSNEReducer
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install -e '.[viz]'"
            )

        self.reducer = reducer or TSNEReducer()

    def reduce_embeddings(
        self, embeddings: list[list[float]] | np.ndarray
    ) -> np.ndarray:
        """Reduce embeddings to 2D coordinates.

        Args:
            embeddings: List or array of embedding vectors

        Returns:
            2D coordinates as numpy array of shape (n_samples, 2)
        """
        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings, dtype=np.float32)

        # Apply dimensionality reduction
        coords_2d = self.reducer.fit_transform(embeddings)

        return coords_2d

    def plot_2d_scatter(
        self,
        coords_2d: np.ndarray,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        image_paths: list[str] | None = None,
        marker_size: int = 50,
        alpha: float = 0.7,
        figsize: tuple[float, float] = (12, 8),
        title: str | None = None,
        legend_labels: dict[str, str] | None = None,
        show_text_labels: bool = False,
        image_zoom: float = 0.05,
        unlabeled_alpha: float = 0.35,
        unlabeled_color: str = "#9aa0a6",
        unlabeled_label: str = "Unlabeled",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create 2D scatter plot with various marker types.

        Marker types (in order of precedence):
        1. images: Use image files as markers
        2. colors: Color-coded scatter points
        3. labels: Text annotations at each point

        Args:
            coords_2d: 2D coordinates of shape (n_samples, 2)
            labels: Text labels for annotations
            colors: Colors for each point
            image_paths: Paths to images to use as markers
            marker_size: Size of scatter markers
            alpha: Transparency of markers
            figsize: Figure size (width, height)
            title: Plot title
            legend_labels: Dict mapping class name to color for legend
            show_text_labels: Whether to show text labels
            image_zoom: Zoom factor for image markers
            unlabeled_alpha: Transparency for unlabeled points
            unlabeled_color: Color for unlabeled points
            unlabeled_label: Label to treat as unlabeled

        Returns:
            Tuple of (figure, axes)
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Set axis limits with padding
        x_min, x_max = float(coords_2d[:, 0].min()), float(coords_2d[:, 0].max())
        y_min, y_max = float(coords_2d[:, 1].min()), float(coords_2d[:, 1].max())
        x_pad = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # Plot based on marker type
        images_drawn = 0

        if image_paths:
            # Use images as markers
            for idx, (x_val, y_val) in enumerate(coords_2d):
                if idx >= len(image_paths):
                    break

                img_path = image_paths[idx]
                if not img_path or not Path(img_path).exists():
                    # Plot a small point as fallback
                    ax.scatter(
                        [x_val],
                        [y_val],
                        s=marker_size // 4,
                        alpha=alpha * 0.5,
                        color="gray",
                    )
                    continue

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img.thumbnail((96, 96), Image.Resampling.LANCZOS)
                        arr = np.asarray(img)

                    image_box = OffsetImage(arr, zoom=image_zoom)
                    ab = AnnotationBbox(
                        image_box,
                        (x_val, y_val),
                        frameon=True,
                        pad=0.1,
                        xycoords="data",
                        box_alignment=(0.5, 0.5),
                        bboxprops={
                            "edgecolor": "black",
                            "linewidth": 1.0,
                            "facecolor": "none",
                        },
                    )
                    ax.add_artist(ab)
                    images_drawn += 1

                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    # Plot a small point as fallback
                    ax.scatter(
                        [x_val],
                        [y_val],
                        s=marker_size // 4,
                        alpha=alpha * 0.5,
                        color="gray",
                    )

        elif colors:
            # Color-coded scatter plot
            if legend_labels:
                # Plot by class for legend
                for class_name, class_color in legend_labels.items():
                    mask = np.array([c == class_color for c in colors])
                    if not np.any(mask):
                        continue

                    count = np.sum(mask)
                    if class_name == unlabeled_label:
                        alpha_val = unlabeled_alpha
                    else:
                        alpha_val = alpha

                    ax.scatter(
                        coords_2d[mask, 0],
                        coords_2d[mask, 1],
                        s=marker_size,
                        alpha=alpha_val,
                        color=class_color,
                        label=f"{class_name} ({count})",
                    )
            else:
                # Simple scatter with colors
                ax.scatter(
                    coords_2d[:, 0],
                    coords_2d[:, 1],
                    s=marker_size,
                    alpha=alpha,
                    c=colors,
                )
        else:
            # Default scatter plot
            ax.scatter(
                coords_2d[:, 0],
                coords_2d[:, 1],
                s=marker_size,
                alpha=alpha,
                color="blue",
            )

        # Add text labels if requested
        if show_text_labels and labels:
            for idx, (x_val, y_val) in enumerate(coords_2d):
                if idx >= len(labels):
                    break
                if labels[idx]:
                    ax.annotate(
                        str(labels[idx]),
                        (x_val, y_val),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=6,
                        color="white",
                        bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.5},
                    )

        # Add title and labels
        if title:
            ax.set_title(title)
        else:
            reducer_name = self.reducer.__class__.__name__.replace("Reducer", "")
            ax.set_title(f"{reducer_name} Visualization ({len(coords_2d)} documents)")

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        # Add legend if we have class labels
        if legend_labels and colors:
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()

        return fig, ax

    def create_grid_composite(
        self,
        image_paths: list[str],
        assignments: list[int],
        grid_rows: int,
        grid_cols: int,
        tile_size: int = 200,
        padding: int = 10,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """Create a grid-based composite image with SOM assignments.

        Args:
            image_paths: List of image file paths
            assignments: Grid node assignment for each image (from SOM.assign_unique_nodes)
            grid_rows: Number of rows in the grid
            grid_cols: Number of columns in the grid
            tile_size: Size of each tile in pixels
            padding: Padding between tiles in pixels
            background_color: RGB tuple for background

        Returns:
            PIL Image with grid composite
        """
        # Calculate canvas dimensions
        width = grid_cols * tile_size + (grid_cols + 1) * padding
        height = grid_rows * tile_size + (grid_rows + 1) * padding

        # Create blank canvas
        canvas = Image.new("RGB", (width, height), color=background_color)

        # Place each image at its assigned grid position
        for img_path, node_idx in zip(image_paths, assignments, strict=False):
            if not img_path or not Path(img_path).exists():
                continue

            # Calculate grid position
            row = node_idx // grid_cols
            col = node_idx % grid_cols

            # Calculate pixel coordinates
            x = padding + col * (tile_size + padding)
            y = padding + row * (tile_size + padding)

            try:
                # Load and prepare tile
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)

                    # Create tile with centered image
                    tile = Image.new(
                        "RGB", (tile_size, tile_size), color=background_color
                    )
                    offset = (
                        (tile_size - img.width) // 2,
                        (tile_size - img.height) // 2,
                    )
                    tile.paste(img, offset)

                    # Paste tile onto canvas
                    canvas.paste(tile, (x, y))

            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                continue

        return canvas

    def visualize_corpus(
        self,
        embeddings: list[list[float]] | np.ndarray,
        documents: list[Any] | None = None,
        labels: list[str] | None = None,
        colors: list[str] | None = None,
        classes: list[str] | None = None,
        image_paths: list[str] | None = None,
        output_path: str | None = None,
        format: str = "png",
        dpi: int = 150,
        **plot_kwargs,
    ) -> tuple[np.ndarray, plt.Figure, plt.Axes]:
        """End-to-end corpus visualization pipeline.

        Args:
            embeddings: List or array of embedding vectors
            documents: Optional document objects for reference
            labels: Text labels for each point
            colors: Explicit colors for each point
            classes: Class labels (auto-generates color mapping)
            image_paths: Paths to images to use as markers
            output_path: Where to save (without extension)
            format: Output format (png, svg, pdf)
            dpi: Resolution for raster formats
            **plot_kwargs: Additional args for plot_2d_scatter

        Returns:
            Tuple of (coords_2d, fig, ax)
        """
        # Step 1: Reduce embeddings to 2D
        print(f"Reducing {len(embeddings)} embeddings to 2D...")
        coords_2d = self.reduce_embeddings(embeddings)

        # Step 2: Generate colors from classes if needed
        legend_labels = None
        if classes and not colors:
            print("Generating colors from class labels...")
            colors, legend_labels = generate_class_colors(classes)
            plot_kwargs["legend_labels"] = legend_labels

        # Step 3: Create plot
        print("Creating visualization...")
        fig, ax = self.plot_2d_scatter(
            coords_2d,
            labels=labels,
            colors=colors,
            image_paths=image_paths,
            **plot_kwargs,
        )

        # Step 4: Save if output_path provided
        if output_path:
            output_file = f"{output_path}.{format}"
            print(f"Saving to {output_file}...")
            fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
            print(f"Saved visualization: {output_file}")

        return coords_2d, fig, ax
