"""Core corpus visualization functionality.

This module provides the main CorpusVisualizer class for creating 2D
visualizations of document corpora using various dimensionality reduction
techniques.
"""

from __future__ import annotations

import logging
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
from multiview.visualization.reducers import (
    DendrogramReducer,
    DimensionalityReducer,
    TSNEReducer,
)

# Optional scipy import for dendrogram visualization
try:
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


def _safe_local_path_exists(path_str: str) -> bool:
    """Return False instead of raising for invalid or oversized pseudo-paths."""
    try:
        return Path(path_str).exists()
    except OSError:
        return False


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
                if not img_path or not _safe_local_path_exists(img_path):
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
                    logger.warning(f"Failed to load image {img_path}: {e}")
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
            if not img_path or not _safe_local_path_exists(img_path):
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
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

        return canvas

    def plot_dendrogram_with_images(
        self,
        image_paths: list[str],
        figsize: tuple[float, float] = (20, 10),
        title: str | None = None,
        image_size: float = 0.8,
        orientation: str = "top",
        line_color: str = "black",
        line_width: float = 1.5,
        images_per_row: int | None = None,
        num_clusters: int | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create dendrogram visualization with images at leaf nodes.

        This method requires a DendrogramReducer to have been fitted.
        It displays the hierarchical clustering tree with thumbnail images
        positioned at the leaf nodes.

        Args:
            image_paths: List of image file paths for leaf nodes
            figsize: Figure size (width, height)
            title: Plot title
            image_size: Size of image thumbnails relative to spacing (0.0-1.0)
            orientation: Dendrogram orientation ('top', 'bottom', 'left', 'right')
            line_color: Color of dendrogram lines
            line_width: Width of dendrogram lines
            images_per_row: Number of images per row in grid layout. If set, images
                          are arranged in multiple rows for better visibility.
                          None = auto-calculate based on leaf spacing.
            num_clusters: Number of clusters to color-code. If set, cuts the dendrogram
                        at a level to create this many clusters and colors branches/images
                        accordingly. None = auto-calculate (~10% of leaves).

        Returns:
            Tuple of (figure, axes)

        Raises:
            ValueError: If reducer is not a DendrogramReducer or not fitted
            ImportError: If scipy is not available
        """
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for dendrogram visualization. "
                "Install it with: pip install scipy"
            )

        if not isinstance(self.reducer, DendrogramReducer):
            raise ValueError(
                f"Dendrogram visualization requires DendrogramReducer, "
                f"got {type(self.reducer).__name__}"
            )

        if self.reducer.linkage_matrix is None:
            raise ValueError(
                "DendrogramReducer must be fitted before visualization. "
                "Call fit_transform() first."
            )

        # Single dendrogram (always)
        fig, ax = plt.subplots(figsize=figsize)

        # Get number of samples
        n_samples = len(image_paths)

        # Determine number of clusters for coloring
        if num_clusters is None:
            # Auto-calculate: ~10% of samples, capped between 3 and 20
            num_clusters = max(3, min(20, n_samples // 10))

        # Cut the dendrogram to get cluster assignments
        from scipy.cluster.hierarchy import fcluster

        cluster_labels = fcluster(
            self.reducer.linkage_matrix, num_clusters, criterion="maxclust"
        )

        # Calculate color threshold for dendrogram coloring
        # We want to cut at the right height to show num_clusters clusters

        cut_heights = self.reducer.linkage_matrix[:, 2]
        # Find the merge that creates num_clusters clusters
        n_merges = len(self.reducer.linkage_matrix)
        merge_idx = n_merges - num_clusters
        if merge_idx >= 0 and merge_idx < n_merges:
            color_threshold = cut_heights[merge_idx] + 1e-10
        else:
            color_threshold = 0

        logger.debug(
            "Dendrogram params: samples=%s clusters=%s color_threshold=%.4f",
            n_samples,
            num_clusters,
            color_threshold,
        )

        # Plot dendrogram structure with cluster coloring
        dendro = scipy_dendrogram(
            self.reducer.linkage_matrix,
            ax=ax,
            orientation=orientation,
            color_threshold=color_threshold,
            above_threshold_color="#888888",  # Gray for small branches
            no_labels=True,
        )

        # Style the dendrogram lines
        for line in ax.collections:
            line.set_linewidth(line_width)

        # Add images to the dendrogram using helper method with cluster info
        self._add_dendrogram_images_single_row(
            ax,
            dendro,
            image_paths,
            orientation,
            figsize,
            image_size,
            images_per_row,
            cluster_labels,
        )

        # Remove axis ticks and labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Add title
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title(
                f"Hierarchical Clustering Dendrogram ({len(image_paths)} documents)",
                fontsize=14,
                pad=20,
            )

        fig.tight_layout()

        return fig, ax

    def _add_dendrogram_images_single_row(
        self,
        ax,
        dendro: dict,
        image_paths: list[str],
        orientation: str,
        figsize: tuple[float, float],
        image_size: float,
        images_per_row: int | None = None,
        cluster_labels: np.ndarray | None = None,
    ):
        """Helper method to add images to dendrogram in grid layout.

        Args:
            ax: Matplotlib axes to add images to
            dendro: Dendrogram dictionary from scipy
            image_paths: List of image paths (None for leaves to skip)
            orientation: Dendrogram orientation
            figsize: Figure size tuple
            image_size: Image zoom factor
            images_per_row: Number of images per row (None = auto)
            cluster_labels: Cluster assignments for each leaf (1-indexed)
        """
        leaf_order = dendro["leaves"]
        n_leaves = len(leaf_order)

        # Auto-calculate images_per_row if not specified (try to fit without overlap)
        fig_width_inches, fig_height_inches = figsize
        if images_per_row is None:
            # Auto-calculate based on figure width and desired image size
            # Assume ~0.5 inches per image minimum
            images_per_row = max(int(fig_width_inches / 0.5), n_leaves)

        images_per_row = min(
            images_per_row, n_leaves
        )  # Can't have more per row than total

        # Calculate number of rows needed
        import math

        n_rows = math.ceil(n_leaves / images_per_row)

        # Standard thumbnail size
        base_thumbnail_size = 100

        # Calculate image size based on grid layout
        # Each image gets 1/images_per_row of the figure width
        width_per_image_inches = (
            fig_width_inches / images_per_row
        ) * 0.9  # 90% to leave some spacing
        max_zoom = (width_per_image_inches * 72) / base_thumbnail_size
        effective_zoom = min(image_size * max_zoom, max_zoom * 1.2)
        effective_zoom = max(effective_zoom, 0.1)

        logger.debug(
            "Dendrogram layout: leaves=%s rows=%s per_row=%s zoom=%.3f",
            n_leaves,
            n_rows,
            images_per_row,
            effective_zoom,
        )

        # Generate colors based on cluster assignments
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        if cluster_labels is not None:
            # Get unique cluster IDs
            unique_clusters = np.unique(cluster_labels)
            n_clusters = len(unique_clusters)

            # Choose appropriate colormap
            if n_clusters <= 10:
                colormap = cm.get_cmap("tab10")
            elif n_clusters <= 20:
                colormap = cm.get_cmap("tab20")
            else:
                colormap = cm.get_cmap("hsv")

            # Map cluster IDs to colors
            cluster_colors = {}
            for i, cluster_id in enumerate(unique_clusters):
                cluster_colors[cluster_id] = mcolors.to_hex(
                    colormap(i / max(n_clusters - 1, 1))
                )

            logger.debug(f"Dendrogram colors: using {n_clusters} cluster colors")
        else:
            cluster_colors = None

        # Add images in grid layout
        if orientation in ["top", "bottom"]:
            # Get current axis limits
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            y_range = y_limits[1] - y_limits[0]
            x_range = x_limits[1] - x_limits[0]

            # Calculate grid spacing - spread images evenly across x-axis
            grid_x_spacing = x_range / images_per_row
            grid_y_spacing = 0.15 * y_range  # Vertical spacing between rows

            # Starting y position (below dendrogram)
            y_start = y_limits[0] - (0.2 * y_range)

            # Adjust y-limits to make room for all image rows
            if orientation == "top":
                new_y_min = y_start - (n_rows * grid_y_spacing)
                ax.set_ylim(new_y_min, y_limits[1])
            else:
                new_y_max = y_limits[1] + (n_rows * grid_y_spacing) + (0.2 * y_range)
                ax.set_ylim(y_limits[0], new_y_max)

            # Add colored background regions to dendrogram for each cluster
            if cluster_colors is not None:
                from matplotlib.patches import Rectangle

                leaf_spacing = 10  # Standard scipy spacing

                # Group consecutive leaves by cluster
                i = 0
                while i < n_leaves:
                    original_idx = leaf_order[i]
                    current_cluster = cluster_labels[original_idx]

                    # Find the end of this cluster run
                    j = i
                    while (
                        j < n_leaves
                        and cluster_labels[leaf_order[j]] == current_cluster
                    ):
                        j += 1

                    # Draw colored rectangle for this cluster segment
                    x_start = i * leaf_spacing
                    x_end = j * leaf_spacing
                    rect = Rectangle(
                        (x_start, y_limits[0]),
                        x_end - x_start,
                        y_range * 0.05,  # Small height at bottom of dendrogram
                        facecolor=cluster_colors[current_cluster],
                        alpha=0.4,
                        edgecolor="none",
                        zorder=0,  # Behind everything
                    )
                    ax.add_patch(rect)

                    i = j

            # Place images in grid
            for idx in range(n_leaves):
                original_idx = leaf_order[idx]
                if original_idx >= len(image_paths):
                    continue

                img_path = image_paths[original_idx]
                if not img_path or not _safe_local_path_exists(img_path):
                    continue

                # Calculate grid position
                grid_row = idx // images_per_row
                grid_col = idx % images_per_row

                # Get color based on cluster assignment
                if cluster_colors is not None and cluster_labels is not None:
                    border_color = cluster_colors[cluster_labels[original_idx]]
                else:
                    border_color = "gray"

                # Calculate actual x,y position
                x_pos = x_limits[0] + (grid_col + 0.5) * grid_x_spacing
                y_pos = y_start - (grid_row * grid_y_spacing)

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img.thumbnail(
                            (base_thumbnail_size, base_thumbnail_size),
                            Image.Resampling.LANCZOS,
                        )
                        arr = np.asarray(img)

                    image_box = OffsetImage(arr, zoom=effective_zoom)
                    ab = AnnotationBbox(
                        image_box,
                        (x_pos, y_pos),
                        frameon=True,
                        pad=0.0,
                        xycoords="data",
                        box_alignment=(0.5, 0.5),
                        bboxprops={
                            "edgecolor": border_color,
                            "linewidth": 2.5,
                            "facecolor": "white",
                        },
                    )
                    ax.add_artist(ab)

                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    continue

        else:  # left or right orientation - similar grid logic
            # Get current axis limits
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            x_range = x_limits[1] - x_limits[0]
            y_range = y_limits[1] - y_limits[0]

            # For vertical orientation, images_per_row becomes images_per_col
            images_per_col = images_per_row
            n_cols = n_rows  # Swap terminology

            # Calculate grid spacing
            grid_y_spacing = y_range / images_per_col
            grid_x_spacing = 0.15 * x_range

            # Starting x position
            x_start = (
                x_limits[0] - (0.2 * x_range)
                if orientation == "left"
                else x_limits[1] + (0.2 * x_range)
            )

            # Adjust x-limits
            if orientation == "left":
                new_x_min = x_start - (n_cols * grid_x_spacing)
                ax.set_xlim(new_x_min, x_limits[1])
            else:
                new_x_max = x_start + (n_cols * grid_x_spacing)
                ax.set_xlim(x_limits[0], new_x_max)

            # Add colored background regions to dendrogram for each cluster
            if cluster_colors is not None:
                from matplotlib.patches import Rectangle

                leaf_spacing = 10  # Standard scipy spacing

                # Group consecutive leaves by cluster
                i = 0
                while i < n_leaves:
                    original_idx = leaf_order[i]
                    current_cluster = cluster_labels[original_idx]

                    # Find the end of this cluster run
                    j = i
                    while (
                        j < n_leaves
                        and cluster_labels[leaf_order[j]] == current_cluster
                    ):
                        j += 1

                    # Draw colored rectangle for this cluster segment
                    y_start_rect = i * leaf_spacing
                    y_end_rect = j * leaf_spacing
                    rect = Rectangle(
                        (
                            x_limits[0]
                            if orientation == "left"
                            else x_limits[1] - x_range * 0.05,
                            y_start_rect,
                        ),
                        x_range * 0.05,  # Small width at side of dendrogram
                        y_end_rect - y_start_rect,
                        facecolor=cluster_colors[current_cluster],
                        alpha=0.4,
                        edgecolor="none",
                        zorder=0,
                    )
                    ax.add_patch(rect)

                    i = j

            # Place images in grid
            for idx in range(n_leaves):
                original_idx = leaf_order[idx]
                if original_idx >= len(image_paths):
                    continue

                img_path = image_paths[original_idx]
                if not img_path or not _safe_local_path_exists(img_path):
                    continue

                grid_col = idx // images_per_col
                grid_row = idx % images_per_col

                # Get color based on cluster assignment
                if cluster_colors is not None and cluster_labels is not None:
                    border_color = cluster_colors[cluster_labels[original_idx]]
                else:
                    border_color = "gray"

                y_pos = y_limits[0] + (grid_row + 0.5) * grid_y_spacing
                if orientation == "left":
                    x_pos = x_start - (grid_col * grid_x_spacing)
                else:
                    x_pos = x_start + (grid_col * grid_x_spacing)

                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img.thumbnail(
                            (base_thumbnail_size, base_thumbnail_size),
                            Image.Resampling.LANCZOS,
                        )
                        arr = np.asarray(img)

                    image_box = OffsetImage(arr, zoom=effective_zoom)
                    ab = AnnotationBbox(
                        image_box,
                        (x_pos, y_pos),
                        frameon=True,
                        pad=0.0,
                        xycoords="data",
                        box_alignment=(0.5, 0.5),
                        bboxprops={
                            "edgecolor": border_color,
                            "linewidth": 2.5,
                            "facecolor": "white",
                        },
                    )
                    ax.add_artist(ab)

                except Exception as e:
                    logger.warning(f"Failed to load image {img_path}: {e}")
                    continue

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
        logger.debug(f"Projecting {len(embeddings)} embeddings to 2D")
        coords_2d = self.reduce_embeddings(embeddings)

        # Step 2: Generate colors from classes if needed
        legend_labels = None
        if classes and not colors:
            logger.debug("Generating colors from class labels")
            colors, legend_labels = generate_class_colors(classes)
            plot_kwargs["legend_labels"] = legend_labels

        # Step 3: Create plot
        logger.debug("Rendering scatter visualization")
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
            logger.debug(f"Writing visualization: {output_file}")
            fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
            logger.debug(f"Visualization saved: {output_file}")

        return coords_2d, fig, ax
