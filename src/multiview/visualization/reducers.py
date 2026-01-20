"""Dimensionality reduction methods for corpus visualization.

This module provides various dimensionality reduction techniques for
visualizing high-dimensional embeddings in 2D space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Optional dependencies with graceful degradation
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class DimensionalityReducer(ABC):
    """Abstract base class for dimensionality reduction methods."""

    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings to 2D coordinates.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            2D coordinates of shape (n_samples, 2)
        """
        pass


class TSNEReducer(DimensionalityReducer):
    """t-SNE dimensionality reduction using scikit-learn."""

    def __init__(
        self,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize t-SNE reducer.

        Args:
            perplexity: The perplexity parameter (5-50 typical range)
            max_iter: Maximum number of iterations for optimization
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments passed to TSNE
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for t-SNE. "
                "Install it with: pip install -e '.[viz]'"
            )

        self.perplexity = perplexity
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply t-SNE to reduce embeddings to 2D.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            2D coordinates of shape (n_samples, 2)
        """
        # Adjust perplexity if needed
        n_samples = embeddings.shape[0]
        perplexity = min(self.perplexity, (n_samples - 1) / 3.0)
        if perplexity < 5:
            perplexity = max(5, n_samples // 3)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **self.kwargs,
        )
        return tsne.fit_transform(embeddings)


class PCAReducer(DimensionalityReducer):
    """PCA dimensionality reduction with whitening and standardization.

    Following the pattern from the reference code, this applies
    StandardScaler followed by PCA with whitening.
    """

    def __init__(
        self,
        n_components: int = 2,
        whiten: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize PCA reducer.

        Args:
            n_components: Number of components (default 2 for 2D)
            whiten: Whether to whiten the data
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments passed to PCA
        """
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for PCA. "
                "Install it with: pip install -e '.[viz]'"
            )

        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        self.explained_variance_ratio_ = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA with standardization to reduce embeddings to 2D.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            2D coordinates of shape (n_samples, 2)
        """
        # Standardize the embeddings
        whitened = self.scaler.fit_transform(embeddings)

        # Apply PCA with whitening
        pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
            **self.kwargs,
        )
        coords = pca.fit_transform(whitened)

        # Store explained variance for later reference
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

        # Return first 2 components
        return coords[:, :2]


class UMAPReducer(DimensionalityReducer):
    """UMAP dimensionality reduction using umap-learn."""

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
        **kwargs,
    ):
        """Initialize UMAP reducer.

        Args:
            n_neighbors: Number of neighbors to consider (5-50 typical)
            min_dist: Minimum distance in embedding space (0.0-0.99)
            metric: Distance metric to use
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments passed to UMAP
        """
        if not HAS_UMAP:
            raise ImportError(
                "umap-learn is required for UMAP. "
                "Install it with: pip install -e '.[viz]'"
            )

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.kwargs = kwargs

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP to reduce embeddings to 2D.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            2D coordinates of shape (n_samples, 2)
        """
        # Adjust n_neighbors if needed
        n_samples = embeddings.shape[0]
        n_neighbors = min(self.n_neighbors, n_samples - 1)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            **self.kwargs,
        )
        return reducer.fit_transform(embeddings)


class SOMReducer(DimensionalityReducer):
    """Self-organizing map (SOM) for dimensionality reduction.

    This custom implementation maps high-dimensional embeddings to a 2D grid
    using competitive learning. Adapted from the reference code.
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (20, 20),
        learning_rate: float = 0.5,
        sigma: float | None = None,
        iterations: int = 1000,
        random_state: int = 42,
        unique_assignment: bool = True,
    ):
        """Initialize SOM reducer.

        Args:
            grid_size: Tuple of (rows, cols) for the SOM grid
            learning_rate: Initial learning rate (0.1-1.0)
            sigma: Initial neighborhood radius (defaults to max(rows, cols) / 2)
            iterations: Number of training iterations
            random_state: Random seed for reproducibility
            unique_assignment: If True, assign each sample to a unique grid node
                (avoids overlaps and blank spaces, but requires n_samples <= grid_size).
                If False, allows multiple samples per node (classic BMU assignment).
        """
        self.rows, self.cols = grid_size
        self.learning_rate = learning_rate
        self.sigma = sigma or max(self.rows, self.cols) / 2.0
        self.iterations = iterations
        self.random_state = random_state
        self.unique_assignment = unique_assignment
        self.weights = None
        self._coords = None
        self.rng = np.random.default_rng(random_state)

    def _build_coords(self) -> np.ndarray:
        """Build grid coordinates for all neurons."""
        coords = []
        for r in range(self.rows):
            for c in range(self.cols):
                coords.append((r, c))
        return np.array(coords, dtype=np.float32)

    def _initialize_weights(self, data: np.ndarray) -> None:
        """Initialize neuron weights with random samples from data."""
        n_neurons = self.rows * self.cols
        indices = self.rng.integers(0, len(data), size=n_neurons)
        self.weights = data[indices].copy().astype(np.float32)

    def _find_bmu(self, vec: np.ndarray) -> int:
        """Find best matching unit (BMU) for a given vector."""
        diff = self.weights - vec
        dists = np.linalg.norm(diff, axis=1)
        return int(np.argmin(dists))

    def _train(self, data: np.ndarray) -> None:
        """Train the SOM on the given data."""
        if len(data) == 0:
            raise ValueError("No data provided for SOM training")

        if self.weights is None:
            self._initialize_weights(data)

        for step in range(self.iterations):
            # Select random sample
            sample = data[self.rng.integers(0, len(data))]

            # Find best matching unit
            bmu_idx = self._find_bmu(sample)

            # Calculate learning rate and neighborhood radius
            lr = self.learning_rate * np.exp(-step / self.iterations)
            sigma = self.sigma * np.exp(-step / self.iterations)

            # Calculate influence of BMU on all neurons
            grid_dists = np.sum((self._coords - self._coords[bmu_idx]) ** 2, axis=1)
            influence = np.exp(-grid_dists / (2 * sigma * sigma + 1e-9))

            # Update weights
            delta = (sample - self.weights) * influence[:, None]
            self.weights += lr * delta

    def distances_to_nodes(self, data: np.ndarray) -> np.ndarray:
        """Compute distances from each sample to each node.

        Args:
            data: Input data of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, n_nodes)
        """
        # Broadcasting: data[:, None, :] is (n_samples, 1, n_features)
        #               weights[None, :, :] is (1, n_nodes, n_features)
        # Result is (n_samples, n_nodes, n_features)
        diff = data[:, None, :] - self.weights[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def assign_unique_nodes(self, data: np.ndarray) -> list[int]:
        """Assign every sample to a unique grid node using greedy algorithm.

        This ensures no two samples map to the same grid position, avoiding
        blank spaces in the visualization.

        Args:
            data: Input data of shape (n_samples, n_features)

        Returns:
            List of node indices, one per sample

        Raises:
            ValueError: If grid has fewer nodes than samples to place
            RuntimeError: If assignment fails
        """
        distances = self.distances_to_nodes(data)
        num_nodes = self.rows * self.cols

        if data.shape[0] > num_nodes:
            raise ValueError(
                f"Grid has fewer nodes ({num_nodes}) than samples to place ({data.shape[0]}). "
                f"Increase grid_size or reduce number of samples."
            )

        # Sort samples by their minimum distance to any node
        # Samples with better fits get assigned first
        order = np.argsort(np.min(distances, axis=1))

        assignments = [-1] * data.shape[0]
        used_nodes = set()

        for sample_idx in order:
            # Get nodes sorted by distance to this sample
            node_candidates = np.argsort(distances[sample_idx])

            # Assign to the closest unused node
            for node_idx in node_candidates:
                if node_idx not in used_nodes:
                    assignments[sample_idx] = int(node_idx)
                    used_nodes.add(int(node_idx))
                    break

        if -1 in assignments:
            missing = assignments.count(-1)
            raise RuntimeError(f"Failed to assign {missing} samples to nodes")

        return assignments

    def _map_to_grid(self, data: np.ndarray) -> np.ndarray:
        """Map data points to normalized 2D coordinates based on SOM grid.

        Behavior depends on unique_assignment parameter:
        - If True: Uses greedy unique assignment (one sample per grid node)
        - If False: Uses classic BMU assignment (multiple samples can share nodes)

        Returns coordinates normalized to [0, 1] range.
        """
        if self.unique_assignment:
            # Use unique assignment to avoid blank spaces and overlaps
            assignments = self.assign_unique_nodes(data)
        else:
            # Classic approach: assign each point to its BMU (allows duplicates)
            assignments = []
            for vec in data:
                bmu_idx = self._find_bmu(vec)
                assignments.append(bmu_idx)

        # Convert grid indices to coordinates
        coords_2d = np.array([self._coords[idx] for idx in assignments])

        # Normalize to [0, 1] range
        coords_2d[:, 0] /= (self.rows - 1) if self.rows > 1 else 1.0
        coords_2d[:, 1] /= (self.cols - 1) if self.cols > 1 else 1.0

        return coords_2d

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply SOM to reduce embeddings to 2D grid coordinates.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            2D coordinates of shape (n_samples, 2) normalized to [0, 1]
        """
        embeddings = embeddings.astype(np.float32)
        self._coords = self._build_coords()
        self._train(embeddings)
        return self._map_to_grid(embeddings)


class DendrogramReducer(DimensionalityReducer):
    """Hierarchical clustering dendrogram for dimensionality reduction.

    This reducer uses scipy's hierarchical clustering to create a dendrogram
    structure, then extracts leaf positions for 2D visualization. Best used
    with image thumbnails to show the hierarchical clustering visually.
    """

    def __init__(
        self,
        method: str = "average",
        metric: str = "euclidean",
        optimal_ordering: bool = True,
        **kwargs,
    ):
        """Initialize dendrogram reducer.

        Args:
            method: Linkage method for hierarchical clustering.
                Options: 'single', 'complete', 'average', 'weighted', 'centroid',
                'median', 'ward' (default: 'average')
            metric: Distance metric (default: 'euclidean').
                For 'ward' method, only 'euclidean' is supported.
            optimal_ordering: If True, reorder linkage to minimize distance
                between successive leaves (default: True)
            **kwargs: Additional arguments passed to scipy.cluster.hierarchy.linkage
        """
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for dendrogram. "
                "Install it with: pip install scipy"
            )

        self.method = method
        self.metric = metric
        self.optimal_ordering = optimal_ordering
        self.kwargs = kwargs
        self.linkage_matrix = None
        self.dendrogram_data = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply hierarchical clustering and extract leaf positions.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)

        Returns:
            2D coordinates of shape (n_samples, 2) where:
            - x-coordinate: horizontal position from dendrogram leaf ordering
            - y-coordinate: set to 0 (leaves are at bottom of dendrogram)
        """
        embeddings = embeddings.astype(np.float32)

        # Compute linkage matrix
        if self.method == "ward":
            # Ward requires euclidean distance
            self.linkage_matrix = linkage(
                embeddings,
                method=self.method,
                metric="euclidean",
                optimal_ordering=self.optimal_ordering,
                **self.kwargs,
            )
        else:
            # Compute pairwise distances first for other methods
            distances = pdist(embeddings, metric=self.metric)
            self.linkage_matrix = linkage(
                distances,
                method=self.method,
                optimal_ordering=self.optimal_ordering,
                **self.kwargs,
            )

        # Generate dendrogram structure (without plotting)
        self.dendrogram_data = dendrogram(
            self.linkage_matrix,
            no_plot=True,
        )

        # Extract leaf positions
        # dendrogram_data['leaves'] gives the order of original samples at the bottom
        # dendrogram_data['icoord'] and 'dcoord' give the dendrogram structure
        leaf_order = self.dendrogram_data["leaves"]

        # Create position mapping: original_index -> x_position
        n_samples = len(leaf_order)
        coords_2d = np.zeros((n_samples, 2), dtype=np.float32)

        # Map each sample to its position in the dendrogram ordering
        for new_pos, original_idx in enumerate(leaf_order):
            coords_2d[original_idx, 0] = new_pos
            coords_2d[original_idx, 1] = 0  # All leaves at y=0

        # Normalize x-coordinates to [0, 1] range
        if n_samples > 1:
            coords_2d[:, 0] /= n_samples - 1

        return coords_2d
