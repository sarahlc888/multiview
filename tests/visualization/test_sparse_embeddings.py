"""Test that reducers handle sparse embeddings with some zero-variance dimensions.

This is a regression test for an issue where pseudologit embeddings were rejected
because they had some zero-variance dimensions (unused categories).
"""
import numpy as np
import pytest

from multiview.visualization import PCAReducer, TSNEReducer


def test_sparse_embeddings_with_zero_variance_dimensions():
    """Reducers should work with embeddings where some dimensions have zero variance.

    This is common with pseudologit embeddings where not all categories are used.
    The reducers should only fail if ALL embeddings are identical.
    """
    # Create embeddings where some dimensions have zero variance
    # but embeddings are NOT all identical (like the real pseudologit case)
    embeddings = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.float32)

    # Verify some dimensions have zero variance
    assert np.any(np.var(embeddings, axis=0) == 0), "Should have some zero-variance dims"
    # But not all embeddings are identical
    assert not np.all(embeddings == embeddings[0]), "Embeddings should differ"

    # PCA should succeed
    pca = PCAReducer(n_components=2)
    coords_pca = pca.fit_transform(embeddings)
    assert coords_pca.shape == (4, 2)


def test_constant_embeddings_should_fail():
    """Reducers should fail if ALL embeddings are identical."""
    # All embeddings are the same
    embeddings = np.ones((5, 10), dtype=np.float32)

    pca = PCAReducer(n_components=2)
    with pytest.raises(ValueError, match="zero variance.*constant vectors"):
        pca.fit_transform(embeddings)

    # t-SNE should also fail
    tsne = TSNEReducer(perplexity=2)
    with pytest.raises(ValueError, match="zero variance.*constant vectors"):
        tsne.fit_transform(embeddings)
