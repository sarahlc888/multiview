"""Similarity computation functions for different embedding types.

This module provides similarity functions for both single-vector and multi-vector
embeddings, with automatic dispatching based on array dimensions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(
    vec_a: list[float] | NDArray, vec_b: list[float] | NDArray
) -> float:
    """Compute cosine similarity between two single vectors.

    Args:
        vec_a: First vector (1D array)
        vec_b: Second vector (1D array)

    Returns:
        Cosine similarity score in range [-1, 1]
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def maxsim_similarity(multi_vec_a: NDArray, multi_vec_b: NDArray) -> float:
    """Compute MaxSim similarity between two multi-vector embeddings (ColBERT-style).

    MaxSim(Q, D) = mean over tokens_q of max over tokens_d of cosine(q, d)

    This implementation assumes embeddings are already normalized and uses dot product
    for efficiency. Zero-padded vectors (from ColBERT padding) are filtered out.

    Args:
        multi_vec_a: First multi-vector embedding, shape (num_tokens_a, embedding_dim)
        multi_vec_b: Second multi-vector embedding, shape (num_tokens_b, embedding_dim)

    Returns:
        MaxSim similarity score in range [0, 1] for normalized embeddings
    """
    # Convert to numpy arrays
    a = np.array(multi_vec_a)
    b = np.array(multi_vec_b)

    # Filter out zero-padded vectors (ColBERT pads with zeros)
    # A token vector is considered padding if its norm is very small
    a_norms = np.linalg.norm(a, axis=1)
    b_norms = np.linalg.norm(b, axis=1)

    a_valid = a[a_norms > 1e-6]
    b_valid = b[b_norms > 1e-6]

    # Handle edge cases
    if len(a_valid) == 0 or len(b_valid) == 0:
        return 0.0

    # Normalize embeddings (in case they aren't already normalized)
    a_valid = a_valid / (np.linalg.norm(a_valid, axis=1, keepdims=True) + 1e-8)
    b_valid = b_valid / (np.linalg.norm(b_valid, axis=1, keepdims=True) + 1e-8)

    # Compute pairwise dot products: shape (num_tokens_a, num_tokens_b)
    # For normalized vectors, dot product = cosine similarity
    similarities = np.dot(a_valid, b_valid.T)

    # MaxSim: for each token in a, take max similarity with any token in b, then average
    max_sims = np.max(similarities, axis=1)
    maxsim_score = float(np.mean(max_sims))

    return maxsim_score


def compute_similarity(
    vec_a: list[float] | NDArray, vec_b: list[float] | NDArray
) -> float:
    """Automatically dispatch to appropriate similarity function based on array dimensions.

    - 1D arrays: Use cosine similarity
    - 2D arrays: Use MaxSim similarity (multi-vector ColBERT-style)

    Args:
        vec_a: First embedding (1D or 2D array)
        vec_b: Second embedding (1D or 2D array)

    Returns:
        Similarity score

    Raises:
        ValueError: If arrays have mismatched dimensions or unsupported shapes
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    # Check dimensions match
    if a.ndim != b.ndim:
        raise ValueError(
            f"Embedding dimension mismatch: vec_a has {a.ndim}D shape, "
            f"vec_b has {b.ndim}D shape. Both must be either 1D or 2D."
        )

    # Dispatch based on dimensionality
    if a.ndim == 1:
        # Single-vector embeddings: use cosine similarity
        return cosine_similarity(a, b)
    elif a.ndim == 2:
        # Multi-vector embeddings: use MaxSim
        return maxsim_similarity(a, b)
    else:
        raise ValueError(
            f"Unsupported embedding shape: {a.ndim}D. "
            f"Only 1D (single-vector) and 2D (multi-vector) embeddings are supported."
        )
