"""Sampling utilities for deterministic operations."""

from __future__ import annotations

import hashlib
import random


def deterministic_sample(items: list, k: int, seed_base: str) -> list:
    """Sample k items deterministically based on seed_base.

    This function ensures that the same inputs always produce the same sample,
    which is important for caching and reproducibility.

    Args:
        items: List of items to sample from
        k: Number of items to sample
        seed_base: String to use as seed base (e.g., criterion name).
            The same seed_base will always produce the same sample.

    Returns:
        Deterministically sampled list of k items

    Example:
        >>> docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> sample1 = deterministic_sample(docs, 3, "my_criterion")
        >>> sample2 = deterministic_sample(docs, 3, "my_criterion")
        >>> assert sample1 == sample2  # Always the same
    """
    # Create a deterministic seed from the seed_base
    seed = int(hashlib.md5(seed_base.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    return rng.sample(items, min(k, len(items)))
