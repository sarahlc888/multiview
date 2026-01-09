"""Utilities for triplet creation and sampling."""

import random


def create_random_triplets(
    documents: list[dict],
    max_triplets: int | None = None,
) -> list[tuple]:
    """Create random triplets from documents.

    Args:
        documents: List of documents
        max_triplets: Maximum number of triplets to create (None = unlimited)

    Returns:
        List of (anchor, positive, negative) triplets
    """
    if len(documents) < 3:
        return []

    triplets = []
    num_triplets = max_triplets if max_triplets is not None else len(documents)

    for _ in range(num_triplets):
        # Sample 3 random distinct documents
        sampled = random.sample(documents, 3)
        triplets.append(tuple(sampled))

    return triplets


def create_lm_triplets(
    documents: list[dict],
    max_triplets: int | None = None,
) -> list[tuple]:
    """Create triplets using language model to determine similarity.

    Args:
        documents: List of documents
        max_triplets: Maximum number of triplets to create (None = unlimited)

    Returns:
        List of (anchor, positive, negative) triplets
    """
    raise NotImplementedError("LM-based triplet creation not yet implemented")
