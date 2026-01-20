"""Triplet creation for pairwise similarity datasets.

For datasets like InstructSTSB where similarity is defined pairwise rather than
via document labels.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


def create_pairwise_similarity_triplets(
    documents: list[dict],
    docset: Any,  # BaseDocSet with get_similar_documents/get_dissimilar_documents
    criterion: str,
    max_triplets: int | None = None,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Create triplets using pairwise similarity relationships.

    Uses docset.get_similar_documents() and docset.get_dissimilar_documents()
    to find positives and negatives for each anchor.

    Args:
        documents: List of document dicts
        docset: DocSet instance with get_similar_documents/get_dissimilar_documents methods
        criterion: The criterion name
        max_triplets: Maximum number of triplets to create
        seed: Random seed

    Returns:
        List of (anchor_idx, positive_idx, negative_idx) triplets
    """
    random.seed(seed)

    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Build triplets
    triplets = []
    valid_anchor_count = 0

    # Try each document as an anchor
    for anchor_idx, anchor_doc in enumerate(documents):
        # Get similar and dissimilar documents
        similar_docs = docset.get_similar_documents(anchor_doc, criterion, documents)
        dissimilar_docs = docset.get_dissimilar_documents(
            anchor_doc, criterion, documents
        )

        if not similar_docs or not dissimilar_docs:
            continue

        valid_anchor_count += 1

        # Find indices of similar and dissimilar documents
        similar_indices = [
            i
            for i, doc in enumerate(documents)
            if doc in similar_docs and i != anchor_idx
        ]
        dissimilar_indices = [
            i
            for i, doc in enumerate(documents)
            if doc in dissimilar_docs and i != anchor_idx
        ]

        if not similar_indices or not dissimilar_indices:
            continue

        # Create one triplet per anchor by randomly sampling positive and negative
        pos_idx = random.choice(similar_indices)
        neg_idx = random.choice(dissimilar_indices)
        triplets.append((anchor_idx, pos_idx, neg_idx))

        if max_triplets and len(triplets) >= max_triplets:
            break

    logger.info(
        f"Created {len(triplets)} pairwise similarity triplets from "
        f"{valid_anchor_count} valid anchors (out of {len(documents)} documents)"
    )

    return triplets
