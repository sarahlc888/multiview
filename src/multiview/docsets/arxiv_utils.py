"""Shared utilities for ArXiv document sets.

Contains common functions for loading and processing ArXiv papers.
"""

import logging
import re
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Dataset path for ArXiv metadata
ARXIV_DATASET_PATH = "librarian-bots/arxiv-metadata-snapshot"


def load_arxiv_abstracts(
    category_filter: str = "cs.AI",
    max_abstracts: int | None = None,
    split: str = "train",
    seed: int = 42,
) -> Iterator[dict[str, Any]]:
    """Load ArXiv paper abstracts with category filtering.

    Args:
        category_filter: Category to filter by (e.g., "cs.AI", "cs.LG")
        max_abstracts: Maximum number of abstracts to load
        split: Dataset split to use
        seed: Random seed for shuffling

    Yields:
        Dict with keys: "id", "abstract", "categories", and other metadata
    """
    logger.info(f"Loading ArXiv papers from HuggingFace: {ARXIV_DATASET_PATH}")
    logger.debug(f"Filtering by category: {category_filter}")

    # Always use streaming to avoid downloading the entire dataset
    logger.debug("Using streaming mode")
    dataset = load_dataset(ARXIV_DATASET_PATH, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    # Iterate and filter
    count = 0
    for example in dataset:
        # Filter by category
        categories = example.get("categories", "")
        if category_filter and category_filter not in categories:
            continue

        # Skip if no abstract
        abstract = example.get("abstract", "").strip()
        if not abstract:
            continue

        yield example

        count += 1
        if max_abstracts is not None and count >= max_abstracts:
            break

    logger.debug(f"Loaded {count} ArXiv abstracts")


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics.

    Args:
        text: Input text to split

    Returns:
        List of sentence strings
    """
    # Normalize whitespace: convert newlines and multiple spaces to single spaces
    text = re.sub(r"\s+", " ", text.strip())

    # Basic sentence splitting on period, exclamation, or question mark followed by space
    # This is a simple heuristic and may not be perfect for all cases
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]
