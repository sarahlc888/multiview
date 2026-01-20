"""Shared utilities for D5 dataset variants.

Common functionality for loading D5 ABC news articles with descriptor applicability scores.
"""

from __future__ import annotations

import logging
import pickle
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# D5 dataset configuration
D5_REPO_URL = "https://github.com/ruiqi-zhong/D5"
D5_PKL_URL = "https://github.com/ruiqi-zhong/D5/raw/main/output.pkl"
D5_CACHE_DIR = Path.home() / ".cache" / "multiview" / "D5"
D5_OUTPUT_PATH = D5_CACHE_DIR / "output.pkl"


def ensure_pkl_downloaded() -> None:
    """Download D5 PKL file if not cached."""
    if not D5_OUTPUT_PATH.exists():
        logger.info(f"Downloading D5 output.pkl to {D5_OUTPUT_PATH}")
        try:
            D5_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(D5_PKL_URL, str(D5_OUTPUT_PATH))
            logger.info("Successfully downloaded D5 output.pkl")
        except Exception as e:
            raise RuntimeError(f"Failed to download D5 output.pkl: {e}") from e
    else:
        logger.debug(f"D5 output.pkl already cached at {D5_OUTPUT_PATH}")


def load_d5_data() -> tuple[list[str], list[str], np.ndarray]:
    """Load D5 data from PKL file.

    Returns:
        Tuple of (descriptor_names, doc_texts, applicability_matrix)
        - descriptor_names: List of descriptor text strings
        - doc_texts: List of document text strings
        - applicability_matrix: Shape (n_docs, n_descriptors) with applicability scores
    """
    ensure_pkl_downloaded()

    logger.info(f"Loading D5 from {D5_OUTPUT_PATH}")

    # Load PKL file
    try:
        with open(D5_OUTPUT_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load D5 PKL: {e}") from e

    # Extract descriptor names
    descriptor_names = list(data.keys())

    # Extract document texts (from first descriptor's sample2score keys)
    first_descriptor = descriptor_names[0]
    doc_texts = list(data[first_descriptor]["sample2score"].keys())

    logger.debug(
        f"Found {len(doc_texts)} documents and {len(descriptor_names)} descriptors"
    )

    # Build applicability matrix: shape (n_docs, n_descriptors)
    n_docs = len(doc_texts)
    n_descriptors = len(descriptor_names)

    applicability_matrix = np.zeros((n_docs, n_descriptors))

    for desc_idx, descriptor in enumerate(descriptor_names):
        for doc_idx, doc_text in enumerate(doc_texts):
            score = data[descriptor]["sample2score"].get(doc_text, 0.0)
            applicability_matrix[doc_idx, desc_idx] = score

    logger.debug(
        f"Built applicability matrix: shape {applicability_matrix.shape}, "
        f"mean score: {applicability_matrix.mean():.3f}"
    )

    return descriptor_names, doc_texts, applicability_matrix
