"""D5 ABC news dataset loader with descriptor applicability scores.

The D5 dataset contains:
- 2000 ABC news documents (headlines/snippets)
- 60 descriptors (hypotheses about content)
- Continuous applicability scores [0, 1] for each document-descriptor pair

PKL Structure:
{
    "descriptor_text_1": {
        "hypothesis": "descriptor text",
        "sample2score": {"doc_text_1": 0.91, "doc_text_2": 0.03, ...},
        ...
    },
    ...
}

For prelabeled triplet creation, we binarize scores at threshold 0.5:
- "applicable" if score >= 0.5
- "not_applicable" if score < 0.5
"""

from __future__ import annotations

import logging
import pickle
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

# D5 dataset configuration
D5_REPO_URL = "https://github.com/ruiqi-zhong/D5"
D5_PKL_URL = "https://github.com/ruiqi-zhong/D5/raw/main/output.pkl"
D5_CACHE_DIR = Path.home() / ".cache" / "multiview" / "D5"
D5_OUTPUT_PATH = D5_CACHE_DIR / "output.pkl"


class D5DocSet(BaseDocSet):
    """D5 ABC news articles with descriptor applicability scores.

    Supports prelabeled triplet creation using binarized applicability scores.

    Config parameters:
        max_docs (int, optional): Maximum number of documents to load
        binarization_threshold (float): Threshold for binarizing scores (default: 0.5)

    Criterion format:
        description_N: Use descriptor at index N (e.g., description_0 for first descriptor)

    Usage:
        tasks:
          - document_set: d5
            criterion: description_0  # Use first descriptor
            triplet_style: prelabeled
            binarization_threshold: 0.5
    """

    DATASET_PATH = str(D5_OUTPUT_PATH)
    DESCRIPTION = "D5 ABC news with 60 descriptor applicability scores"

    # Known criteria dynamically populated: ["description_0", "description_1", ...]
    KNOWN_CRITERIA = []

    def __init__(self, config: dict | None = None):
        """Initialize D5DocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)

        self.binarization_threshold = self.config.get("binarization_threshold", 0.5)

        # These will be populated during load
        self.applicability_matrix = None  # Shape: (n_docs, m_descriptors)
        self.descriptor_names = []  # List of descriptor text strings
        self.doc_texts = []  # List of document texts (for indexing)

        # Ensure PKL is downloaded
        self._ensure_pkl_downloaded()

        # Load descriptor names to populate KNOWN_CRITERIA and CRITERION_METADATA
        self._load_descriptor_names()

        # Initialize precomputed annotations
        self.PRECOMPUTED_ANNOTATIONS = {}

    def _ensure_pkl_downloaded(self) -> None:
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

    def _load_descriptor_names(self) -> None:
        """Load descriptor names from PKL to populate KNOWN_CRITERIA and CRITERION_METADATA."""
        try:
            with open(D5_OUTPUT_PATH, "rb") as f:
                data = pickle.load(f)

            self.descriptor_names = list(data.keys())
            self.KNOWN_CRITERIA = [
                f"description_{i}" for i in range(len(self.descriptor_names))
            ]

            # Populate CRITERION_METADATA with actual descriptor text
            self.CRITERION_METADATA = {}
            for i, descriptor_text in enumerate(self.descriptor_names):
                criterion_name = f"description_{i}"
                self.CRITERION_METADATA[criterion_name] = {
                    "description": f"applicability of the descriptor, '{descriptor_text}'",
                }

            logger.debug(f"Loaded {len(self.descriptor_names)} descriptors from D5 PKL")
        except Exception as e:
            logger.warning(f"Failed to load descriptor names: {e}")
            self.descriptor_names = []
            self.KNOWN_CRITERIA = []
            self.CRITERION_METADATA = {}

    def load_documents(self) -> list[Any]:
        """Load ABC news articles from D5 PKL file.

        Returns:
            List of document text strings
        """
        logger.info(f"Loading D5 from {D5_OUTPUT_PATH}")

        # Load PKL file
        try:
            with open(D5_OUTPUT_PATH, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load D5 PKL: {e}") from e

        # Extract document texts (from first descriptor's sample2score keys)
        first_descriptor = list(data.keys())[0]
        self.doc_texts = list(data[first_descriptor]["sample2score"].keys())

        logger.debug(
            f"Found {len(self.doc_texts)} documents and {len(data)} descriptors"
        )

        # Build applicability matrix: shape (n_docs, m_descriptors)
        n_docs = len(self.doc_texts)
        m_descriptors = len(self.descriptor_names)

        self.applicability_matrix = np.zeros((n_docs, m_descriptors))

        for desc_idx, descriptor in enumerate(self.descriptor_names):
            for doc_idx, doc_text in enumerate(self.doc_texts):
                score = data[descriptor]["sample2score"].get(doc_text, 0.0)
                self.applicability_matrix[doc_idx, desc_idx] = score

        logger.debug(
            f"Built applicability matrix: shape {self.applicability_matrix.shape}, "
            f"mean score: {self.applicability_matrix.mean():.3f}"
        )

        # Apply max_docs if specified
        max_docs = self.config.get("max_docs")
        if max_docs is not None and max_docs < n_docs:
            self.doc_texts = self.doc_texts[:max_docs]
            self.applicability_matrix = self.applicability_matrix[:max_docs, :]

        logger.info(f"Loaded {len(self.doc_texts)} D5 documents")

        # Return documents as simple text strings
        return self.doc_texts

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document (string or dict)

        Returns:
            Text content
        """
        if isinstance(document, dict):
            return document.get("text", str(document))
        return str(document)

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract criterion value for known criteria.

        Supports:
        - word_count: from base class
        - description_N: Binarized applicability for descriptor at index N

        Args:
            document: A document
            criterion: The criterion name (e.g., "description_0")

        Returns:
            "applicable" or "not_applicable" for descriptor criteria
        """
        # Check if it's a descriptor criterion
        if criterion.startswith("description_"):
            try:
                descriptor_idx = int(criterion.split("_")[1])

                if descriptor_idx >= len(self.descriptor_names):
                    logger.warning(
                        f"Descriptor index {descriptor_idx} out of range "
                        f"(have {len(self.descriptor_names)} descriptors)"
                    )
                    return None

                # Find document index
                doc_text = self.get_document_text(document)

                try:
                    doc_idx = self.doc_texts.index(doc_text)
                except ValueError:
                    logger.warning(
                        f"Document not found in doc_texts: {doc_text[:50]}..."
                    )
                    return None

                # Get applicability score and binarize
                score = self.applicability_matrix[doc_idx, descriptor_idx]
                return (
                    "applicable"
                    if score >= self.binarization_threshold
                    else "not_applicable"
                )

            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Invalid descriptor criterion format: {criterion}, error: {e}"
                )
                return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations_for_descriptor(self, criterion: str) -> None:
        """Build precomputed annotations for a specific descriptor.

        Args:
            criterion: Descriptor criterion name (e.g., "description_0")
        """
        if not criterion.startswith("description_"):
            logger.warning(
                f"Cannot build precomputed annotations for non-descriptor criterion: {criterion}"
            )
            return

        try:
            descriptor_idx = int(criterion.split("_")[1])
        except (ValueError, IndexError):
            logger.warning(f"Invalid descriptor criterion format: {criterion}")
            return

        if descriptor_idx >= len(self.descriptor_names):
            logger.warning(
                f"Descriptor index {descriptor_idx} out of range "
                f"(have {len(self.descriptor_names)} descriptors)"
            )
            return

        descriptor_text = self.descriptor_names[descriptor_idx]
        logger.info(
            f"Building precomputed annotations for {criterion} "
            f'(descriptor: "{descriptor_text[:60]}...") '
            f"with threshold={self.binarization_threshold}"
        )

        annotations = {}
        n_applicable = 0

        for doc_idx, doc_text in enumerate(self.doc_texts):
            score = self.applicability_matrix[doc_idx, descriptor_idx]

            # Binarize at threshold
            criterion_value = (
                "applicable"
                if score >= self.binarization_threshold
                else "not_applicable"
            )

            if criterion_value == "applicable":
                n_applicable += 1

            # Create annotation with a summary explaining the label
            # This makes prelabeled annotations look like rich annotations
            summary_text = f"This document is labeled as '{criterion_value}' for the descriptor: {descriptor_text}"

            annotations[doc_text] = {
                "criterion_value": criterion_value,
                "summary": {"final_summary": summary_text},
            }

        self.PRECOMPUTED_ANNOTATIONS[criterion] = annotations

        logger.info(
            f"Built precomputed annotations for {criterion}: "
            f"{n_applicable}/{len(self.doc_texts)} applicable "
            f"({100*n_applicable/len(self.doc_texts):.1f}%)"
        )

    def has_precomputed_annotations(self, criterion: str) -> bool:
        """Check if criterion has pre-computed annotations.

        For D5, we build annotations on-demand for descriptor criteria.

        Args:
            criterion: The criterion name

        Returns:
            True if precomputed annotations are available
        """
        # If already built, return True
        if criterion in self.PRECOMPUTED_ANNOTATIONS:
            return True

        # If it's a valid descriptor criterion, build it on-demand
        if criterion.startswith("description_"):
            try:
                descriptor_idx = int(criterion.split("_")[1])
                if 0 <= descriptor_idx < len(self.descriptor_names):
                    # Build annotations for this descriptor
                    self._build_precomputed_annotations_for_descriptor(criterion)
                    return criterion in self.PRECOMPUTED_ANNOTATIONS
            except (ValueError, IndexError):
                pass

        return False
