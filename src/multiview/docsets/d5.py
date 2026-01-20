"""D5 ABC news dataset loader with descriptor applicability scores.

Loads 2000 ABC news articles with 60 descriptor applicability scores (0-1 continuous).
Downloads output.pkl from GitHub (ruiqi-zhong/D5) on first use, caches locally.

Structure: PKL with descriptors as keys, each containing sample2score dict mapping
document text to applicability score. Binarized at threshold 0.5 (configurable).

Key features:
- Pre-labeled multi-criteria dataset (no LM annotation needed)
- 60 KNOWN_CRITERIA (description_0 through description_59)
- Supports prelabeled triplet creation using binary labels
- Mean score ~0.13 (13% positive labels)
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.docsets.base import BaseDocSet
from multiview.docsets.d5_utils import D5_OUTPUT_PATH, load_d5_data

logger = logging.getLogger(__name__)


class D5DocSet(BaseDocSet):
    """D5 ABC news articles with descriptor applicability scores (doc-to-doc variant).

    Supports prelabeled triplet creation using binarized applicability scores.
    This is the original D5 task where documents are compared to each other.

    Config parameters:
        max_docs (int, optional): Maximum number of documents to load
        binarization_threshold (float): Threshold for binarizing scores (default: 0.5)

    Criterion format:
        description_N: Use descriptor at index N (e.g., description_0 for first descriptor)

    Usage:
        tasks:
          - document_set: d5_doc2doc
            criterion: description_0  # Use first descriptor
            triplet_style: prelabeled
            binarization_threshold: 0.5
    """

    DATASET_PATH = str(D5_OUTPUT_PATH)
    DESCRIPTION = "D5 ABC news with 60 descriptor applicability scores (doc-to-doc)"

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

        # Load descriptor names and data to populate KNOWN_CRITERIA and CRITERION_METADATA
        self._load_data()

        # Initialize precomputed annotations
        self.PRECOMPUTED_ANNOTATIONS = {}

    def _load_data(self) -> None:
        """Load D5 data and populate KNOWN_CRITERIA and CRITERION_METADATA."""
        self.descriptor_names, self.doc_texts, self.applicability_matrix = (
            load_d5_data()
        )

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

    def load_documents(self) -> list[Any]:
        """Load ABC news articles from D5 PKL file.

        Returns:
            List of document text strings
        """
        logger.info(f"Loading D5 doc-to-doc from {D5_OUTPUT_PATH}")

        # Data already loaded in __init__ via _load_data()
        # Apply max_docs if specified
        max_docs = self.config.get("max_docs")
        if max_docs is not None and max_docs < len(self.doc_texts):
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
                "prelabel": criterion_value,
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
