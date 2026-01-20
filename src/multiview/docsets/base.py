"""Base class for document_sets."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseDocSet(ABC):
    """Abstract base class for all document_sets.

    Subclasses must implement:
    - load_documents(): Load and return the document_set documents
    - get_document_text(): Extract text from a document for universal criteria

    Subclasses must define:
    - DATASET_PATH: Path to the document_set
    - DESCRIPTION: Human-readable description
    - DOCUMENT_TYPE: Human-readable description of document type (e.g., "math word problem")
    - KNOWN_CRITERIA: List of criteria that can be extracted deterministically
      (word_count is automatically included)
    """

    # Subclasses must define these
    DATASET_PATH: str
    DESCRIPTION: str
    DOCUMENT_TYPE: str = "document"  # Default fallback
    KNOWN_CRITERIA: list[str] = []

    # Optional: Metadata for LM-based criteria (schema hints, descriptions, etc.)
    # Format: {criterion_name: {description: str, default_hint: str, category_schema_hint: str, ...}}
    # Hint resolution: specific hint > default_hint > None
    #
    # Note: values may be non-strings (e.g., dicts for structured hints like
    # triplet_example_hint), so we type this as Any.
    CRITERION_METADATA: dict[str, dict[str, Any]] = {}
    # Synthesis prompts for criterion-specific document generation
    # Maps criterion name → {remix_prompt}
    # Subclasses can override to provide custom synthesis logic per criterion
    SYNTHESIS_CONFIGS: dict[str, dict[str, str]] = {}

    # Optional: Pre-computed annotations for datasets with gold labels
    # Format: {criterion_name: {document_text: {"prelabel": value}}}
    # Subclasses can populate this to skip LM-based annotation generation
    PRECOMPUTED_ANNOTATIONS: dict[str, dict[str, dict]] = {}

    def __init__(self, config: dict | None = None):
        """Initialize document_set.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.document_set_path = Path(
            self.config.get("document_set_path", self.DATASET_PATH)
        )

    @abstractmethod
    def load_documents(self) -> list[Any]:
        """Load documents from the document_set.

        Note: For dataset loading with max_docs < 100, use streaming mode
        to efficiently assemble the data without loading the entire dataset.

        Important: Subclasses should call self._deduplicate(documents) before
        returning to ensure no duplicate documents exist.

        Returns:
            List of documents
        """
        pass

    def _deduplicate(self, documents: list[Any]) -> list[Any]:
        """Remove duplicate documents based on text content.

        This prevents issues where anchor and positive could be the same document,
        which creates trivial triplets that don't test anything meaningful.

        Args:
            documents: List of documents (can be strings or dicts)

        Returns:
            List of unique documents (preserving order of first occurrence)
        """
        if not documents:
            return documents

        original_count = len(documents)
        seen_texts = {}  # text -> first index where it appears
        unique_docs = []

        for idx, doc in enumerate(documents):
            # Extract text from document (handle both dicts and strings)
            text = self.get_document_text(doc)

            # Keep only first occurrence of each unique text
            if text not in seen_texts:
                seen_texts[text] = idx
                unique_docs.append(doc)
            else:
                first_idx = seen_texts[text]
                logger.debug(
                    f"Removing duplicate document at index {idx} "
                    f"(duplicate of document at index {first_idx})"
                )

        num_duplicates = original_count - len(unique_docs)
        if num_duplicates > 0:
            logger.warning(
                f"Removed {num_duplicates} duplicate document(s) "
                f"({original_count} -> {len(unique_docs)} documents)"
            )

        return unique_docs

    @abstractmethod
    def get_document_text(self, document: Any) -> str:
        """Extract text from a document for universal criteria like word_count.

        Args:
            document: A single document

        Returns:
            The text content of the document
        """
        pass

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract criterion value for known criteria.

        Universal criteria (handled by base class):
        - word_count: Count of words in the document text

        Subclasses can override this method to add custom criteria.

        Args:
            document: A single document
            criterion: The criterion name

        Returns:
            The criterion value, or None if not a known criterion
        """
        # Universal criteria
        if criterion == "word_count":
            text = self.get_document_text(document)
            return len(text.split())

        return None

    def get_criterion_metadata(self, criterion: str) -> dict[str, Any]:
        """Get metadata for a criterion (description, schema hints, etc.)."""
        return self.CRITERION_METADATA.get(criterion, {})

    def has_precomputed_annotations(self, criterion: str) -> bool:
        """Check if criterion has pre-computed annotations.

        Args:
            criterion: The criterion name

        Returns:
            True if precomputed annotations are available for this criterion
        """
        return criterion in self.PRECOMPUTED_ANNOTATIONS

    def get_precomputed_annotations(self, criterion: str) -> dict[str, dict]:
        """Get pre-computed annotations for a criterion.

        Args:
            criterion: The criterion name

        Returns:
            Dict mapping document text -> annotation dict
            (e.g., {"doc text": {"prelabel": "value"}})
            Returns empty dict if no precomputed annotations exist
        """
        return self.PRECOMPUTED_ANNOTATIONS.get(criterion, {})

    def get_document_image(self, document: Any) -> str | None:
        """Extract image source from a document for vision tasks.

        Subclasses can override this to provide image sources for documents.
        Image sources can be URLs or local file paths.

        Args:
            document: A single document

        Returns:
            Image source (URL or file path), or None if document has no image
        """
        # Default: no images
        return None
