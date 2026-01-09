"""Base class for document_sets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseDocSet(ABC):
    """Abstract base class for all document_sets.

    Subclasses must implement:
    - load_documents(): Load and return the document_set documents
    - get_document_text(): Extract text from a document for universal criteria

    Subclasses must define:
    - DATASET_PATH: Path to the document_set
    - DESCRIPTION: Human-readable description
    - KNOWN_CRITERIA: List of criteria that can be extracted deterministically
      (word_count is automatically included)
    """

    # Subclasses must define these
    DATASET_PATH: str
    DESCRIPTION: str
    KNOWN_CRITERIA: list[str] = []

    # Optional: Metadata for LM-based criteria (schema hints, descriptions, etc.)
    # Format: {criterion_name: {description: str, category_schema_hint: str, ...}}
    CRITERION_METADATA: dict[str, dict[str, str]] = {}

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

        Returns:
            List of documents
        """
        pass

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

    def get_criterion_metadata(self, criterion: str) -> dict[str, str]:
        """Get metadata for a criterion (description, schema hints, etc.).

        Args:
            criterion: The criterion name

        Returns:
            Dict with metadata keys like:
            - description: Criterion description
            - category_schema_hint: Hint for category schema generation
            - tag_schema_hint: Hint for tag schema generation
            - summary_guidance_hint: Hint for summary guidance
            - summary_format_hint: Hint for summary format
            Returns empty dict if no metadata is defined.
        """
        return self.CRITERION_METADATA.get(criterion, {})
