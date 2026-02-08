"""D5 Applicability: Property-text matching in joint embedding space.

This variant creates a joint embedding space containing:
- Properties (60 descriptors formatted as "property: <text>")
- Headlines (~2000 news headlines formatted as "headline: <text>")

The task is to match properties to headlines based on applicability.

STRUCTURE:
----------
Documents (~2060 total):
  - 60 properties: "property: mention the coronavirus and the pandemic's effects"
  - ~2000 headlines: "headline: nsw records 16 new cases; all linked to known"

Single criterion: "applicability"
  - Determines which property-headline pairs match

Applicability matrix: Shape (n_headlines, n_properties)
  - Matrix[headline_idx, property_idx] = applicability score (0-1)
  - Binarized at threshold (default 0.5)

TRIPLET STRUCTURE:
------------------
Two types of triplets (mixed together):

Type 1: Property-anchored
  Anchor: property: highlight struggles of certain industries
  Positive: headline: crew of stranded coal ship (applicable)
  Negative: headline: sports news (not applicable)

Type 2: Headline-anchored
  Anchor: headline: north korea fires projectiles
  Positive: property: discuss politics and government responses (applicable)
  Negative: property: discuss criminal cases (not applicable)

EXAMPLE DOCUMENTS:
------------------
Properties:
  0. property: mention the coronavirus and the pandemic's effects
  1. property: discuss the politics of the situation, such as government responses
  2. property: discuss safety protocols and measures to prevent the spread

Headlines:
  60. headline: nsw records 16 new cases; all linked to known
  61. headline: musical theatre star caroline oconnor swaps west end
  62. headline: coronavirus queensland cargo ship sunshine coast two new cases

USAGE:
------
config = {
    "document_set": "d5_applicability",
    "criterion": "applicability",  # Single criterion!
    "triplet_style": "prelabeled",
    "max_docs": 500,  # Limits texts only (properties always included)
    "binarization_threshold": 0.5
}

KEY DIFFERENCES FROM D5_DOC2DOC:
---------------------------------
D5_doc2doc:
  - Documents: ~2000 texts only
  - Criteria: 60 different (description_0-59)
  - Task: Doc-to-doc comparison
  - Triplets: (doc, doc, doc) with same/diff labels

D5_applicability:
  - Documents: 60 properties + ~2000 texts
  - Criteria: 1 (applicability)
  - Task: Property-text matching
  - Triplets: (property, text, text) or (text, property, property)

WHAT IT TESTS:
--------------
This task evaluates whether an embedding model can:
1. Embed properties and headlines in the same space
2. Recognize when a property is applicable to a headline
3. Retrieve applicable headlines for a given property
4. Retrieve applicable properties for a given headline

For embedding models:
- Property side: "Given the following property, embed it so that it is close to headlines that it is applicable to."
- Headline side: "Given the following headline, embed it so that it is close to applicable properties."
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.docsets.base import BaseDocSet
from multiview.docsets.d5_utils import D5_OUTPUT_PATH, load_d5_data

logger = logging.getLogger(__name__)


class D5ApplicabilityDocSet(BaseDocSet):
    """D5 property-headline matching task in joint embedding space.

    Documents include:
    - All 60 properties: "property: <descriptor_text>"
    - All ~2000 headlines: "headline: <text>"

    Single criterion: "applicability"
    - Determines which property-headline pairs match

    Config parameters:
        max_docs (int, optional): Maximum number of headline documents to load (properties always included)
        binarization_threshold (float): Threshold for binarizing applicability scores (default: 0.5)

    Usage:
        tasks:
          - document_set: d5_applicability
            criterion: applicability
            triplet_style: prelabeled
            binarization_threshold: 0.5
    """

    DATASET_PATH = str(D5_OUTPUT_PATH)
    DESCRIPTION = "D5 property-text applicability in joint embedding space"

    # Single criterion for applicability matching
    KNOWN_CRITERIA = ["applicability"]

    # Static criterion metadata (since we only have one criterion)
    _CRITERION_METADATA = {
        "applicability": {
            "description": "Whether a property is applicable to a text",
        }
    }

    def __init__(self, config: dict | None = None):
        """Initialize D5ApplicabilityDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)

        self.binarization_threshold = self.config.get("binarization_threshold", 0.5)

        # These will be populated during load
        self.applicability_matrix = None  # Shape: (n_texts, n_properties)
        self.property_names = []  # List of property text strings
        self.text_contents = []  # List of raw text contents
        self.all_documents = []  # All documents (properties + texts)

        # Mapping from document index to property/text index
        self.doc_idx_to_property_idx = {}  # {doc_idx: property_idx}
        self.doc_idx_to_text_idx = {}  # {doc_idx: text_idx}
        self.property_start_idx = None  # Where properties start in all_documents
        self.text_start_idx = None  # Where texts start in all_documents

        # Load data
        self._load_data()

    def _load_data(self) -> None:
        """Load D5 data."""
        self.property_names, self.text_contents, self.applicability_matrix = (
            load_d5_data()
        )
        logger.debug(
            f"Loaded {len(self.property_names)} properties and "
            f"{len(self.text_contents)} texts from D5 PKL"
        )

    def _format_text(self, text: str) -> str:
        """Format text as a headline.

        Args:
            text: Raw text content

        Returns:
            Formatted text with headline prefix
        """
        return f"headline: {text}"

    def load_documents(self) -> list[Any]:
        """Load all documents (properties + texts).

        Returns:
            List of all documents (properties first, then texts)
        """
        logger.info(f"Loading D5 applicability from {D5_OUTPUT_PATH}")

        # Apply max_docs to texts if specified
        max_text_docs = self.config.get("max_docs")
        if max_text_docs is not None and max_text_docs < len(self.text_contents):
            self.text_contents = self.text_contents[:max_text_docs]
            self.applicability_matrix = self.applicability_matrix[:max_text_docs, :]

        # Format all properties
        formatted_properties = [f"property: {prop}" for prop in self.property_names]

        # Format all texts as headlines
        formatted_headlines = [self._format_text(text) for text in self.text_contents]

        # Combine: properties first, then headlines
        self.all_documents = formatted_properties + formatted_headlines
        self.property_start_idx = 0
        self.text_start_idx = len(formatted_properties)

        # Build index mappings
        for doc_idx in range(len(self.all_documents)):
            if doc_idx < self.text_start_idx:
                # This is a property
                property_idx = doc_idx - self.property_start_idx
                self.doc_idx_to_property_idx[doc_idx] = property_idx
            else:
                # This is a text
                text_idx = doc_idx - self.text_start_idx
                self.doc_idx_to_text_idx[doc_idx] = text_idx

        logger.info(
            f"Loaded {len(formatted_properties)} properties and "
            f"{len(formatted_headlines)} headlines = {len(self.all_documents)} total documents"
        )

        return self.all_documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document string

        Returns:
            Text content
        """
        if isinstance(document, dict):
            return document.get("text", str(document))
        return str(document)

    def is_property(self, doc_idx: int) -> bool:
        """Check if document index is a property.

        Args:
            doc_idx: Document index

        Returns:
            True if property, False if text
        """
        return doc_idx in self.doc_idx_to_property_idx

    def is_text(self, doc_idx: int) -> bool:
        """Check if document index is a text.

        Args:
            doc_idx: Document index

        Returns:
            True if text, False if property
        """
        return doc_idx in self.doc_idx_to_text_idx

    def get_applicability(self, property_idx: int, text_idx: int) -> str:
        """Get applicability label for a property-text pair.

        Args:
            property_idx: Property index
            text_idx: Text index

        Returns:
            "applicable" or "not_applicable"
        """
        if property_idx >= len(self.property_names):
            logger.warning(f"Property index {property_idx} out of range")
            return "not_applicable"
        if text_idx >= len(self.text_contents):
            logger.warning(f"Text index {text_idx} out of range")
            return "not_applicable"

        score = self.applicability_matrix[text_idx, property_idx]
        return (
            "applicable" if score >= self.binarization_threshold else "not_applicable"
        )

    def get_document_metadata(self, doc_idx: int) -> dict[str, Any]:
        """Get metadata for a document at the given index.

        Args:
            doc_idx: Document index in all_documents

        Returns:
            Dict with doc_type ("property" or "headline")
        """
        if doc_idx < 0 or doc_idx >= len(self.all_documents):
            return {}

        if self.is_property(doc_idx):
            return {"doc_type": "property"}
        elif self.is_text(doc_idx):
            return {"doc_type": "headline"}

        return {}
