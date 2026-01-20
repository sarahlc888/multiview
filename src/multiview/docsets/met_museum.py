"""Met Museum Collection Images docset.

This docset provides access to artworks from the Metropolitan Museum of Art's
open access collection. Documents are image URLs from Met's collection.

Example usage in benchmark config:
    tasks:
      - document_set: met_museum
        criterion: "artistic_style"
        triplet_style: lm_judge
        config:
          max_docs: 50

        inference_config:
          provider: gemini
          model_name: gemini-2.0-flash-exp
          temperature: 0.0
          parser: text

Config parameters:
    max_docs (int, optional): Maximum documents to load
    object_ids (list[int], optional): Specific Met object IDs to use
"""

from __future__ import annotations

import logging

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


# Sample artworks from Met's collection
DEFAULT_ARTWORKS = [
    (
        "https://images.metmuseum.org/CRDImages/ep/original/DT1567.jpg",
        "Van Gogh - Wheat Field with Cypresses",
    ),
    (
        "https://images.metmuseum.org/CRDImages/ep/original/DP146007.jpg",
        "Vermeer - Young Woman with a Water Pitcher",
    ),
    (
        "https://images.metmuseum.org/CRDImages/ep/original/DT1562.jpg",
        "Monet - The Manneport",
    ),
    (
        "https://images.metmuseum.org/CRDImages/ep/original/DP145470.jpg",
        "Rembrandt - Aristotle with a Bust of Homer",
    ),
]


class MetMuseumDocSet(BaseDocSet):
    """Met Museum Collection Images dataset.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        artworks (list[tuple], optional): List of (image_url, text) tuples
    """

    DATASET_PATH = "data/met_museum"
    DESCRIPTION = "Metropolitan Museum of Art collection images"
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[dict]:
        """Load artwork documents as image URLs."""
        max_docs = self.config.get("max_docs")
        artworks = self.config.get("artworks", DEFAULT_ARTWORKS)

        documents = []
        for image_url, text in artworks[:max_docs]:
            documents.append({"image_path": image_url, "text": text})

        logger.info(f"Loaded {len(documents)} Met Museum artworks")
        return documents

    def get_document_text(self, document: dict) -> str:
        """Extract text description from document."""
        return document.get("text", "")

    def get_document_image(self, document: dict) -> str | None:
        """Extract image URL from document."""
        return document.get("image_path")
