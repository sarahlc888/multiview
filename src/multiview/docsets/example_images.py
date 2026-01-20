"""Example image dataset for VLM evaluation.

This is a minimal example docset demonstrating how to use vision-language models
with the multiview framework. Replace this with your own image dataset.

Example usage:
    tasks:
      - document_set: example_images
        criterion: "artistic_style"
        triplet_style: lm_judge
        config:
          max_docs: 10

        inference_config:
          provider: gemini
          model_name: gemini-2.0-flash-exp
          prompt_template: "Describe the artistic style shown in this image."
          temperature: 0.0
          parser: text
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class ExampleImagesDocSet(BaseDocSet):
    """Example image dataset with simple text descriptions.

    This is a placeholder docset for demonstration purposes.
    To use this with real images:
    1. Place images in data/example_images/ directory
    2. Update SAMPLE_IMAGES list with your image filenames
    3. Update descriptions to match your images

    Alternatively, use image URLs instead of local paths.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        images_dir (str, optional): Directory containing images (default: data/example_images)
    """

    DATASET_PATH = "data/example_images"
    DESCRIPTION = "Example image dataset for VLM demonstration"
    KNOWN_CRITERIA = []  # Vision tasks typically require LM-based annotation

    # Sample images (replace with your own)
    SAMPLE_IMAGES = [
        {
            "filename": "sample1.jpg",
            "description": "A landscape with mountains",
            "category": "nature",
        },
        {
            "filename": "sample2.jpg",
            "description": "An abstract painting with geometric shapes",
            "category": "abstract_art",
        },
        {
            "filename": "sample3.jpg",
            "description": "A portrait photograph",
            "category": "portrait",
        },
        {
            "filename": "sample4.jpg",
            "description": "An architectural building facade",
            "category": "architecture",
        },
        {
            "filename": "sample5.jpg",
            "description": "A still life with fruits",
            "category": "still_life",
        },
    ]

    def __init__(self, config: dict | None = None):
        """Initialize Example Images dataset.

        Config params:
            max_docs: Maximum documents to load (optional)
            images_dir: Directory containing images (default: data/example_images)
            use_urls: If True, treat filenames as URLs instead of local paths
        """
        super().__init__(config)
        self.images_dir = Path(self.config.get("images_dir", self.DATASET_PATH))
        self.use_urls = self.config.get("use_urls", False)

    def load_documents(self) -> list[dict]:
        """Load image documents with metadata.

        Returns:
            List of document dicts with "text", "image_path", and "category" fields
        """
        max_docs = self.config.get("max_docs")

        documents = []
        for item in self.SAMPLE_IMAGES:
            # Build image path/URL
            if self.use_urls:
                # Treat filename as URL directly
                image_source = item["filename"]
            else:
                # Build local file path
                image_source = str(self.images_dir / item["filename"])

            doc = {
                "text": item["description"],
                "image_path": image_source,
                "category": item["category"],
            }

            documents.append(doc)

            # Check max_docs limit
            if max_docs and len(documents) >= max_docs:
                break

        logger.info(f"Loaded {len(documents)} example image documents")

        return documents

    def get_document_text(self, document: dict) -> str:
        """Extract text description from document.

        Args:
            document: Document dict

        Returns:
            Text description of the image
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document)

    def get_document_image(self, document: dict) -> str | None:
        """Extract image path/URL from document.

        Args:
            document: Document dict

        Returns:
            Image source (path or URL), or None if not found
        """
        if isinstance(document, dict):
            return document.get("image_path")
        return None

    def get_known_criterion_value(self, document: dict, criterion: str) -> Any:
        """Get the known criterion value for a document.

        Args:
            document: Document dict
            criterion: Criterion name (e.g., "category")

        Returns:
            Criterion value or None
        """
        # Category is a known criterion
        if isinstance(document, dict) and criterion == "category":
            return document.get("category")

        return None
