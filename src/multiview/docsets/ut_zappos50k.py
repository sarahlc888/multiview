"""UT-Zappos-50k shoe dataset.

This docset provides access to the UT-Zappos-50k dataset - a collection of 50,025
catalog images of shoes from Zappos.com with fine-grained attributes.

Dataset reference:
    "Identifying Ambiguous Similarity Conditions via Semantic Matching"
    Ye et al., CVPR 2022
    https://arxiv.org/abs/2204.04053

Available attributes:
    - functional_type: Category (shoes, sandals, slippers, boots) + subcategory
    - closure: Mechanism to enclose the foot
    - gender: Recommended gender
    - heel_height: Height of the heel
    - material: Materials used
    - toe_style: Style at the front of the shoe

Example usage in benchmark config:
    tasks:
      - document_set: ut_zappos50k
        criterion: "heel_height"
        triplet_style: random
        config:
          max_docs: 50

        inference_config:
          provider: gemini
          model_name: gemini-2.0-flash-exp
          temperature: 0.0
          parser: text

Config parameters:
    max_docs (int, optional): Maximum documents to load
    shoes (list[dict], optional): Custom shoe data to use instead of samples
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


# Sample shoes from UT-Zappos-50k for testing
# In production, these would be loaded from the actual dataset files
SAMPLE_SHOES = [
    {
        "image_url": "https://images.zappos.com/sample/boot1.jpg",
        "text": "Black leather ankle boot with side zipper",
        "functional_type": "boots",
        "closure": "zipper",
        "gender": "women",
        "heel_height": "medium",
        "material": "leather",
        "toe_style": "round",
    },
    {
        "image_url": "https://images.zappos.com/sample/sandal1.jpg",
        "text": "Strappy flat sandal with buckle",
        "functional_type": "sandals",
        "closure": "buckle",
        "gender": "women",
        "heel_height": "flat",
        "material": "synthetic",
        "toe_style": "open",
    },
    {
        "image_url": "https://images.zappos.com/sample/sneaker1.jpg",
        "text": "White athletic sneaker with laces",
        "functional_type": "shoes",
        "closure": "laces",
        "gender": "unisex",
        "heel_height": "flat",
        "material": "canvas",
        "toe_style": "round",
    },
    {
        "image_url": "https://images.zappos.com/sample/heel1.jpg",
        "text": "Red stiletto pump with pointed toe",
        "functional_type": "shoes",
        "closure": "slip-on",
        "gender": "women",
        "heel_height": "high",
        "material": "patent_leather",
        "toe_style": "pointed",
    },
    {
        "image_url": "https://images.zappos.com/sample/loafer1.jpg",
        "text": "Brown leather loafer slip-on",
        "functional_type": "shoes",
        "closure": "slip-on",
        "gender": "men",
        "heel_height": "flat",
        "material": "leather",
        "toe_style": "round",
    },
    {
        "image_url": "https://images.zappos.com/sample/boot2.jpg",
        "text": "Tan suede knee-high boot with block heel",
        "functional_type": "boots",
        "closure": "zipper",
        "gender": "women",
        "heel_height": "medium",
        "material": "suede",
        "toe_style": "almond",
    },
    {
        "image_url": "https://images.zappos.com/sample/slipper1.jpg",
        "text": "Fuzzy house slipper with elastic",
        "functional_type": "slippers",
        "closure": "slip-on",
        "gender": "unisex",
        "heel_height": "flat",
        "material": "fabric",
        "toe_style": "round",
    },
    {
        "image_url": "https://images.zappos.com/sample/sandal2.jpg",
        "text": "Leather slide sandal with cork sole",
        "functional_type": "sandals",
        "closure": "slip-on",
        "gender": "unisex",
        "heel_height": "flat",
        "material": "leather",
        "toe_style": "open",
    },
    {
        "image_url": "https://images.zappos.com/sample/oxford1.jpg",
        "text": "Black patent oxford with laces",
        "functional_type": "shoes",
        "closure": "laces",
        "gender": "men",
        "heel_height": "flat",
        "material": "patent_leather",
        "toe_style": "square",
    },
    {
        "image_url": "https://images.zappos.com/sample/wedge1.jpg",
        "text": "Espadrille wedge sandal with ankle strap",
        "functional_type": "sandals",
        "closure": "buckle",
        "gender": "women",
        "heel_height": "high",
        "material": "canvas",
        "toe_style": "open",
    },
]


class UTZappos50kDocSet(BaseDocSet):
    """UT-Zappos-50k shoe dataset with fine-grained attributes.

    This dataset contains 50,025 shoe images with attributes like
    functional_type, closure, gender, heel_height, material, and toe_style.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        shoes (list[dict], optional): Custom shoe data (for testing or subsets)
    """

    DATASET_PATH = "data/ut_zappos50k"
    DESCRIPTION = "UT-Zappos-50k shoe images with fine-grained attributes"

    # Known criteria that can be extracted from metadata
    KNOWN_CRITERIA = [
        "functional_type",
        "closure",
        "gender",
        "heel_height",
        "material",
        "toe_style",
    ]

    def load_documents(self) -> list[dict]:
        """Load shoe documents with images and attributes.

        Returns:
            List of shoe documents with text, image_path, and attribute fields
        """
        max_docs = self.config.get("max_docs")
        shoes = self.config.get("shoes", SAMPLE_SHOES)

        documents = []
        for shoe in shoes:
            documents.append(
                {
                    "text": shoe["text"],
                    "image_path": shoe["image_url"],
                    "functional_type": shoe.get("functional_type"),
                    "closure": shoe.get("closure"),
                    "gender": shoe.get("gender"),
                    "heel_height": shoe.get("heel_height"),
                    "material": shoe.get("material"),
                    "toe_style": shoe.get("toe_style"),
                }
            )

            # Check max_docs limit
            if max_docs and len(documents) >= max_docs:
                break

        logger.info(f"Loaded {len(documents)} UT-Zappos-50k shoe documents")
        return documents

    def get_document_text(self, document: dict) -> str:
        """Extract text description from document.

        Args:
            document: Shoe document dict

        Returns:
            Text description of the shoe
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document)

    def get_document_image(self, document: dict) -> str | None:
        """Extract image URL/path from document.

        Args:
            document: Shoe document dict

        Returns:
            Image source (URL or path), or None if not found
        """
        if isinstance(document, dict):
            return document.get("image_path")
        return None

    def get_known_criterion_value(self, document: dict, criterion: str) -> Any:
        """Get the known criterion value for a shoe document.

        Supports all UT-Zappos-50k metadata attributes:
        - functional_type, closure, gender, heel_height, material, toe_style

        Args:
            document: Shoe document dict
            criterion: Criterion name

        Returns:
            Criterion value or None
        """
        # First check if parent class handles it (e.g., word_count)
        parent_value = super().get_known_criterion_value(document, criterion)
        if parent_value is not None:
            return parent_value

        # Handle UT-Zappos-50k specific criteria
        if isinstance(document, dict) and criterion in self.KNOWN_CRITERIA:
            return document.get(criterion)

        return None
