"""UT-Zappos-50k shoe dataset.

This docset provides access to UT-Zappos shoe images via HuggingFace Datasets.
Uses streaming mode to avoid downloading the full dataset to disk.

Dataset source:
    - HuggingFace: XuVV/ut-zappos-rl (29k images, streaming)
    - Original: UT-Zappos-50k dataset (50,025 images)

Dataset reference:
    "Identifying Ambiguous Similarity Conditions via Semantic Matching"
    Ye et al., CVPR 2022
    https://arxiv.org/abs/2204.04053

Available attributes:
    - functional_type: Category (shoes, sandals, slippers, boots)
    - closure: Mechanism to enclose the foot
    - gender: Recommended gender
    - heel_height: Height of the heel (flat, high)
    - material: Materials used (leather, suede, canvas, etc.)
    - toe_style: Style at the front of the shoe

Example usage in benchmark config:
    tasks:
      - document_set: ut_zappos50k
        criterion: "heel_height"
        triplet_style: lm_judge
        config:
          max_docs: 50  # Streams 50 images, no disk usage

        inference_config:
          provider: gemini
          model_name: gemini-2.5-flash
          temperature: 0.5
          parser: json

Config parameters:
    max_docs (int, optional): Maximum documents to load
    shoes (list[dict], optional): Custom shoe data to use instead of HF dataset

Requirements:
    pip install datasets  # For HuggingFace streaming
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


# Sample shoes for testing when HuggingFace dataset is not available
# NOTE: These are simple placeholder URLs for infrastructure testing only
# The real dataset is loaded from HuggingFace: XuVV/ut-zappos-rl (streaming mode)
SAMPLE_SHOES = [
    {
        "image_url": "https://picsum.photos/seed/boot1/400/400",
        "text": "Black leather ankle boot with side zipper",
        "functional_type": "boots",
        "closure": "zipper",
        "gender": "women",
        "heel_height": "medium",
        "material": "leather",
        "toe_style": "round",
    },
    {
        "image_url": "https://picsum.photos/seed/sandal1/400/400",
        "text": "Strappy flat sandal with buckle",
        "functional_type": "sandals",
        "closure": "buckle",
        "gender": "women",
        "heel_height": "flat",
        "material": "synthetic",
        "toe_style": "open",
    },
    {
        "image_url": "https://picsum.photos/seed/sneaker1/400/400",
        "text": "White athletic sneaker with laces",
        "functional_type": "shoes",
        "closure": "laces",
        "gender": "unisex",
        "heel_height": "flat",
        "material": "canvas",
        "toe_style": "round",
    },
    {
        "image_url": "https://picsum.photos/seed/heel1/400/400",
        "text": "Red stiletto pump with pointed toe",
        "functional_type": "shoes",
        "closure": "slip-on",
        "gender": "women",
        "heel_height": "high",
        "material": "patent_leather",
        "toe_style": "pointed",
    },
    {
        "image_url": "https://picsum.photos/seed/loafer1/400/400",
        "text": "Brown leather loafer slip-on",
        "functional_type": "shoes",
        "closure": "slip-on",
        "gender": "men",
        "heel_height": "flat",
        "material": "leather",
        "toe_style": "round",
    },
    {
        "image_url": "https://picsum.photos/seed/boot2/400/400",
        "text": "Tan suede knee-high boot with block heel",
        "functional_type": "boots",
        "closure": "zipper",
        "gender": "women",
        "heel_height": "medium",
        "material": "suede",
        "toe_style": "almond",
    },
    {
        "image_url": "https://picsum.photos/seed/slipper1/400/400",
        "text": "Fuzzy house slipper with elastic",
        "functional_type": "slippers",
        "closure": "slip-on",
        "gender": "unisex",
        "heel_height": "flat",
        "material": "fabric",
        "toe_style": "round",
    },
    {
        "image_url": "https://picsum.photos/seed/sandal2/400/400",
        "text": "Leather slide sandal with cork sole",
        "functional_type": "sandals",
        "closure": "slip-on",
        "gender": "unisex",
        "heel_height": "flat",
        "material": "leather",
        "toe_style": "open",
    },
    {
        "image_url": "https://picsum.photos/seed/oxford1/400/400",
        "text": "Black patent oxford with laces",
        "functional_type": "shoes",
        "closure": "laces",
        "gender": "men",
        "heel_height": "flat",
        "material": "patent_leather",
        "toe_style": "square",
    },
    {
        "image_url": "https://picsum.photos/seed/wedge1/400/400",
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
    DATASET_NAME = "ut_zappos50k"

    # No known criteria - all attributes should be annotated via vision/LM
    # (metadata is available but not used for automatic extraction)
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[dict]:
        """Load shoe documents with images and attributes.

        Returns:
            List of shoe documents with text, image_path, and attribute fields
        """
        max_docs = self.config.get("max_docs")
        shoes = self.config.get("shoes")

        # If custom shoes provided, use them (for testing)
        if shoes is not None:
            documents = []
            for shoe in shoes:
                documents.append(
                    {
                        "text": "<image>",  # Placeholder for vision-only evaluation
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

            logger.info(f"Loaded {len(documents)} custom UT-Zappos-50k shoe documents")
            return documents

        # Otherwise, load from actual dataset files
        return self._load_from_dataset(max_docs)

    def _load_from_dataset(self, max_docs: int | None = None) -> list[dict]:
        """Load documents from HuggingFace UT-Zappos dataset (streaming).

        Uses XuVV/ut-zappos-rl dataset which has 29k images with attributes.
        Streams data without downloading to disk.

        Args:
            max_docs: Maximum number of documents to load

        Returns:
            List of shoe documents
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning(
                "HuggingFace datasets package not installed. "
                "Install with: pip install datasets"
            )
            return self._load_sample_shoes(max_docs)

        try:
            logger.info("Loading UT-Zappos dataset from HuggingFace (streaming mode)")

            # Load dataset in streaming mode (no disk usage)
            dataset = load_dataset(
                "XuVV/ut-zappos-rl",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            documents = []
            for _i, example in enumerate(dataset):
                # Extract image from HF dataset
                # Images are stored as PIL images in HF datasets
                image = example.get("images")
                if isinstance(image, list) and len(image) > 0:
                    image = image[0]  # Take first image if it's a list

                # Get label/answer for attributes
                answer = example.get("answer", "")

                # Parse attributes from answer field
                # The answer field contains categories like "Boots.Ankle", "Leather", etc.
                attributes = self._parse_attributes(answer)

                # Convert PIL image to data URI for immediate use
                # This avoids saving to disk
                image_data_uri = self._pil_to_data_uri(image)

                documents.append(
                    {
                        "text": "<image>",  # Placeholder for vision-only evaluation
                        "image_path": image_data_uri,  # Data URI, not file path
                        "label": answer,
                        "functional_type": attributes.get("functional_type"),
                        "closure": attributes.get("closure"),
                        "gender": attributes.get("gender"),
                        "heel_height": attributes.get("heel_height"),
                        "material": attributes.get("material"),
                        "toe_style": attributes.get("toe_style"),
                    }
                )

                # Check max_docs limit
                if max_docs and len(documents) >= max_docs:
                    break

            logger.info(
                f"Loaded {len(documents)} documents from HuggingFace UT-Zappos dataset"
            )
            return documents

        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}. Using fallback.")
            return self._load_sample_shoes(max_docs)

    def _parse_attributes(self, answer: str) -> dict:
        """Parse attributes from the answer field.

        The answer field contains labels like:
        - "Boots.Ankle", "Sandals.Flats", "Shoes.Oxfords"
        - "Leather", "Suede", "Canvas"

        Args:
            answer: Label string from dataset

        Returns:
            Dict of parsed attributes
        """
        attributes = {}

        # Parse functional type from patterns like "Boots.Ankle"
        if "Boots" in answer:
            attributes["functional_type"] = "boots"
        elif "Sandals" in answer:
            attributes["functional_type"] = "sandals"
        elif "Shoes" in answer:
            attributes["functional_type"] = "shoes"
        elif "Slippers" in answer:
            attributes["functional_type"] = "slippers"

        # Parse material
        materials = {
            "Leather": "leather",
            "Suede": "suede",
            "Canvas": "canvas",
            "Patent": "patent_leather",
            "Sheepskin": "sheepskin",
        }
        for mat_key, mat_val in materials.items():
            if mat_key in answer:
                attributes["material"] = mat_val
                break

        # Parse heel height from subcategory
        if "Flats" in answer:
            attributes["heel_height"] = "flat"
        elif "Heels" in answer or "Heel" in answer:
            attributes["heel_height"] = "high"

        return attributes

    def _pil_to_data_uri(self, image) -> str:
        """Convert PIL image to data URI.

        Args:
            image: PIL Image object

        Returns:
            Data URI string (e.g., "data:image/jpeg;base64,...")
        """
        import base64
        from io import BytesIO

        # Convert PIL image to bytes
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()

        # Encode to base64
        b64_data = base64.b64encode(image_bytes).decode("utf-8")

        # Return data URI
        return f"data:image/jpeg;base64,{b64_data}"

    def _load_sample_shoes(self, max_docs: int | None = None) -> list[dict]:
        """Load sample shoes as fallback.

        Args:
            max_docs: Maximum number of documents

        Returns:
            List of sample shoe documents
        """
        documents = []
        for shoe in SAMPLE_SHOES:
            documents.append(
                {
                    "text": "<image>",
                    "image_path": shoe["image_url"],
                    "functional_type": shoe.get("functional_type"),
                    "closure": shoe.get("closure"),
                    "gender": shoe.get("gender"),
                    "heel_height": shoe.get("heel_height"),
                    "material": shoe.get("material"),
                    "toe_style": shoe.get("toe_style"),
                }
            )

            if max_docs and len(documents) >= max_docs:
                break

        logger.info(f"Loaded {len(documents)} sample shoe documents")
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
