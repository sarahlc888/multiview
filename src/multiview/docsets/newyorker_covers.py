"""New Yorker magazine cover illustrations docset.

Loads cover images from a local directory of scraped New Yorker covers
organized by year. Each cover is a standalone illustration — distinct from
New Yorker caption contest cartoons.

Expected directory structure:
    {images_dir}/{year}/{hex_id}.jpg
    e.g. newyorker_covers/2024/6584427c986226975368f9b6.jpg

Example usage in benchmark config:
    tasks:
      - document_set: newyorker_covers
        criterion: visual_style
        triplet_style: lm_tags
        config:
          max_docs: 100

Config parameters:
    max_docs (int, optional): Maximum documents to load
    images_dir (str, optional): Root directory containing year subdirs
    seed (int, optional): Random seed for sampling (default: 42)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_DIR = "/Users/sarahchen/code/pproj/scrape/newyorker_covers"


class NewYorkerCoversDocSet(BaseDocSet):
    """New Yorker magazine cover illustrations dataset.

    Loads cover images from a local directory organized as {year}/{id}.jpg.
    Spans 1925-2025 (~5,100 images).

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        images_dir (str, optional): Root directory containing year subdirs
        seed (int, optional): Random seed for reproducible sampling (default: 42)
    """

    DATASET_PATH = DEFAULT_IMAGES_DIR
    DESCRIPTION = "New Yorker magazine cover illustrations"
    DOCUMENT_TYPE = "magazine cover illustration"
    DATASET_NAME = "newyorker_covers"
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[dict]:
        """Load cover image documents from year-organized directory."""
        images_dir = Path(self.config.get("images_dir", self.DATASET_PATH))
        max_docs = self.config.get("max_docs")
        seed = self.config.get("seed", 42)

        if not images_dir.is_dir():
            raise RuntimeError(
                f"New Yorker covers directory not found: {images_dir}. "
                f"Set images_dir in config to the correct path."
            )

        # Collect all jpg files from year subdirectories
        documents = []
        for year_dir in sorted(images_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            for img_path in sorted(year_dir.glob("*.jpg")):
                documents.append(
                    {
                        "image_path": str(img_path),
                        "text": f"New Yorker cover, {year}",
                        "year": year,
                        "cover_id": img_path.stem,
                    }
                )

        if not documents:
            raise RuntimeError(
                f"No cover images found in {images_dir}. "
                f"Expected structure: {{year}}/{{id}}.jpg"
            )

        logger.info(
            f"Found {len(documents)} New Yorker covers "
            f"({documents[0]['year']}-{documents[-1]['year']})"
        )

        # Sample if max_docs is set
        if max_docs and max_docs < len(documents):
            rng = random.Random(seed)
            documents = rng.sample(documents, max_docs)
            logger.info(f"Sampled {max_docs} covers (seed={seed})")

        return self._deduplicate(documents)

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document)

    def get_document_image(self, document: Any) -> str | None:
        """Extract image path from a document."""
        if isinstance(document, dict):
            return document.get("image_path")
        return None
