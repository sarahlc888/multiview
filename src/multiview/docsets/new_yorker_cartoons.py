"""New Yorker Caption Contest cartoons docset (text-only).

Loads cartoon caption contest entries from the HuggingFace dataset
jmhessel/newyorker_caption_contest (explanation config). Text-only
analysis using image descriptions and winning captions; cartoon images
available as thumbnails for visualization/dashboard.

Each document contains:
    - text: formatted "Cartoon: {desc}\nCaption: {caption}"
    - image_description: textual description of the cartoon
    - caption: the winning/funny caption
    - contest_number: contest identifier
    - image: PIL Image (for thumbnail display, excluded from serialization)

Example usage in benchmark config:
    tasks:
      - document_set: new_yorker_cartoons
        criterion: humor_move
        triplet_style: lm_tags
        config:
          max_docs: 100

Config parameters:
    max_docs (int, optional): Maximum documents to load
    split (str, optional): Dataset split (default: "train")
    seed (int, optional): Random seed for sampling (default: 42)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

# Cache directory for saved thumbnail images
_THUMBNAIL_DIR = Path(tempfile.gettempdir()) / "newyorker_cartoons_thumbnails"


class NewYorkerCartoonsDocSet(BaseDocSet):
    """New Yorker Caption Contest cartoons (text-only analysis with thumbnails).

    Loads from HuggingFace jmhessel/newyorker_caption_contest, explanation config.
    Analysis is text-based using image descriptions and captions.
    Cartoon images are available as thumbnails for visualization.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str, optional): Dataset split (default: "train")
        seed (int, optional): Random seed for reproducible sampling (default: 42)
    """

    DATASET_PATH = "jmhessel/newyorker_caption_contest"
    DATASET_CONFIG = "explanation"
    DESCRIPTION = "New Yorker Caption Contest cartoons"
    DOCUMENT_TYPE = "cartoon caption"
    DATASET_NAME = "new_yorker_cartoons"
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[dict]:
        """Load caption contest entries from HuggingFace."""
        logger.info(
            f"Loading New Yorker Caption Contest from HuggingFace: "
            f"{self.DATASET_PATH} ({self.DATASET_CONFIG})"
        )

        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        seed = self.config.get("seed", 42)
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(
                self.DATASET_PATH,
                self.DATASET_CONFIG,
                split=split,
                streaming=True,
            )
            dataset = dataset.shuffle(seed=seed, buffer_size=10000).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, self.DATASET_CONFIG, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=seed)

        documents = []
        for example in dataset:
            image_description = example.get("image_description", "")
            caption = example.get("caption_choices", "")
            # caption_choices is the winning caption in explanation config
            if isinstance(caption, list):
                caption = caption[0] if caption else ""

            if not image_description and not caption:
                continue

            desc = image_description.strip()
            cap = caption.strip()
            # Build text field for downstream pipeline compatibility
            parts = []
            if desc:
                parts.append(f"Cartoon: {desc}")
            if cap:
                parts.append(f"Caption: {cap}")

            # Save thumbnail to disk for visualization pipeline
            contest_num = example.get("contest_number", "")
            image_path = self._save_thumbnail(example.get("image"), contest_num)

            doc = {
                "text": "\n".join(parts),
                "image_description": desc,
                "caption": cap,
                "contest_number": contest_num,
                "image_path": image_path,
            }
            documents.append(doc)

            if (
                not use_streaming
                and max_docs is not None
                and len(documents) >= max_docs
            ):
                break

        logger.debug(
            f"Loaded {len(documents)} documents from New Yorker Caption Contest"
        )
        return self._deduplicate(documents)

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document)

    def get_document_image(self, document: Any) -> str | None:
        """Return thumbnail image path from document."""
        if isinstance(document, dict):
            return document.get("image_path")
        return None

    @staticmethod
    def _save_thumbnail(pil_image, contest_num) -> str | None:
        """Save PIL image to disk and return path."""
        if pil_image is None:
            return None
        key = f"nyc_{contest_num}" if contest_num else str(id(pil_image))
        path = _THUMBNAIL_DIR / f"{key}.jpg"
        if path.exists():
            return str(path)
        try:
            _THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
            pil_image.save(str(path), format="JPEG")
            return str(path)
        except Exception as e:
            logger.debug(f"Failed to save thumbnail for contest {contest_num}: {e}")
            return None
