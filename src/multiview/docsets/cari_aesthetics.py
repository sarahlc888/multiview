"""CARI Aesthetics docset.

Loads aesthetic movement images from Consumer Aesthetics Research Institute
(CARI) data scraped from Are.na galleries. Each gallery corresponds to a named
aesthetic movement (~90 movements, ~3,185 images total).

Data source: Local gallery JSONs + aesthetics.csv for the similarity graph.

Example usage in benchmark config:
    tasks:
      - document_set: cari_aesthetics
        criterion: aesthetic_name
        triplet_style: lm_tags
        config:
          max_docs: 100

Config parameters:
    max_docs (int, optional): Maximum documents to load
    seed (int, optional): Random seed for sampling (default: 42)
"""

from __future__ import annotations

import csv
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

DATASET_PATH = "/Users/sarahchen/code/pproj/bounding_bosch/scraping_data/data/cari"
GALLERIES_DIR = Path(DATASET_PATH) / "galleries"
AESTHETICS_CSV = Path(DATASET_PATH) / "aesthetics.csv"
CACHE_PATH = Path("data/cari_aesthetics/documents_cache.json")
IMAGES_DIR = Path("data/cari_aesthetics/images")


def _download_image(url: str, arena_id: int) -> Path | None:
    """Download an image to the local cache. Returns the path on success."""
    dest = IMAGES_DIR / f"{arena_id}.jpg"
    if dest.exists():
        return dest
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        logger.debug(f"Failed to download image for arena block {arena_id}: {e}")
        return None


def _load_aesthetics_map() -> dict[str, str]:
    """Load aesthetic slug -> display name mapping from CSV."""
    mapping = {}
    if not AESTHETICS_CSV.exists():
        return mapping
    with open(AESTHETICS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("aesthetic_url_slug", "")
            name = row.get("aesthetic_name", "")
            if slug and name:
                mapping[slug] = name
    return mapping


class CARIAestheticsDocSet(BaseDocSet):
    """CARI aesthetic movement images from Are.na galleries.

    Loads images from locally scraped Are.na gallery JSONs, one per aesthetic
    movement. Images are downloaded and cached locally on first use.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        seed (int, optional): Random seed for reproducible sampling (default: 42)
    """

    DATASET_PATH = DATASET_PATH
    DESCRIPTION = "CARI aesthetic movement images from Are.na galleries"
    DOCUMENT_TYPE = "image"
    DATASET_NAME = "cari_aesthetics"
    KNOWN_CRITERIA = ["aesthetic_name"]

    def load_documents(self) -> list[dict]:
        """Load aesthetic movement image documents."""
        max_docs = self.config.get("max_docs")
        seed = self.config.get("seed", 42)

        # Try loading from cache first
        cached = self._load_cache()
        if cached:
            # Verify cached image files still exist
            documents = [d for d in cached if Path(d.get("image_path", "")).exists()]
            if documents:
                if max_docs and max_docs < len(documents):
                    rng = random.Random(seed)
                    documents = rng.sample(documents, max_docs)
                logger.info(f"Loaded {len(documents)} CARI aesthetic images from cache")
                return self._deduplicate(documents)

        # Load from gallery JSONs
        aesthetics_map = _load_aesthetics_map()
        raw_items = self._collect_gallery_items(aesthetics_map)

        if not raw_items:
            raise RuntimeError(
                f"No gallery items found in {GALLERIES_DIR}. "
                f"Check that gallery JSON files exist."
            )

        # Download images concurrently
        documents = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_download_image, item["url"], item["arena_id"]): item
                for item in raw_items
            }
            for future in as_completed(futures):
                item = futures[future]
                local_path = future.result()
                if local_path is not None:
                    documents.append(
                        {
                            "image_path": str(local_path),
                            "text": f"{item['title']} — {item['aesthetic_name']}",
                            "aesthetic_name": item["aesthetic_name"],
                            "title": item["title"],
                            "description": item["description"],
                            "arena_id": item["arena_id"],
                        }
                    )

        if not documents:
            raise RuntimeError(
                "Could not download any CARI aesthetic images. "
                "Check network connectivity or try again later."
            )

        # Cache all documents for future runs
        self._save_cache(documents)

        logger.info(f"Loaded {len(documents)} CARI aesthetic images from galleries")

        # Sample if max_docs is set
        if max_docs and max_docs < len(documents):
            rng = random.Random(seed)
            documents = rng.sample(documents, max_docs)
            logger.info(f"Sampled {max_docs} CARI images (seed={seed})")

        return self._deduplicate(documents)

    def _collect_gallery_items(self, aesthetics_map: dict[str, str]) -> list[dict]:
        """Parse all gallery JSONs and collect downloadable image items."""
        items = []

        for gallery_path in sorted(GALLERIES_DIR.glob("*.json")):
            slug = gallery_path.stem
            aesthetic_name = aesthetics_map.get(slug, slug.replace("-", " ").title())

            try:
                with open(gallery_path, encoding="utf-8") as f:
                    gallery = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load gallery {gallery_path}: {e}")
                continue

            for entry in gallery:
                # Skip videos/media — we only want static images
                if entry.get("class") == "Media":
                    continue

                image_url = entry.get("image_original", "")
                arena_id = entry.get("id")
                if not image_url or not arena_id:
                    continue

                items.append(
                    {
                        "url": image_url,
                        "arena_id": arena_id,
                        "aesthetic_name": aesthetic_name,
                        "title": entry.get("title", ""),
                        "description": entry.get("description", ""),
                    }
                )

        return items

    def _load_cache(self) -> list[dict] | None:
        """Load cached document data if available."""
        if not CACHE_PATH.exists():
            return None
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load CARI cache: {e}")
            return None

    def _save_cache(self, documents: list[dict]) -> None:
        """Save document data to local cache."""
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(documents)} CARI documents to {CACHE_PATH}")
        except OSError as e:
            logger.warning(f"Failed to save CARI cache: {e}")

    def get_document_text(self, document: dict) -> str:
        """Extract text description from document."""
        return document.get("text", "")

    def get_document_image(self, document: dict) -> str | None:
        """Extract image path from document."""
        return document.get("image_path")

    def get_known_criterion_value(self, document: dict, criterion: str):
        """Extract criterion value for known criteria."""
        if criterion == "aesthetic_name":
            return document.get("aesthetic_name", "")
        return super().get_known_criterion_value(document, criterion)
