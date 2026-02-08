"""Whole Earth Catalog cover images docset.

Loads cover images from Archive.org scraping data. The source JSON
contains ~515 image entries scraped from Whole Earth Catalog
digitized editions on Archive.org (1968-1998).

Source JSON path (default):
    /Users/sarahchen/code/pproj/bounding_bosch/scraping_data/data/whole_earth/output/images_run2.json

Example usage in benchmark config:
    tasks:
      - document_set: whole_earth_catalog
        criterion: visual_style
        triplet_style: lm_tags
        config:
          max_docs: 50

Config parameters:
    max_docs (int, optional): Maximum documents to load
    source_json (str, optional): Path to scraped images JSON file
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_JSON = "/Users/sarahchen/code/pproj/bounding_bosch/scraping_data/data/whole_earth/output/images_run2.json"
CACHE_PATH = Path("data/whole_earth_catalog/documents_cache.json")
IMAGES_DIR = Path("data/whole_earth_catalog/images")


def _extract_archive_id(url: str) -> str:
    """Extract the archive.org item slug from a download URL.

    Example:
        https://archive.org/download/wholeearthcatalo00unse_8/page/cover_medium.jpg
        -> wholeearthcatalo00unse_8
    """
    parts = urlparse(url).path.split("/")
    # URL format: /download/{archive_id}/page/cover_medium.jpg
    try:
        dl_idx = parts.index("download")
        return parts[dl_idx + 1]
    except (ValueError, IndexError):
        # Fallback: use the third path segment
        return parts[2] if len(parts) > 2 else url.split("/")[-1]


def _download_image(url: str, archive_id: str) -> Path | None:
    """Download an image to the local cache. Returns the path on success, None on failure."""
    dest = IMAGES_DIR / f"{archive_id}.jpg"
    if dest.exists():
        return dest
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        logger.debug(f"Failed to download image for {archive_id}: {e}")
        return None


class WholeEarthCatalogDocSet(BaseDocSet):
    """Whole Earth Catalog cover images dataset.

    Loads cover images from Archive.org scraping data, downloading and
    caching images locally. Results are cached to avoid repeated downloads.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        source_json (str, optional): Path to scraped images JSON file
    """

    DATASET_PATH = "data/whole_earth_catalog"
    DESCRIPTION = "Whole Earth Catalog cover images (1968-1998)"
    DATASET_NAME = "whole_earth_catalog"
    DOCUMENT_TYPE = "magazine cover"
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[dict]:
        """Load cover image documents from Archive.org scraping data."""
        max_docs = self.config.get("max_docs")
        source_json = Path(self.config.get("source_json", DEFAULT_SOURCE_JSON))

        # Try loading from cache first
        cached = self._load_cache()
        if cached:
            # Verify cached image files still exist on disk
            documents = [d for d in cached if Path(d.get("image_path", "")).exists()]
            if max_docs:
                documents = documents[:max_docs]
            if documents:
                logger.info(
                    f"Loaded {len(documents)} Whole Earth Catalog covers from cache"
                )
                return self._deduplicate(documents)

        # Load source JSON
        if not source_json.exists():
            raise RuntimeError(
                f"Source JSON not found: {source_json}. "
                f"Set source_json in config to the correct path."
            )

        with open(source_json, encoding="utf-8") as f:
            entries = json.load(f)

        logger.info(f"Loading {len(entries)} entries from {source_json}")

        # Download images concurrently (lower workers for Archive.org rate limiting)
        documents = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            for entry in entries:
                url = entry.get("source", "")
                if not url:
                    continue
                archive_id = _extract_archive_id(url)
                future = pool.submit(_download_image, url, archive_id)
                futures[future] = (entry, archive_id, url)

            for future in as_completed(futures):
                entry, archive_id, url = futures[future]
                local_path = future.result()
                if local_path is None:
                    continue

                documents.append(
                    {
                        "image_path": str(local_path),
                        "text": archive_id,
                        "source_url": url,
                        "index": entry.get("index", 0),
                        "archive_id": archive_id,
                    }
                )

                if max_docs and len(documents) >= max_docs:
                    for f in futures:
                        f.cancel()
                    break

        if not documents:
            raise RuntimeError(
                "Could not load any Whole Earth Catalog cover images. "
                "Check network connectivity or try again later."
            )

        # Sort by index for consistent ordering
        documents.sort(key=lambda d: d["index"])

        # Cache for future runs
        self._save_cache(documents)

        logger.info(f"Loaded {len(documents)} Whole Earth Catalog covers")
        return self._deduplicate(documents)

    def _load_cache(self) -> list[dict] | None:
        """Load cached document data if available."""
        if not CACHE_PATH.exists():
            return None
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load Whole Earth Catalog cache: {e}")
            return None

    def _save_cache(self, documents: list[dict]) -> None:
        """Save document data to local cache."""
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            logger.info(
                f"Cached {len(documents)} Whole Earth Catalog covers to {CACHE_PATH}"
            )
        except OSError as e:
            logger.warning(f"Failed to save Whole Earth Catalog cache: {e}")

    def get_document_text(self, document: dict) -> str:
        """Extract text description from document."""
        return document.get("text", "")

    def get_document_image(self, document: dict) -> str | None:
        """Extract image path from document."""
        return document.get("image_path")
