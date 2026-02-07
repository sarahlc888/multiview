"""Met Museum Collection Images docset.

This docset provides access to artworks from the Metropolitan Museum of Art's
open access collection via their public API.

API docs: https://metmuseum.github.io/

Example usage in benchmark config:
    tasks:
      - document_set: met_museum
        criterion: "period_or_movement"
        triplet_style: lm_tags
        config:
          max_docs: 50

Config parameters:
    max_docs (int, optional): Maximum documents to load
    departments (list[str], optional): Filter by department names
    object_ids (list[int], optional): Specific Met object IDs to use
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

MET_API_BASE = "https://collectionapi.metmuseum.org/public/collection/v1"
CACHE_PATH = Path("data/met_museum/objects_cache.json")
IMAGES_DIR = Path("data/met_museum/images")


def _fetch_object(object_id: int) -> dict | None:
    """Fetch a single object from the Met API. Returns None on failure."""
    try:
        resp = requests.get(f"{MET_API_BASE}/objects/{object_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.debug(f"Failed to fetch Met object {object_id}: {e}")
        return None


def _search_public_domain_objects() -> list[int]:
    """Search the Met API for public domain objects with images."""
    params = {"q": "*", "hasImages": "true", "isPublicDomain": "true"}
    try:
        resp = requests.get(f"{MET_API_BASE}/search", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("objectIDs", []) or []
    except Exception as e:
        logger.warning(f"Met API search failed: {e}")
        return []


def _download_image(url: str, object_id: int) -> Path | None:
    """Download an image to the local cache. Returns the path on success, None on failure."""
    dest = IMAGES_DIR / f"{object_id}.jpg"
    if dest.exists():
        return dest
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return dest
    except Exception as e:
        logger.debug(f"Failed to download image for Met object {object_id}: {e}")
        return None


def _fetch_and_download(object_id: int) -> dict | None:
    """Fetch object metadata and download its image. Returns a document dict or None."""
    obj = _fetch_object(object_id)
    if obj is None:
        return None
    image_url = obj.get("primaryImage", "")
    if not image_url:
        return None

    obj_id_val = obj.get("objectID", object_id)
    local_path = _download_image(image_url, obj_id_val)
    if local_path is None:
        return None

    artist = obj.get("artistDisplayName", "")
    title = obj.get("title", "Unknown")
    text = f"{artist} - {title}" if artist else title

    return {
        "image_path": str(local_path),
        "text": text,
        "object_id": obj_id_val,
        "department": obj.get("department", ""),
        "culture": obj.get("culture", ""),
        "period": obj.get("objectDate", ""),
        "medium": obj.get("medium", ""),
        "artist": artist,
        "title": title,
    }


class MetMuseumDocSet(BaseDocSet):
    """Met Museum Collection Images dataset.

    Fetches artworks from the Met's public API, filtering for public domain
    works with available images. Results are cached locally to avoid repeated
    API calls.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        departments (list[str], optional): Filter by department names
        object_ids (list[int], optional): Specific Met object IDs to use
    """

    DATASET_PATH = "data/met_museum"
    DESCRIPTION = "Metropolitan Museum of Art collection images"
    DATASET_NAME = "met_museum"
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[dict]:
        """Load artwork documents from the Met API."""
        max_docs = self.config.get("max_docs", 200)
        object_ids = self.config.get("object_ids")
        departments = self.config.get("departments")

        # Try loading from cache first
        cached = self._load_cache()
        if cached and not object_ids:
            documents = self._filter_documents(cached, departments, max_docs)
            if documents:
                logger.info(f"Loaded {len(documents)} Met Museum artworks from cache")
                return documents

        # Fetch from API
        if object_ids:
            ids_to_fetch = object_ids
        else:
            ids_to_fetch = _search_public_domain_objects()
            if not ids_to_fetch:
                raise RuntimeError(
                    "Met Museum API search returned no results. "
                    "Check network connectivity or try again later."
                )

        # Fetch metadata + download images concurrently
        documents = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_fetch_and_download, obj_id): obj_id
                for obj_id in ids_to_fetch
            }
            for future in as_completed(futures):
                doc = future.result()
                if doc is not None:
                    documents.append(doc)
                    if len(documents) >= max_docs:
                        # Cancel remaining futures and stop collecting
                        for f in futures:
                            f.cancel()
                        break

        if not documents:
            raise RuntimeError(
                "Could not load any Met Museum artworks with public domain images. "
                "Check network connectivity or try again later."
            )

        # Cache for future runs (save all fetched, not just max_docs)
        self._save_cache(documents)

        logger.info(f"Loaded {len(documents)} Met Museum artworks from API")
        return documents

    def _filter_documents(
        self, documents: list[dict], departments: list[str] | None, max_docs: int
    ) -> list[dict]:
        """Filter cached documents by department and max_docs."""
        if departments:
            documents = [d for d in documents if d.get("department") in departments]
        # Verify cached image files still exist on disk
        documents = [d for d in documents if Path(d.get("image_path", "")).exists()]
        return documents[:max_docs]

    def _load_cache(self) -> list[dict] | None:
        """Load cached artwork data if available."""
        if not CACHE_PATH.exists():
            return None
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load Met cache: {e}")
            return None

    def _save_cache(self, documents: list[dict]) -> None:
        """Save artwork data to local cache."""
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Cached {len(documents)} Met artworks to {CACHE_PATH}")
        except OSError as e:
            logger.warning(f"Failed to save Met cache: {e}")

    def get_document_text(self, document: dict) -> str:
        """Extract text description from document."""
        return document.get("text", "")

    def get_document_image(self, document: dict) -> str | None:
        """Extract image URL from document."""
        return document.get("image_path")
