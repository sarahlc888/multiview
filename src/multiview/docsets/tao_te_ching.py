"""Tao Te Ching document set loader.

Fetches the Ursula K. Le Guin translation of the Tao Te Ching from GitHub
and parses it into 81 individual chapters/verses.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import requests

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

# Tao Te Ching dataset configuration
TAO_SOURCE_URL = (
    "https://raw.githubusercontent.com/nrrb/tao-te-ching/master/"
    "Ursula%20K%20Le%20Guin.md"
)
TAO_CACHE_DIR = Path.home() / ".cache" / "multiview" / "tao_te_ching"
TAO_CHAPTERS_PATH = TAO_CACHE_DIR / "chapters.pkl"


def fetch_and_parse_chapters(url: str = TAO_SOURCE_URL) -> list[dict]:
    """Fetch the Tao Te Ching markdown and parse into individual chapters.

    Returns:
        List of dicts with keys: chapter_number, title, text
    """
    logger.info("Fetching Tao Te Ching markdown from GitHub...")
    response = requests.get(url)
    response.raise_for_status()
    content = response.text

    # Split on chapter headings (# 1, # 2, ..., # 81)
    # Each chapter starts with "# <number>\n## <TITLE>"
    chapter_pattern = re.compile(r"^# (\d+)\s*$", re.MULTILINE)
    splits = list(chapter_pattern.finditer(content))

    chapters = []
    for i, match in enumerate(splits):
        chapter_number = int(match.group(1))
        start = match.end()
        if i + 1 < len(splits):
            end = splits[i + 1].start()
        else:
            # For the last chapter, stop at the end-of-book notes section
            notes_match = re.search(r"^# [A-Z]", content[start:], re.MULTILINE)
            end = start + notes_match.start() if notes_match else len(content)
        raw = content[start:end].strip()

        # Extract title from ## heading
        title = ""
        title_match = re.match(r"^## (.+)", raw)
        if title_match:
            title = title_match.group(1).strip()
            raw = raw[title_match.end() :].strip()

        # Separate the poem text from the translator's note
        # Notes start with "> **Note**"
        note = ""
        note_match = re.search(r"^> \*\*Note\*\*", raw, re.MULTILINE)
        if note_match:
            poem_text = raw[: note_match.start()].strip()
            note = raw[note_match.start() :].strip()
        else:
            poem_text = raw

        # Clean up trailing whitespace from lines
        poem_text = "\n".join(line.rstrip() for line in poem_text.split("\n"))
        # Collapse runs of 3+ blank lines down to 2
        poem_text = re.sub(r"\n{3,}", "\n\n", poem_text).strip()

        chapters.append(
            {
                "chapter_number": chapter_number,
                "title": title,
                "text": poem_text,
                "note": note,
            }
        )

    logger.info(f"Parsed {len(chapters)} chapters")
    return chapters


class TaoTeChingDocSet(BaseDocSet):
    """Tao Te Ching (Ursula K. Le Guin translation) document set."""

    DATASET_PATH = str(TAO_CHAPTERS_PATH)
    DESCRIPTION = "Tao Te Ching passages (Ursula K. Le Guin translation)"
    DOCUMENT_TYPE = "philosophical verse"
    DATASET_NAME = "tao_te_ching"
    KNOWN_CRITERIA = []
    SYNTHESIS_CONFIGS = {}

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self._ensure_cached()

    def _ensure_cached(self) -> None:
        """Fetch and cache chapters if not already cached."""
        if not TAO_CHAPTERS_PATH.exists():
            logger.info(f"Fetching Tao Te Ching chapters to {TAO_CHAPTERS_PATH}")
            try:
                TAO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                chapters = fetch_and_parse_chapters()
                with open(TAO_CHAPTERS_PATH, "wb") as f:
                    pickle.dump(chapters, f)
                logger.info(f"Cached {len(chapters)} chapters")
            except Exception as e:
                raise RuntimeError(f"Failed to fetch Tao Te Ching: {e}") from e
        else:
            logger.debug(f"Tao Te Ching already cached at {TAO_CHAPTERS_PATH}")

    def load_documents(self) -> list[Any]:
        """Load Tao Te Ching chapters from cache.

        Returns:
            List of chapter texts
        """
        logger.info(f"Loading Tao Te Ching from {TAO_CHAPTERS_PATH}")

        try:
            with open(TAO_CHAPTERS_PATH, "rb") as f:
                chapters = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load Tao Te Ching: {e}") from e

        documents = [ch["text"] for ch in chapters]

        max_docs = self.config.get("max_docs")
        if max_docs is not None:
            documents = documents[:max_docs]
            logger.info(f"Limited to {max_docs} documents")

        logger.info(f"Loaded {len(documents)} chapters")
        return self._deduplicate(documents)

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        return document if isinstance(document, str) else ""
