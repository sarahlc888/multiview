"""Emily Dickinson poetry document set loader.

Scrapes Emily Dickinson poems from Project Gutenberg.
Based on reference implementation from project/src/s3/benchmark/utils/factory_utils/custom_tasks/dickinson.py
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)

# Dickinson dataset configuration
DICKINSON_URL = "https://www.gutenberg.org/files/12242/12242-h/12242-h.htm"
DICKINSON_CACHE_DIR = Path.home() / ".cache" / "multiview" / "dickinson"
DICKINSON_POEMS_PATH = DICKINSON_CACHE_DIR / "poems.pkl"


def scrape_dickinson_poems(
    url="https://www.gutenberg.org/files/12242/12242-h/12242-h.htm",
) -> list[dict]:
    """
    Scrape Emily Dickinson poems from Project Gutenberg.

    Returns:
        List of dictionaries with poem data and metadata
    """
    logger.info("Fetching HTML from Project Gutenberg...")
    response = requests.get(url)
    response.encoding = "utf-8"

    logger.info("Parsing HTML...")
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()

    logger.info("Extracting poems section...")
    # Find start of poems
    start_idx = text.find("I. LIFE.")
    if start_idx == -1:
        raise ValueError("Could not find start of poems (looking for 'I. LIFE.')")

    # Find end of poems
    end_markers = [
        "Index of First Lines",
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]
    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker, start_idx)
        if idx != -1:
            end_idx = min(end_idx, idx)

    poems_section = text[start_idx:end_idx]

    logger.info("Parsing individual poems...")
    # Parse poems with metadata
    lines = poems_section.split("\n")
    poems_data = []
    current_poem = []
    current_section = None
    current_series = "First Series"  # Track which series we're in
    in_poem = False
    skip_until_close_bracket = False
    poem_number_in_section = 0

    # Patterns
    section_header_pattern = re.compile(r"^([IVX]+)\.\s+([A-Z][A-Z\s]+)\.$")
    poem_number_pattern = re.compile(r"^([IVXLCDM]+)\.$")
    series_pattern = re.compile(r"Second Series|Third Series")

    for line in lines:
        stripped = line.strip()

        # Check if we've moved to a new series
        series_match = series_pattern.search(stripped)
        if series_match:
            current_series = series_match.group(0)
            continue

        # Handle multi-line editorial notes
        if skip_until_close_bracket:
            if "]" in stripped:
                skip_until_close_bracket = False
            continue

        if stripped.startswith("["):
            if not stripped.endswith("]"):
                skip_until_close_bracket = True
            continue

        # Check if this is a section header (e.g., "I. LIFE.", "II. LOVE.")
        section_match = section_header_pattern.match(stripped)
        if section_match:
            if current_poem:
                poem_text = "\n".join(current_poem).strip()
                if poem_text:
                    poems_data.append(
                        {
                            "text": poem_text,
                            "section": current_section,
                            "section_number": poem_number_in_section,
                            "series": current_series,
                        }
                    )
                current_poem = []
            current_section = section_match.group(2).strip()
            poem_number_in_section = 0
            in_poem = False
            continue

        # Check if this is a poem number
        poem_match = poem_number_pattern.match(stripped)
        if poem_match and len(stripped) <= 6:
            if current_poem:
                poem_text = "\n".join(current_poem).strip()
                if poem_text:
                    poems_data.append(
                        {
                            "text": poem_text,
                            "section": current_section,
                            "section_number": poem_number_in_section,
                            "series": current_series,
                        }
                    )
                current_poem = []
            in_poem = True
            poem_number_in_section += 1
            continue

        # Collect poem lines
        if in_poem:
            current_poem.append(line)

    # Don't forget the last poem
    if current_poem:
        poem_text = "\n".join(current_poem).strip()
        if poem_text:
            poems_data.append(
                {
                    "text": poem_text,
                    "section": current_section,
                    "section_number": poem_number_in_section,
                    "series": current_series,
                }
            )

    logger.info(f"Successfully extracted {len(poems_data)} poems!")
    return poems_data


class DickinsonDocSet(BaseDocSet):
    """Emily Dickinson poetry document set."""

    # Metadata
    DATASET_PATH = str(DICKINSON_POEMS_PATH)
    DESCRIPTION = "Emily Dickinson poetry from Project Gutenberg"

    # Criteria that can be extracted deterministically (no LLM needed)
    # word_count is automatically included by base class
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria (descriptions and schema hints)
    CRITERION_METADATA = {}
    # Synthesis prompts for LM-based document generation
    SYNTHESIS_CONFIGS = {}

    def __init__(self, config: dict | None = None):
        """Initialize DickinsonDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)

        # Ensure poems are scraped and cached
        self._ensure_poems_cached()

    def _ensure_poems_cached(self) -> None:
        """Scrape and cache poems if not already cached."""
        if not DICKINSON_POEMS_PATH.exists():
            logger.info(f"Scraping Dickinson poems to {DICKINSON_POEMS_PATH}")
            try:
                DICKINSON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

                # Scrape poems from Project Gutenberg
                poems_data = scrape_dickinson_poems(url=DICKINSON_URL)

                # Save to cache
                with open(DICKINSON_POEMS_PATH, "wb") as f:
                    pickle.dump(poems_data, f)

                logger.info(f"Successfully scraped and cached {len(poems_data)} poems")
            except Exception as e:
                raise RuntimeError(f"Failed to scrape Dickinson poems: {e}") from e
        else:
            logger.debug(f"Dickinson poems already cached at {DICKINSON_POEMS_PATH}")

    def load_documents(self) -> list[Any]:
        """Load Emily Dickinson poems from cache.

        Returns:
            List of poem texts
        """
        logger.info(f"Loading Emily Dickinson poems from {DICKINSON_POEMS_PATH}")

        # Load from cache
        try:
            with open(DICKINSON_POEMS_PATH, "rb") as f:
                poems_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load Dickinson poems: {e}") from e

        # Extract just the text for documents
        documents = [poem["text"] for poem in poems_data]

        # Apply max_docs if specified
        max_docs = self.config.get("max_docs")
        if max_docs is not None:
            documents = documents[:max_docs]
            logger.info(f"Limited to {max_docs} documents")

        logger.info(f"Loaded {len(documents)} poems")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        return document if isinstance(document, str) else ""
