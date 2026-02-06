"""Goodreads quotes document set loader.

Loads quotes from Goodreads with author, tags, and likes metadata.
Uses streaming mode due to large dataset size.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class GoodreadsQuotesDocSet(BaseDocSet):
    """Goodreads quotes dataset.

    Analyzes quotes and their potential for interesting pairwise conversations
    and connections. Documents are dicts with 'text', 'author', 'tags', and 'likes' fields.
    Metadata is preserved for filtering and analysis.

    Config parameters:
        split (str): Dataset split to use (default: "train")
        max_docs (int, optional): Maximum documents to load
        min_likes (int, optional): Minimum number of likes for filtering
        authors (list[str], optional): Filter to specific authors (case-insensitive partial match)
        seed (int): Random seed for shuffling (default: 42)
        include_curated (bool): Include manually curated quotes from data/curated_quotes.json (default: True)

    Usage:
        tasks:
          - document_set: goodreads_quotes
            criterion: positive_sum
            config:
              split: train
              max_docs: 1000
              min_likes: 10
              authors: ["Tolstoy", "Nabokov"]  # Optional: filter by authors
    """

    # Metadata
    DATASET_PATH = "EhsanShahbazi/goodreads-quotes"
    DESCRIPTION = "Quotes from Goodreads with author, tags, and likes metadata"

    # Criteria that can be extracted deterministically (no LLM needed)
    # word_count is automatically included by base class
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria (descriptions and schema hints)
    DATASET_NAME = "goodreads_quotes"

    # Synthesis prompts for LM-based document generation
    SYNTHESIS_CONFIGS = {}

    def load_documents(self) -> list[Any]:
        """Load quotes from Hugging Face using streaming mode.

        Returns:
            List of quote strings formatted as "{text}\n- {author}"
        """
        logger.info(f"Loading Goodreads quotes from Hugging Face: {self.DATASET_PATH}")

        # Config parameters
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        min_likes = self.config.get("min_likes")
        authors_filter = self.config.get("authors")  # List of author names to filter
        seed = self.config.get("seed", 42)
        include_curated = self.config.get("include_curated", True)

        # IMPORTANT: Always use streaming=True for this large dataset
        logger.info("Using streaming mode for large Goodreads quotes dataset")
        if authors_filter:
            logger.info(f"Filtering for authors: {', '.join(authors_filter)}")

        dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)

        # Only shuffle if not filtering by authors (shuffle can miss rare authors)
        if not authors_filter:
            dataset = dataset.shuffle(seed=seed, buffer_size=10000)
        else:
            logger.info("Skipping shuffle to ensure all requested authors are found")

        # When filtering by multiple authors, collect from each author separately
        if authors_filter and len(authors_filter) > 1 and max_docs:
            # Divide max_docs equally among authors
            quotes_per_author = max_docs // len(authors_filter)
            logger.info(f"Collecting ~{quotes_per_author} quotes per author")

            documents = []
            author_counts = dict.fromkeys(authors_filter, 0)
            processed_count = 0

            for example in dataset:
                processed_count += 1

                # Extract fields
                quote_text = example.get("quote", "").strip()
                author = example.get("author", "Unknown").strip()
                tags = example.get("tags", "").strip()
                likes = example.get("likes", 0)

                # Skip empty quotes
                if not quote_text:
                    continue

                # Check which author this matches
                matching_author = None
                for filter_author in authors_filter:
                    if filter_author.lower() in author.lower():
                        matching_author = filter_author
                        break

                if not matching_author:
                    continue

                # Check if we've collected enough from this author
                if author_counts[matching_author] >= quotes_per_author:
                    continue

                # Apply minimum likes filter
                if min_likes is not None and likes < min_likes:
                    continue

                # Create document as dict with metadata
                doc = {
                    "text": quote_text,
                    "author": author,
                    "tags": tags,
                    "likes": likes,
                }
                documents.append(doc)
                author_counts[matching_author] += 1

                # Check if we have enough from all authors
                if all(count >= quotes_per_author for count in author_counts.values()):
                    logger.info(
                        f"Collected {quotes_per_author} quotes from each author"
                    )
                    break

                # Log progress
                if processed_count % 25000 == 0:
                    logger.info(
                        f"Processed {processed_count} examples, collected: {dict(author_counts)}"
                    )

                # Safety limit
                if processed_count >= 250000:
                    logger.warning(
                        f"Stopped after {processed_count} examples. Collected: {dict(author_counts)}"
                    )
                    break

        else:
            # Original logic for single author or no author filter
            documents = []
            processed_count = 0

            for example in dataset:
                processed_count += 1

                # Extract fields from the dataset
                quote_text = example.get("quote", "").strip()
                author = example.get("author", "Unknown").strip()
                tags = example.get("tags", "").strip()
                likes = example.get("likes", 0)

                # Skip empty quotes
                if not quote_text:
                    continue

                # Apply author filter if specified (case-insensitive partial match)
                if authors_filter:
                    author_match = any(
                        filter_author.lower() in author.lower()
                        for filter_author in authors_filter
                    )
                    if not author_match:
                        continue

                # Apply minimum likes filter if specified
                if min_likes is not None and likes < min_likes:
                    continue

                # Create document as dict with metadata
                doc = {
                    "text": quote_text,
                    "author": author,
                    "tags": tags,
                    "likes": likes,
                }
                documents.append(doc)

                # Check max_docs limit
                if max_docs is not None and len(documents) >= max_docs:
                    break

                # Log progress for large datasets (when filtering by author)
                if authors_filter and processed_count % 10000 == 0:
                    logger.info(
                        f"Processed {processed_count} examples, collected {len(documents)} quotes"
                    )

                # For safety: stop after processing too many examples when filtering
                if authors_filter and processed_count >= 200000:
                    logger.warning(
                        f"Stopped after processing {processed_count} examples to avoid timeout"
                    )
                    break

        logger.info(f"Loaded {len(documents)} quotes from Goodreads")

        # Load and merge curated quotes if enabled
        if include_curated:
            curated_docs = self._load_curated_quotes(authors_filter, min_likes)
            if curated_docs:
                logger.info(f"Adding {len(curated_docs)} curated quotes")
                documents.extend(curated_docs)

        return self._deduplicate(documents)

    def _load_curated_quotes(self, authors_filter=None, min_likes=None) -> list[Any]:
        """Load manually curated quotes from data/curated_quotes.json.

        Args:
            authors_filter: Optional list of author names to filter
            min_likes: Optional minimum likes threshold

        Returns:
            List of curated quote strings formatted as "{text}\n- {author}"
        """
        # Find the data directory relative to the project root
        # __file__ is in src/multiview/docsets/, so go up 4 levels to project root
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        curated_file = project_root / "data" / "curated_quotes.json"

        if not curated_file.exists():
            logger.debug(f"Curated quotes file not found at {curated_file}")
            return []

        try:
            with open(curated_file, encoding="utf-8") as f:
                curated_quotes = json.load(f)

            documents = []
            for quote_data in curated_quotes:
                author = quote_data.get("author", "Unknown")

                # Apply author filter if specified
                if authors_filter:
                    author_match = any(
                        filter_author.lower() in author.lower()
                        for filter_author in authors_filter
                    )
                    if not author_match:
                        continue

                # Apply minimum likes filter if specified
                likes = quote_data.get("likes", 0)
                if min_likes is not None and likes < min_likes:
                    continue

                # Create document as dict with metadata
                text = quote_data.get("quote", "")
                tags = quote_data.get("tags", "")
                doc = {
                    "text": text,
                    "author": author,
                    "tags": tags,
                    "likes": likes,
                }
                documents.append(doc)

            logger.info(f"Loaded {len(documents)} curated quotes from {curated_file}")
            return documents

        except Exception as e:
            logger.warning(f"Failed to load curated quotes: {e}")
            return []

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: Document dict with 'text' and 'author' fields

        Returns:
            Quote text formatted as: {text}\n- {author}
        """
        if isinstance(document, dict):
            text = document.get("text", "")
            author = document.get("author", "Unknown")
            return f"{text}\n- {author}"
        return str(document)

    def get_known_criterion_value(self, document: Any, criterion: str) -> Any:
        """Get the known criterion value for a document.

        Args:
            document: Document string
            criterion: Criterion name

        Returns:
            None (documents are strings, no metadata available)
        """
        # Documents are now strings, not dicts, so no metadata is available
        return None
