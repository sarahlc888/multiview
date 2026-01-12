"""HackerNews posts document_set loader.

Loads posts from HuggingFace (julien040/hacker-news-posts) and formats as "title\\nurl".
Posts filtered by score threshold (default: 10). Uses streaming mode for max_docs < 100.

Note: Score values not preserved after formatting (documents are strings, not dicts).
"""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import HACKERNEWS_CRITERIA

logger = logging.getLogger(__name__)


class HackerNewsDocSet(BaseDocSet):
    """HackerNews posts document_set.

    Loads posts from HuggingFace dataset "julien040/hacker-news-posts" and formats
    them as "title\\nurl". Posts are filtered by score threshold.

    Config parameters:
        max_docs (int, optional): Maximum number of documents to load
        split (str): Dataset split to use (default: "train")
        min_score (int): Minimum score threshold for posts (default: 10)

    Implementation notes:
        - Uses streaming mode when max_docs < 100 for efficiency
        - Filters during iteration to find enough posts above score threshold
        - Skips posts missing title or url fields
    """

    # Metadata
    DATASET_PATH = "julien040/hacker-news-posts"
    DESCRIPTION = "HackerNews posts with title and URL"

    # Known criteria (only deterministic ones)
    KNOWN_CRITERIA = []  # word_count auto-included by base class

    # Metadata for LM-based criteria (descriptions, schema hints, etc.)
    CRITERION_METADATA = HACKERNEWS_CRITERIA

    def load_documents(self) -> list[Any]:
        """Load HackerNews posts from HuggingFace.

        Loads posts and formats as "title\\nurl", filtering by score threshold.

        Returns:
            List of formatted documents (title+url strings)
        """
        logger.info(f"Loading HackerNews from HuggingFace: {self.DATASET_PATH}")

        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        min_score = self.config.get("min_score", 10)
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42)
        else:
            dataset = load_dataset(self.DATASET_PATH, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        # Format and filter documents
        documents = []
        for _, example in enumerate(dataset):
            # Check score threshold
            score = example.get("score", 0)
            if score > min_score:
                title = example.get("title", "")
                url = example.get("url", "")
                if title and url:
                    formatted_doc = f"{title}\n{url}"
                    documents.append(formatted_doc)

            # Respect max_docs
            if max_docs is not None and len(documents) >= max_docs:
                break

        logger.debug(
            f"Loaded {len(documents)} documents from HackerNews "
            f"(filtered by score > {min_score})"
        )
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A single document (formatted string)

        Returns:
            The text content of the document
        """
        return document if isinstance(document, str) else ""
