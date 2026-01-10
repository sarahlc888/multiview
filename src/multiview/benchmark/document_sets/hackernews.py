"""HackerNews posts document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.benchmark.document_sets.base import BaseDocSet

logger = logging.getLogger(__name__)


class HackerNewsDocSet(BaseDocSet):
    """HackerNews posts document_set."""

    # Metadata
    DATASET_PATH = "julien040/hacker-news-posts"
    DESCRIPTION = "HackerNews posts with title and URL"

    # Known criteria (only deterministic ones)
    KNOWN_CRITERIA = []  # word_count auto-included by base class

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
        for i, example in enumerate(dataset):
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
