"""Onion News headlines document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import ONION_NEWS_CRITERIA

logger = logging.getLogger(__name__)


class OnionNewsDocSet(BaseDocSet):
    """Onion News satirical headlines document_set."""

    DATASET_PATH = "Biddls/Onion_News"
    DESCRIPTION = "Satirical headlines from The Onion"

    # Criteria that can be extracted deterministically (no LLM needed)
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria
    CRITERION_METADATA = ONION_NEWS_CRITERIA

    def load_documents(self) -> list[Any]:
        """Load Onion News headlines as documents from Hugging Face."""
        logger.info(f"Loading Onion News from Hugging Face: {self.DATASET_PATH}")

        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        documents = []
        for i, example in enumerate(dataset):
            # Extract headline from text field, taking content before #~#
            text = example.get("text", "")
            if text:
                headline = text.split("#~#")[0].strip()
                if headline:
                    documents.append(headline)

            if (
                not use_streaming
                and max_docs is not None
                and len(documents) >= max_docs
            ):
                break

        logger.debug(f"Loaded {len(documents)} documents from Onion News")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        return document if isinstance(document, str) else ""
