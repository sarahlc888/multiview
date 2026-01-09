"""ROCStories document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.benchmark.document_sets.base import BaseDocSet

logger = logging.getLogger(__name__)


class RocStoriesDocSet(BaseDocSet):
    """ROCStories document_set."""

    DATASET_PATH = "mintujupally/ROCStories"
    DESCRIPTION = "ROCStories short stories"

    # Criteria that can be extracted deterministically (no LLM needed)
    # word_count is automatically included by base class
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[Any]:
        """Load ROCStories as documents from Hugging Face.

        Loads the ROCStories dataset and formats each story. Stories can come
        from various field formats (story, text, or sentence1-sentence5).

        Returns:
            List of formatted story strings
        """
        logger.info(f"Loading ROCStories from Hugging Face: {self.DATASET_PATH}")

        # Determine if we should use streaming mode
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            # Shuffle and take the first max_docs
            dataset = dataset.shuffle(seed=42).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, split=split)
            if max_docs is not None:
                # Shuffle and slice for non-streaming mode
                dataset = dataset.shuffle(seed=42)

        # Format documents
        documents = []
        for i, example in enumerate(dataset):
            story = self._build_story(example)
            if story:
                documents.append(story)

            # Respect max_docs in non-streaming mode
            if not use_streaming and max_docs is not None and len(documents) >= max_docs:
                break

        logger.debug(f"Loaded {len(documents)} documents from ROCStories")
        return documents

    def _build_story(self, item: dict) -> str:
        """Build story text from various possible field formats.

        Args:
            item: A single ROCStories example

        Returns:
            The story text
        """
        # Check for direct story or text field
        if "story" in item and item["story"]:
            return item["story"]
        if "text" in item and item["text"]:
            return item["text"]

        # Build from sentence fields (sentence1-sentence5)
        sentence_fields = [f"sentence{i}" for i in range(1, 6)]
        if all(field in item for field in sentence_fields):
            sentences = [item[field] for field in sentence_fields if item.get(field)]
            return " ".join(sentences)

        return ""

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A single document (story string)

        Returns:
            The text content of the document
        """
        # Documents are already formatted as strings
        return document if isinstance(document, str) else ""
