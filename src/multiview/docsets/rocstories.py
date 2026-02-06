"""ROCStories document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class RocStoriesDocSet(BaseDocSet):
    """ROCStories document_set."""

    DATASET_PATH = "mintujupally/ROCStories"
    DESCRIPTION = "ROCStories short stories"
    DOCUMENT_TYPE = "Simple story made up of a few sentences"

    # Criteria that can be extracted deterministically (no LLM needed)
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria
    DATASET_NAME = "rocstories"

    def load_documents(self) -> list[Any]:
        """Load ROCStories as documents from Hugging Face."""
        logger.info(f"Loading ROCStories from Hugging Face: {self.DATASET_PATH}")

        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42, buffer_size=10000).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        documents = []
        for _i, example in enumerate(dataset):
            story = self._build_story(example)
            if story:
                documents.append(story)
            if (
                not use_streaming
                and max_docs is not None
                and len(documents) >= max_docs
            ):
                break

        logger.debug(f"Loaded {len(documents)} documents from ROCStories")
        return self._deduplicate(documents)

    def _build_story(self, item: dict) -> str:
        if "story" in item and item["story"]:
            return item["story"]
        if "text" in item and item["text"]:
            return item["text"]

        sentence_fields = [f"sentence{i}" for i in range(1, 6)]
        if all(field in item for field in sentence_fields):
            sentences = [item[field] for field in sentence_fields if item.get(field)]
            return " ".join(sentences)

        return ""

    def get_document_text(self, document: Any) -> str:
        return document if isinstance(document, str) else ""
