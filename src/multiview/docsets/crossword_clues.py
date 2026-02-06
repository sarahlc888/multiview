"""Crossword clues document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class CrosswordCluesDocSet(BaseDocSet):
    """Crossword clues document_set."""

    DATASET_PATH = "albertxu/CrosswordQA"
    DESCRIPTION = "Crossword clues and answers"

    # Criteria that can be extracted deterministically (no LLM needed)
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria
    DATASET_NAME = "crossword"

    def load_documents(self) -> list[Any]:
        """Load crossword clues as documents from Hugging Face."""
        logger.info(f"Loading Crossword clues from Hugging Face: {self.DATASET_PATH}")

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
            clue = example.get("clue", "")
            answer = example.get("answer", "")
            if clue and answer:
                documents.append(f"Clue: {clue}\nAnswer: {answer}")

            if (
                not use_streaming
                and max_docs is not None
                and len(documents) >= max_docs
            ):
                break

        logger.debug(f"Loaded {len(documents)} documents from Crossword clues")
        return self._deduplicate(documents)

    def get_document_text(self, document: Any) -> str:
        return document if isinstance(document, str) else ""
