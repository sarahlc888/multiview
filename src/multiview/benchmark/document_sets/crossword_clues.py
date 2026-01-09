"""Crossword clues document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.benchmark.document_sets.base import BaseDocSet

logger = logging.getLogger(__name__)


class CrosswordCluesDocSet(BaseDocSet):
    """Crossword clues document_set."""

    DATASET_PATH = "albertxu/CrosswordQA"
    DESCRIPTION = "Crossword clues and answers"

    # Criteria that can be extracted deterministically (no LLM needed)
    # word_count is automatically included by base class
    KNOWN_CRITERIA = []

    def load_documents(self) -> list[Any]:
        """Load crossword clues as documents from Hugging Face.

        Loads the CrosswordQA dataset and formats each example as:
        "Clue: {clue}\nAnswer: {answer}"

        Returns:
            List of formatted documents (clue-answer pairs)
        """
        logger.info(f"Loading Crossword clues from Hugging Face: {self.DATASET_PATH}")

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
        for _i, example in enumerate(dataset):
            # Format as "Clue: ...\nAnswer: ..."
            clue = example.get("clue", "")
            answer = example.get("answer", "")
            if clue and answer:
                formatted_doc = f"Clue: {clue}\nAnswer: {answer}"
                documents.append(formatted_doc)

            # Respect max_docs in non-streaming mode
            if (
                not use_streaming
                and max_docs is not None
                and len(documents) >= max_docs
            ):
                break

        logger.debug(f"Loaded {len(documents)} documents from Crossword clues")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A single document (formatted string)

        Returns:
            The text content of the document
        """
        # Documents are already formatted as strings
        return document if isinstance(document, str) else ""
