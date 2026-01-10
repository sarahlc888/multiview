"""GSM8K document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.benchmark.document_sets.base import BaseDocSet
from multiview.benchmark.document_sets.criteria_metadata import GSM8K_CRITERIA
from multiview.benchmark.document_sets.synthesis_configs import (
    GSM8K_SYNTHESIS_CONFIGS,
)

logger = logging.getLogger(__name__)


class GSM8KDocSet(BaseDocSet):
    """GSM8K math word problems document_set."""

    # Metadata
    DATASET_PATH = "openai/gsm8k"
    DESCRIPTION = "GSM8K math word problems"

    # Criteria that can be extracted deterministically (no LLM needed)
    # word_count is automatically included by base class
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria (descriptions and schema hints)
    CRITERION_METADATA = GSM8K_CRITERIA
    # Synthesis prompts for LM-based document generation
    SYNTHESIS_CONFIGS = GSM8K_SYNTHESIS_CONFIGS

    def load_documents(self) -> list[Any]:
        """Load GSM8K problems as documents from Hugging Face.

        Loads the GSM8K dataset and formats each example as:
        "Question: {question}\nAnswer: {answer}"

        Returns:
            List of formatted documents (problems)
        """
        logger.info(f"Loading GSM8K from Hugging Face: {self.DATASET_PATH}")

        # Determine if we should use streaming mode
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(
                self.DATASET_PATH, "main", split=split, streaming=True
            )
            # Shuffle and take the first max_docs
            dataset = dataset.shuffle(seed=42).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, "main", split=split)
            if max_docs is not None:
                # Shuffle and slice for non-streaming mode
                dataset = dataset.shuffle(seed=42)

        # Format documents
        documents = []
        for i, example in enumerate(dataset):
            # Format as "Question: ...\nAnswer: ..."
            formatted_doc = (
                f"Question: {example['question']}\nAnswer: {example['answer']}"
            )
            documents.append(formatted_doc)

            # Respect max_docs in non-streaming mode
            if not use_streaming and max_docs is not None and i + 1 >= max_docs:
                break

        logger.debug(f"Loaded {len(documents)} documents from GSM8K")
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
