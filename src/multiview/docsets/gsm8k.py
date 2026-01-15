"""GSM8K document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import GSM8K_CRITERIA
from multiview.docsets.synthesis_configs import GSM8K_SYNTHESIS_CONFIGS

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

        By default, formats each example as:
        "Question: {question}\nAnswer: {answer}"

        If config["question_only"] is True, returns dict documents with:
        {"text": "Question: {question}", "question": "...", "answer": "..."}
        where only the question is used for annotations/evaluation.
        """
        logger.info(f"Loading GSM8K from Hugging Face: {self.DATASET_PATH}")

        # Determine if we should use streaming mode
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        question_only = self.config.get("question_only", False)
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(
                self.DATASET_PATH, "main", split=split, streaming=True
            )
            dataset = dataset.shuffle(seed=42).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, "main", split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        documents = []
        for i, example in enumerate(dataset):
            if question_only:
                # Dict format with question-only text
                doc = {
                    "text": f"Question: {example['question']}",
                    "question": example["question"],
                    "answer": example["answer"],
                }
            else:
                # String format with question + answer (default)
                doc = f"Question: {example['question']}\nAnswer: {example['answer']}"

            documents.append(doc)

            if not use_streaming and max_docs is not None and i + 1 >= max_docs:
                break

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Supports both string documents (default) and dict documents (question_only=True).
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return document if isinstance(document, str) else ""
