"""MMLU (Massive Multitask Language Understanding) dataset loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import MMLU_CRITERIA

logger = logging.getLogger(__name__)


class MMLUDocSet(BaseDocSet):
    """MMLU (Massive Multitask Language Understanding) multiple-choice questions.

    MMLU is a benchmark covering 57 subjects across STEM, humanities, social sciences,
    and other areas. Each question is a multiple-choice question with 4 answer choices.
    """

    # Metadata
    DATASET_PATH = "cais/mmlu"
    DESCRIPTION = "MMLU multiple-choice questions across 57 subjects"
    DOCUMENT_TYPE = "Multiple-choice question in exam style format"

    # Criteria that can be extracted deterministically (no LLM needed)
    # subject is available as a field in the dataset
    KNOWN_CRITERIA = ["subject"]

    # Metadata for LM-based criteria (Bloom's taxonomy)
    CRITERION_METADATA = MMLU_CRITERIA

    def load_documents(self) -> list[Any]:
        """Load MMLU questions as documents from Hugging Face.

        Each document is formatted as:
        {
            "text": "Question: {question}\\n(A) {choice1}\\n(B) {choice2}\\n(C) {choice3}\\n(D) {choice4}",
            "question": "...",
            "choices": [...],
            "answer": "...",
            "subject": "..."
        }
        """
        logger.info(f"Loading MMLU from Hugging Face: {self.DATASET_PATH}")

        # Determine if we should use streaming mode
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "test")  # MMLU typically uses test split
        subset = self.config.get("subset", "all")  # Can specify a specific subject
        use_streaming = max_docs is not None and max_docs < 100

        # MMLU has different subsets for each subject, or "all" for all subjects
        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(
                self.DATASET_PATH, subset, split=split, streaming=True
            )
            dataset = dataset.shuffle(seed=42, buffer_size=10000).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, subset, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        documents = []
        for i, example in enumerate(dataset):
            # Format the question with choices
            question = example.get("question", "")
            choices = example.get("choices", [])
            answer = example.get("answer", 0)  # Answer index (0-3)
            subject = example.get("subject", "")

            # Build the formatted text
            choice_labels = ["A", "B", "C", "D"]
            choice_text = "\n".join(
                [f"({choice_labels[j]}) {choices[j]}" for j in range(len(choices))]
            )
            formatted_text = f"Question: {question}\n{choice_text}"

            # Create document dict
            doc = {
                "text": formatted_text,
                "question": question,
                "choices": choices,
                "answer": answer,
                "subject": subject,
            }

            documents.append(doc)

            if not use_streaming and max_docs is not None and i + 1 >= max_docs:
                break

        logger.debug(f"Loaded {len(documents)} documents from MMLU")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        if isinstance(document, dict):
            return document.get("text", "")
        return document if isinstance(document, str) else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract criterion value for known criteria.

        MMLU-specific criteria:
        - subject: The subject area of the question
        """
        # Check parent class criteria first (e.g., word_count)
        parent_value = super().get_known_criterion_value(document, criterion)
        if parent_value is not None:
            return parent_value

        # MMLU-specific criteria
        if criterion == "subject" and isinstance(document, dict):
            return document.get("subject", "")

        return None
