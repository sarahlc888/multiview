"""Analogies document_set loader."""

from __future__ import annotations

import logging
import re
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class AnalogiesDocSet(BaseDocSet):
    """Word analogy pairs document_set.

    Loads word analogy pairs from HuggingFace dataset "relbert/analogy_questions".
    Each analogy question generates two documents: the stem pair and the answer pair.

    Config parameters:
        max_docs (int, optional): Maximum number of document pairs to load (applies to pairs, not questions)
        split (str): Dataset split to use (default: "test")
        dataset_config (str): Which analogy set to use - "bats", "sat", or "google" (default: "bats")

    Implementation notes:
        - Each question produces 2 word pairs: stem and answer
        - max_docs applies to total pairs, so max_docs=100 loads ~50 questions
        - Streaming disabled due to datasets 2.x split compatibility issues
        - Documents are dicts with 'text' and 'analogy_type' fields
        - analogy_type is a known criterion extracted from prefix field (e.g., "antonyms - gradable")
        - Category names are extracted from file paths using regex pattern
        - Robust error handling for malformed records
        - Provides precomputed annotations for analogy_type criterion
    """

    DATASET_PATH = "relbert/analogy_questions"
    DESCRIPTION = "Word analogy pairs (stem and answer)"
    KNOWN_CRITERIA = ["analogy_type"]  # prefix field (e.g., "country-capital")

    def __init__(self, config: dict | None = None):
        """Initialize AnalogiesDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}

    def load_documents(self) -> list[Any]:
        """Load analogy pairs from HuggingFace.

        For each analogy, extracts two word pairs:
        1. The stem pair: "word1 : word2"
        2. The answer pair: "word3 : word4"

        Documents are dicts with 'text' and 'analogy_type' keys to support
        analogy_type as a known criterion.

        Returns:
            List of document dicts: {"text": "word : word", "analogy_type": "country-capital"}
        """
        logger.info(f"Loading Analogies from HuggingFace: {self.DATASET_PATH}")

        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "test")  # Default to test
        dataset_config = self.config.get("dataset_config", "bats")

        # max_docs applies to PAIRS (each question generates 2 pairs)
        max_questions = (max_docs // 2) if max_docs else None
        # Disable streaming for analogies due to split compatibility issues in datasets 2.x
        use_streaming = False

        if use_streaming:
            logger.debug(f"Using streaming mode (max_questions={max_questions} < 100)")
            dataset = load_dataset(
                self.DATASET_PATH, dataset_config, split=split, streaming=True
            )
            dataset = dataset.shuffle(seed=42)
        else:
            # Load all splits first, then select the desired split
            # This avoids split parsing issues in datasets 2.x
            dataset_dict = load_dataset(self.DATASET_PATH, dataset_config)
            dataset = dataset_dict[split]
            if max_questions is not None:
                dataset = dataset.shuffle(seed=42)

        # Extract word pairs with analogy type
        documents = []
        for i, example in enumerate(dataset):
            try:
                # Get analogy type (prefix) and extract category name from path
                # Example: "./cache/BATS_3.0/4_Lexicographic_semantics/L09 [antonyms - gradable].txt"
                # Extract: "antonyms - gradable"
                prefix = example.get("prefix", "")
                match = re.search(r"\[([^\]]+)\]", prefix)
                analogy_type = match.group(1) if match else prefix

                # Extract stem pair
                stem = example.get("stem", [])
                if len(stem) >= 2:
                    stem_text = " : ".join(stem)
                    documents.append({"text": stem_text, "analogy_type": analogy_type})

                # Extract answer pair
                choices = example.get("choice", [])
                answer_idx = example.get("answer")
                if answer_idx is not None and 0 <= int(answer_idx) < len(choices):
                    answer_choice = choices[int(answer_idx)]
                    if len(answer_choice) >= 2:
                        answer_text = " : ".join(answer_choice)
                        documents.append(
                            {"text": answer_text, "analogy_type": analogy_type}
                        )

            except (KeyError, IndexError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed analogy at index {i}: {e}")
                continue

            # Check if we've loaded enough questions
            if max_questions is not None and i + 1 >= max_questions:
                break

        # Final max_docs enforcement
        if max_docs is not None and len(documents) > max_docs:
            documents = documents[:max_docs]

        logger.debug(f"Loaded {len(documents)} word pair documents from Analogies")

        # Build precomputed annotations for analogy_type criterion
        self._build_precomputed_annotations(documents)

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        if isinstance(document, dict):
            return document.get("text", "")
        return document if isinstance(document, str) else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - analogy_type: the prefix field (e.g., "country-capital")
        """
        if criterion == "analogy_type":
            if isinstance(document, dict):
                return document.get("analogy_type")
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations(self, documents: list[dict]) -> None:
        """Build precomputed annotations from loaded documents.

        Creates a mapping: {document_text: {"criterion_value": analogy_type}}

        Args:
            documents: List of document dicts with 'text' and 'analogy_type' fields
        """
        annotations = {}

        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text")
                analogy_type = doc.get("analogy_type")

                if text and analogy_type:
                    annotations[text] = {"criterion_value": analogy_type}

        self.PRECOMPUTED_ANNOTATIONS["analogy_type"] = annotations

        logger.info(
            f"Built precomputed annotations for analogy_type: {len(annotations)} documents"
        )
