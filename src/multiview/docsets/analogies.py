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
        - Documents are simple strings (word pairs)
        - analogy_type is a known criterion extracted from prefix field (e.g., "antonyms - gradable")
        - Category names are extracted from file paths using regex pattern
        - Robust error handling for malformed records
        - Provides precomputed annotations for analogy_type criterion
    """

    DATASET_PATH = "relbert/analogy_questions"
    DESCRIPTION = "Word analogy pairs (stem and answer)"
    DOCUMENT_TYPE = "Word analogy pair"
    KNOWN_CRITERIA = ["analogy_type"]  # prefix field (e.g., "country-capital")
    DATASET_NAME = "analogies"

    def __init__(self, config: dict | None = None):
        """Initialize AnalogiesDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}
        # Store document metadata for visualization
        self._doc_metadata: list[dict[str, Any]] = []

    def load_documents(self) -> list[Any]:
        """Load analogy pairs from HuggingFace.

        For each analogy, extracts two word pairs:
        1. The stem pair: "word1 : word2"
        2. The answer pair: "word3 : word4"

        Documents are simple strings (the word pairs).
        Metadata is stored separately in PRECOMPUTED_ANNOTATIONS.

        Returns:
            List of word pair strings: ["word1 : word2", "word3 : word4", ...]
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
            dataset = dataset.shuffle(seed=42, buffer_size=10000)
        else:
            # Load all splits first, then select the desired split
            # This avoids split parsing issues in datasets 2.x
            dataset_dict = load_dataset(self.DATASET_PATH, dataset_config)
            dataset = dataset_dict[split]
            if max_questions is not None:
                dataset = dataset.shuffle(seed=42)

        # Extract word pairs with analogy type
        documents = []
        metadata_list = []  # Store metadata separately for annotation building

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
                    documents.append(stem_text)
                    metadata_list.append(
                        {
                            "text": stem_text,
                            "analogy_type": analogy_type,
                            "pair_type": "stem",
                        }
                    )

                # Extract answer pair
                choices = example.get("choice", [])
                answer_idx = example.get("answer")
                if answer_idx is not None and 0 <= int(answer_idx) < len(choices):
                    answer_choice = choices[int(answer_idx)]
                    if len(answer_choice) >= 2:
                        answer_text = " : ".join(answer_choice)
                        documents.append(answer_text)
                        metadata_list.append(
                            {
                                "text": answer_text,
                                "analogy_type": analogy_type,
                                "pair_type": "answer",
                            }
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
            metadata_list = metadata_list[:max_docs]

        logger.debug(f"Loaded {len(documents)} word pair documents from Analogies")

        # Deduplicate before building precomputed annotations
        documents = self._deduplicate(documents)

        # Build precomputed annotations for analogy_type criterion
        self._build_precomputed_annotations(metadata_list)

        # Store metadata for get_document_metadata
        # Create a mapping from document text to metadata
        metadata_map = {item["text"]: item for item in metadata_list if "text" in item}
        self._doc_metadata = [metadata_map.get(doc, {}) for doc in documents]

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Documents are simple strings (word pairs).
        """
        return document if isinstance(document, str) else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - analogy_type: the prefix field (from PRECOMPUTED_ANNOTATIONS)

        Note: Documents are strings, so criterion values come from PRECOMPUTED_ANNOTATIONS.
        Use get_precomputed_annotation() to access these.
        """
        if criterion == "analogy_type":
            # Criterion values are stored in PRECOMPUTED_ANNOTATIONS
            # The caller should use get_precomputed_annotation() instead
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations(self, documents: list[dict]) -> None:
        """Build precomputed annotations from loaded documents.

        Creates a mapping: {document_text: {"prelabel": analogy_type}}

        Args:
            documents: List of document dicts with 'text' and 'analogy_type' fields
        """
        annotations = {}

        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text")
                analogy_type = doc.get("analogy_type")

                if text and analogy_type:
                    annotations[text] = {"prelabel": analogy_type}

        self.PRECOMPUTED_ANNOTATIONS["analogy_type"] = annotations

        logger.info(
            f"Built precomputed annotations for analogy_type: {len(annotations)} documents"
        )

    def get_document_metadata(self, doc_idx: int) -> dict[str, Any]:
        """Get metadata for a document at the given index.

        Args:
            doc_idx: Document index

        Returns:
            Dict with pair_type ("stem" or "answer") and analogy_type
        """
        if 0 <= doc_idx < len(self._doc_metadata):
            metadata = self._doc_metadata[doc_idx]
            result = {}
            if "pair_type" in metadata:
                result["pair_type"] = metadata["pair_type"]
            if "analogy_type" in metadata:
                result["analogy_type"] = metadata["analogy_type"]
            return result
        return {}
