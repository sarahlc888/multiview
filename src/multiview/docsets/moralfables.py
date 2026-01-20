"""MoralFables document set loader."""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class MoralFablesDocSet(BaseDocSet):
    """Moralistic fables with structured moral metadata.

    Loads moralistic fables from HuggingFace dataset "klusai/ds-tf1-en-100k".
    Each fable includes structured metadata: setting, challenge, outcome, and teaching.

    Config parameters:
        max_docs (int, optional): Maximum number of documents to load
        split (str): Dataset split to use (default: "train")
        dataset_path (str): HuggingFace dataset path (default: "klusai/ds-tf1-en-100k")

    Implementation notes:
        - Uses streaming mode for efficiency when max_docs is specified
        - Documents are simple strings (fable text)
        - Known criteria: setting, challenge, outcome, teaching
        - Provides precomputed annotations for all 4 criteria
        - Robust error handling for malformed prompt fields
    """

    DATASET_PATH = "klusai/ds-tf1-en-100k"
    DESCRIPTION = "Moralistic fables with structured moral metadata"
    KNOWN_CRITERIA = ["setting", "challenge", "outcome", "teaching"]

    def __init__(self, config: dict | None = None):
        """Initialize MoralFablesDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}

    def load_documents(self) -> list[Any]:
        """Load moral fables from HuggingFace.

        For each fable, extracts:
        1. The fable text (returned as document)
        2. Setting, challenge, outcome, teaching (stored in PRECOMPUTED_ANNOTATIONS)

        Documents are simple strings (the fable text).
        Metadata is stored separately in PRECOMPUTED_ANNOTATIONS.

        Returns:
            List of fable text strings
        """
        logger.info(f"Loading MoralFables from HuggingFace: {self.DATASET_PATH}")

        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        dataset_path = self.config.get("dataset_path", self.DATASET_PATH)

        # Use streaming mode for efficiency when max_docs is specified
        use_streaming = max_docs is not None and max_docs < 1000

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs})")
            dataset = load_dataset(dataset_path, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42, buffer_size=10000)
        else:
            logger.debug("Loading full dataset")
            dataset = load_dataset(dataset_path, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        # Extract fables with metadata
        documents = []
        metadata_list = []  # Store metadata separately for annotation building

        for i, example in enumerate(dataset):
            try:
                # Extract fable text
                fable_text = example.get("fable", "")
                if not fable_text:
                    logger.warning(f"Skipping example {i}: missing fable text")
                    continue

                # Parse prompt metadata
                prompt = example.get("prompt", "")
                prompt_lines = [x.strip() for x in prompt.split("\n")]

                # Extract metadata fields
                setting = self._extract_field(prompt_lines, "Setting:")
                challenge = self._extract_field(prompt_lines, "Challenge:")
                outcome = self._extract_field(prompt_lines, "Outcome:")
                teaching = self._extract_field(prompt_lines, "Teaching:")

                # Store text as document (string, not dict)
                documents.append(fable_text)

                # Store metadata separately for precomputed annotations
                metadata_list.append(
                    {
                        "text": fable_text,
                        "setting": setting,
                        "challenge": challenge,
                        "outcome": outcome,
                        "teaching": teaching,
                    }
                )

            except (KeyError, IndexError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed fable at index {i}: {e}")
                continue

            # Check if we've loaded enough documents
            if max_docs is not None and len(documents) >= max_docs:
                break

        logger.debug(f"Loaded {len(documents)} moral fable documents")

        # Deduplicate before building precomputed annotations
        documents = self._deduplicate(documents)

        # Build precomputed annotations for all criteria
        self._build_precomputed_annotations(metadata_list)

        return documents

    def _extract_field(self, prompt_lines: list[str], field_prefix: str) -> str:
        """Extract a field from prompt metadata lines.

        Args:
            prompt_lines: List of lines from the prompt
            field_prefix: Field prefix to search for (e.g., "Setting:")

        Returns:
            Extracted field value or empty string if not found
        """
        for line in prompt_lines:
            if line.startswith(f"- {field_prefix}"):
                return line.split(f"- {field_prefix}", 1)[1].strip()
        return ""

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Documents are simple strings (fable text).
        """
        return document if isinstance(document, str) else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - setting: The fable's setting (from PRECOMPUTED_ANNOTATIONS)
        - challenge: The moral challenge faced (from PRECOMPUTED_ANNOTATIONS)
        - outcome: The story outcome (from PRECOMPUTED_ANNOTATIONS)
        - teaching: The moral lesson/teaching (from PRECOMPUTED_ANNOTATIONS)

        Note: Documents are strings, so criterion values come from PRECOMPUTED_ANNOTATIONS.
        Use get_precomputed_annotation() to access these.
        """
        if criterion in self.KNOWN_CRITERIA:
            # Criterion values are stored in PRECOMPUTED_ANNOTATIONS
            # The caller should use get_precomputed_annotation() instead
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations(self, documents: list[dict]) -> None:
        """Build precomputed annotations from loaded documents.

        Creates mappings for all known criteria:
        {document_text: {"prelabel": prelabel_value}}

        Args:
            documents: List of document dicts with 'text' and criterion fields
        """
        for criterion in self.KNOWN_CRITERIA:
            annotations = {}

            for doc in documents:
                if isinstance(doc, dict):
                    text = doc.get("text")
                    prelabel_value = doc.get(criterion)

                    if text and prelabel_value:
                        annotations[text] = {"prelabel": prelabel_value}

            self.PRECOMPUTED_ANNOTATIONS[criterion] = annotations

            logger.info(
                f"Built precomputed annotations for {criterion}: {len(annotations)} documents"
            )
