"""Abstract-Sim dataset loader from BIU NLP.

Loads Wikipedia sentences with abstract descriptions (good/bad) from HuggingFace.
Uses n*2 labeling scheme to guarantee within-row triplet matching.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import ABSTRACTSIM_CRITERIA

logger = logging.getLogger(__name__)


class AbstractSimDocSet(BaseDocSet):
    """Wikipedia sentences with abstract similarity descriptions.

    Dataset from: https://huggingface.co/datasets/biu-nlp/abstract-sim
    Paper: "Description-Based Text Similarity" (https://arxiv.org/pdf/2305.12517)

    Each row contains:
    - sentence: Original Wikipedia sentence
    - good: List of valid abstract descriptions
    - bad: List of invalid abstract descriptions

    Uses n*2 labeling scheme for prelabeled triplets:
    - sentence + good descriptions: row_X_class_0
    - bad descriptions: row_X_class_1

    This ensures triplets only use documents from the same original row.

    Config parameters:
        max_docs (int, optional): Maximum total documents to create
        split (str): Dataset split - "train", "validation", or "test" (default: "train")

    Usage:
        tasks:
          - document_set: abstractsim
            criterion: abstract_similarity
            triplet_style: prelabeled
            split: validation
            max_docs: 1000
    """

    DATASET_PATH = "biu-nlp/abstract-sim"
    DESCRIPTION = "Wikipedia sentences with abstract descriptions (good/bad)"
    KNOWN_CRITERIA = ["abstract_similarity"]

    # Metadata for LM-based criteria (descriptions, schema hints, etc.)
    CRITERION_METADATA = ABSTRACTSIM_CRITERIA

    def __init__(self, config: dict | None = None):
        """Initialize AbstractSimDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}

    def load_documents(self) -> list[Any]:
        """Load Wikipedia sentences and descriptions from HuggingFace.

        For each row, creates documents with n*2 labeling:
        1. Sentence: row_X_class_0
        2. Good descriptions: row_X_class_0 (same as sentence)
        3. Bad descriptions: row_X_class_1

        This guarantees that triplets only use within-row matches.

        Returns:
            List of document dicts: {"text": str, "abstract_similarity": str}
        """
        logger.info(f"Loading Abstract-Sim from HuggingFace: {self.DATASET_PATH}")

        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")  # Default to train

        # Use streaming for large datasets or when max_docs is set
        use_streaming = max_docs is not None and max_docs < 10000

        if use_streaming:
            logger.debug(f"Using streaming mode with max_docs={max_docs}")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42)
        else:
            logger.debug(f"Loading full dataset split: {split}")
            dataset = load_dataset(self.DATASET_PATH, split=split)
            dataset = dataset.shuffle(seed=42)

        # Extract documents with n*2 labeling
        documents = []
        doc_count = 0

        for row_idx, example in enumerate(dataset):
            try:
                # Extract fields
                sentence = example.get("sentence", "").strip()
                good_list = example.get("good", [])
                bad_list = example.get("bad", [])

                # Skip if invalid (need sentence and at least one good/bad description)
                if not sentence or (not good_list and not bad_list):
                    logger.debug(
                        f"Skipping row {row_idx}: missing sentence or descriptions"
                    )
                    continue

                # Create labels for this row (n*2 scheme)
                label_class_0 = f"row_{row_idx}_class_0"
                label_class_1 = f"row_{row_idx}_class_1"

                # Add sentence as document (class_0)
                # Mark as anchor so prelabeled triplet creation uses it
                documents.append(
                    {
                        "text": sentence,
                        "abstract_similarity": label_class_0,
                        "is_anchor": True,  # Marker for triplet anchor selection
                        "is_sentence": True,  # Also mark as sentence for clarity
                    }
                )
                doc_count += 1

                # Add all good descriptions (class_0 - same label as sentence)
                for good_desc in good_list:
                    good_text = (
                        good_desc.strip()
                        if isinstance(good_desc, str)
                        else str(good_desc)
                    )
                    if good_text:
                        documents.append(
                            {
                                "text": good_text,
                                "abstract_similarity": label_class_0,
                            }
                        )
                        doc_count += 1

                # Add all bad descriptions (class_1 - different label)
                for bad_desc in bad_list:
                    bad_text = (
                        bad_desc.strip() if isinstance(bad_desc, str) else str(bad_desc)
                    )
                    if bad_text:
                        documents.append(
                            {
                                "text": bad_text,
                                "abstract_similarity": label_class_1,
                            }
                        )
                        doc_count += 1

                # Check max_docs limit (applies to total documents)
                if max_docs and doc_count >= max_docs:
                    logger.debug(f"Reached max_docs limit: {max_docs}")
                    break

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed row at index {row_idx}: {e}")
                continue

        logger.info(
            f"Loaded {len(documents)} documents from {row_idx + 1} rows "
            f"(split: {split})"
        )

        # Build precomputed annotations
        self._build_precomputed_annotations(documents)

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document (dict or string)

        Returns:
            Text content
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document) if document else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - abstract_similarity: the row label (row_X_positive or row_X_negative)

        Args:
            document: A document
            criterion: The criterion name

        Returns:
            Criterion value for abstract_similarity, or None
        """
        if criterion == "abstract_similarity":
            if isinstance(document, dict):
                return document.get("abstract_similarity")
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations(self, documents: list[dict]) -> None:
        """Build precomputed annotations from loaded documents.

        Creates a mapping: {document_text: {"criterion_value": label}}
        where label is "row_X_class_0" or "row_X_class_1"

        Args:
            documents: List of document dicts with 'text' and 'abstract_similarity' fields
        """
        annotations = {}
        n_class_0 = 0
        n_class_1 = 0

        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text")
                label = doc.get("abstract_similarity")

                if text and label:
                    annotations[text] = {"criterion_value": label}

                    # Track stats
                    if "_class_0" in label:
                        n_class_0 += 1
                    elif "_class_1" in label:
                        n_class_1 += 1

        self.PRECOMPUTED_ANNOTATIONS["abstract_similarity"] = annotations

        logger.info(
            f"Built precomputed annotations for abstract_similarity: "
            f"{len(annotations)} documents "
            f"({n_class_0} class_0, {n_class_1} class_1 labels)"
        )
