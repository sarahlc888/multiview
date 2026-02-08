"""Abstract-Sim dataset loader from BIU NLP.

Loads Wikipedia sentences with abstract descriptions (good/bad) from HuggingFace.
Uses n*2 labeling scheme to guarantee within-row triplet matching.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

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
    KNOWN_CRITERIA = ["abstract_similarity", "abstraction_level"]

    # Metadata for LM-based criteria (descriptions, schema hints, etc.)
    DATASET_NAME = "abstractsim"

    def __init__(self, config: dict | None = None):
        """Initialize AbstractSimDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}

        # Store operational metadata for triplet generation
        # Maps document_text -> metadata dict
        self._triplet_metadata: dict[str, dict] = {}

    def load_documents(self) -> list[Any]:
        """Load Wikipedia sentences and descriptions from HuggingFace.

        Documents are simple strings (sentences and descriptions).
        Metadata for triplet generation is stored in self._triplet_metadata.

        Returns:
            List of document strings
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
            dataset = dataset.shuffle(seed=42, buffer_size=10000)
        else:
            logger.debug(f"Loading full dataset split: {split}")
            dataset = load_dataset(self.DATASET_PATH, split=split)
            dataset = dataset.shuffle(seed=42)

        # Extract documents with n*2 labeling
        documents = []
        metadata_list = []  # Temporary list to store metadata during loading
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

                # Add sentence - just the text
                documents.append(sentence)
                metadata_list.append(
                    {
                        "text": sentence,
                        "abstract_similarity": label_class_0,
                        "abstraction_level": "concrete",
                        "is_anchor": True,
                        "is_sentence": True,
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
                        documents.append(good_text)
                        metadata_list.append(
                            {
                                "text": good_text,
                                "abstract_similarity": label_class_0,
                                "abstraction_level": "abstract",
                            }
                        )
                        doc_count += 1

                # Add all bad descriptions (class_1 - different label)
                for bad_desc in bad_list:
                    bad_text = (
                        bad_desc.strip() if isinstance(bad_desc, str) else str(bad_desc)
                    )
                    if bad_text:
                        documents.append(bad_text)
                        metadata_list.append(
                            {
                                "text": bad_text,
                                "abstract_similarity": label_class_1,
                                "abstraction_level": "abstract",
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

        # Deduplicate before building metadata lookups
        documents = self._deduplicate(documents)

        # Build metadata lookups
        self._build_metadata_lookups(metadata_list)

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document (string)

        Returns:
            Text content
        """
        return str(document) if document else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - abstract_similarity: the row label (from PRECOMPUTED_ANNOTATIONS)
        - abstraction_level: "concrete" or "abstract" (from PRECOMPUTED_ANNOTATIONS)

        Note: Documents are strings, so criterion values come from PRECOMPUTED_ANNOTATIONS.
        Use get_precomputed_annotation() to access these.

        Args:
            document: A document (string)
            criterion: The criterion name

        Returns:
            Criterion value or None
        """
        if criterion in ["abstract_similarity", "abstraction_level"]:
            # Criterion values are stored in PRECOMPUTED_ANNOTATIONS
            # The caller should use get_precomputed_annotation() instead
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_metadata_lookups(self, metadata_list: list[dict]) -> None:
        """Build metadata lookups from loaded documents.

        Creates:
        1. PRECOMPUTED_ANNOTATIONS["abstract_similarity"]: {document_text: {"prelabel": "row_X_class_0/1"}}
        2. PRECOMPUTED_ANNOTATIONS["abstraction_level"]: {document_text: {"prelabel": "concrete"/"abstract"}}
        3. self._triplet_metadata: {document_text: metadata_dict}

        Args:
            metadata_list: List of metadata dicts with 'text' and other fields
        """
        sim_annotations = {}
        level_annotations = {}
        n_class_0 = 0
        n_class_1 = 0
        n_concrete = 0
        n_abstract = 0

        for meta in metadata_list:
            text = meta.get("text")
            if not text:
                continue

            sim_label = meta.get("abstract_similarity")
            level_label = meta.get("abstraction_level")

            # Build precomputed annotations for criteria
            if sim_label:
                sim_annotations[text] = {"prelabel": sim_label}

                # Track stats
                if "_class_0" in sim_label:
                    n_class_0 += 1
                elif "_class_1" in sim_label:
                    n_class_1 += 1

            if level_label:
                level_annotations[text] = {"prelabel": level_label}

                # Track stats
                if level_label == "concrete":
                    n_concrete += 1
                elif level_label == "abstract":
                    n_abstract += 1

            # Store full metadata for triplet generation
            self._triplet_metadata[text] = meta

        self.PRECOMPUTED_ANNOTATIONS["abstract_similarity"] = sim_annotations
        self.PRECOMPUTED_ANNOTATIONS["abstraction_level"] = level_annotations

        logger.info(
            f"Built precomputed annotations for abstract_similarity: "
            f"{len(sim_annotations)} documents "
            f"({n_class_0} class_0, {n_class_1} class_1 labels)"
        )
        logger.info(
            f"Built precomputed annotations for abstraction_level: "
            f"{len(level_annotations)} documents "
            f"({n_concrete} concrete, {n_abstract} abstract labels)"
        )
        logger.info(
            f"Built triplet metadata for {len(self._triplet_metadata)} documents"
        )

    def get_document_metadata(self, doc_idx: int) -> dict[str, Any]:
        """Get metadata for a document at the given index.

        Args:
            doc_idx: Document index

        Returns:
            Dict with doc_type ("sentence" or "description") and abstraction_level
        """
        if (
            not hasattr(self, "documents")
            or doc_idx < 0
            or doc_idx >= len(self.documents)
        ):
            return {}

        doc_text = str(self.documents[doc_idx])
        metadata = self._triplet_metadata.get(doc_text, {})

        result = {}
        # Determine doc_type based on is_sentence flag
        if metadata.get("is_sentence"):
            result["doc_type"] = "sentence"
        else:
            result["doc_type"] = "description"

        # Add abstraction_level if available
        if "abstraction_level" in metadata:
            result["abstraction_level"] = metadata["abstraction_level"]

        return result
