"""Infinite chats prompts document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.constants import INFINITE_CHATS_DATASET_ID
from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class InfinitePromptsDocSet(BaseDocSet):
    """User prompts from infinite-chats dataset with category tags."""

    DATASET_PATH = INFINITE_CHATS_DATASET_ID
    DESCRIPTION = "User prompts from infinite-chats dataset with category tags"
    KNOWN_CRITERIA = ["categories"]

    def __init__(self, config: dict | None = None):
        """Initialize InfinitePromptsDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}

    def load_documents(self) -> list[Any]:
        """Load prompts (first user messages) from infinite-chats.

        Always uses streaming mode to avoid downloading the full dataset.

        Documents are simple strings (the prompt text).
        Metadata is stored separately in PRECOMPUTED_ANNOTATIONS.

        Returns:
            List of prompt text strings
        """
        logger.info(f"Loading Infinite Prompts from HuggingFace: {self.DATASET_PATH}")

        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        category_filter = self.config.get("category_filter")

        # Always use streaming mode to avoid downloading full dataset
        logger.debug("Using streaming mode for infinite-chats dataset")
        dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
        dataset = dataset.shuffle(seed=42)

        documents = []
        metadata_list = []  # Store metadata separately for annotation building

        for example in dataset:
            # Extract first user message
            messages = example.get("messages", [])
            first_user_msg = None
            for msg in messages:
                if msg.get("role") == "user":
                    first_user_msg = msg.get("content", "").strip()
                    break

            if not first_user_msg:
                continue

            # Extract categories
            categories = example.get("categories", [])
            category_names = []
            for cat in categories:
                if isinstance(cat, dict):
                    cat_name = cat.get("category", "")
                    if cat_name:
                        category_names.append(cat_name)

            # Apply category filter if specified
            if category_filter:
                if not any(cat in category_filter for cat in category_names):
                    continue

            # Store text as document (string, not dict)
            documents.append(first_user_msg)

            # Store metadata separately for precomputed annotations
            metadata_list.append(
                {
                    "text": first_user_msg,
                    "categories": category_names,
                }
            )

            # Respect max_docs limit
            if max_docs is not None and len(documents) >= max_docs:
                break

        logger.debug(
            f"Loaded {len(documents)} prompts from Infinite Chats "
            f"(category_filter={category_filter})"
        )

        # Build precomputed annotations for categories criterion
        self._build_precomputed_annotations(metadata_list)

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Documents are simple strings (prompt text).

        Args:
            document: A document string

        Returns:
            The text content of the document
        """
        return document if isinstance(document, str) else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - categories: comma-separated category tags (from PRECOMPUTED_ANNOTATIONS)

        Note: Documents are strings, so criterion values come from PRECOMPUTED_ANNOTATIONS.
        Use get_precomputed_annotation() to access these.
        """
        if criterion == "categories":
            # Criterion values are stored in PRECOMPUTED_ANNOTATIONS
            # The caller should use get_precomputed_annotation() instead
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations(self, metadata_list: list[dict]) -> None:
        """Build precomputed annotations from loaded documents.

        Creates a mapping: {document_text: {"criterion_value": categories_string}}

        Args:
            metadata_list: List of metadata dicts with 'text' and 'categories' fields
        """
        annotations = {}

        for metadata in metadata_list:
            text = metadata.get("text")
            categories = metadata.get("categories", [])

            if text:
                # Convert categories list to comma-separated string
                categories_str = ", ".join(categories) if categories else ""
                annotations[text] = {"criterion_value": categories_str}

        self.PRECOMPUTED_ANNOTATIONS["categories"] = annotations

        logger.info(
            f"Built precomputed annotations for categories: {len(annotations)} documents"
        )
