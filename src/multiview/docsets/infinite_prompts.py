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

    def load_documents(self) -> list[Any]:
        """Load prompts (first user messages) from infinite-chats.

        Always uses streaming mode to avoid downloading the full dataset.

        Returns:
            List of document dicts with 'text' and 'categories' keys
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

            # Store document with metadata
            doc = {
                "text": first_user_msg,
                "categories": category_names,
                "conversation_id": example.get("conversation_id", ""),
            }
            documents.append(doc)

            # Respect max_docs limit
            if max_docs is not None and len(documents) >= max_docs:
                break

        logger.debug(
            f"Loaded {len(documents)} prompts from Infinite Chats "
            f"(category_filter={category_filter})"
        )
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document dict or string

        Returns:
            The text content of the document
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return document if isinstance(document, str) else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - categories: comma-separated category tags
        """
        if criterion == "categories":
            if isinstance(document, dict):
                categories = document.get("categories", [])
                return ", ".join(categories) if categories else ""
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)
