"""Infinite chats responses document_set loader."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from datasets import load_dataset

from multiview.constants import INFINITE_CHATS_DATASET_ID
from multiview.docsets.base import BaseDocSet
from multiview.utils.sampling_utils import deterministic_sample

logger = logging.getLogger(__name__)


class InfiniteChatsDocSet(BaseDocSet):
    """Full conversations from infinite-chats grouped by prompt (AidanBench-style).

    Streams a sample of conversations, groups them by prompt, then returns
    full conversations for a selected prompt.

    Config parameters:
        prompt_id (int, optional): Index of prompt to load (e.g., 0, 1, 2)
        prompt_text (str, optional): Text to search for in prompts (partial match)
        sample_size (int): Number of conversations to stream for grouping (default: 1000)
        max_docs (int, optional): Maximum responses to return for selected prompt
        min_group_size (int): Minimum responses per prompt to include (default: 1)
        If neither prompt_id/prompt_text provided, uses first prompt.
    """

    DATASET_PATH = INFINITE_CHATS_DATASET_ID
    DESCRIPTION = "Full conversations from infinite-chats grouped by prompt"
    KNOWN_CRITERIA = []  # Documents are strings; no metadata extracted

    def __init__(self, config: dict | None = None):
        """Initialize infinite-chats dataset.

        Config params:
            prompt_id: Index of prompt to load (e.g., 0, 1, 2)
            prompt_text: Text of prompt to load (partial match)
            sample_size: Number of conversations to stream (default: 1000)
            min_group_size: Minimum number of responses per prompt (default: 1)
        """
        super().__init__(config)
        self.prompt_id = config.get("prompt_id") if config else None
        self.prompt_text = config.get("prompt_text") if config else None
        self.sample_size = config.get("sample_size", 1000) if config else 1000
        self.min_group_size = config.get("min_group_size", 1) if config else 1

    def load_documents(self) -> list[Any]:
        """Load full conversations for a SPECIFIC prompt from infinite-chats.

        Streams a sample of conversations, groups by prompt, then returns
        full conversations for the selected prompt.

        Returns:
            List of full conversation strings (formatted as "=== ROLE ===\\ncontent\\n\\n=== ROLE ===\\ncontent...")
        """
        logger.info(f"Loading Infinite Chats from HuggingFace: {self.DATASET_PATH}")

        split = self.config.get("split", "train")
        category_filter = self.config.get("category_filter")

        # Stream a sample of conversations
        logger.debug(f"Streaming {self.sample_size} conversations for grouping")
        dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
        dataset = dataset.shuffle(seed=42)

        # Group conversations by first user message (prompt)
        prompt_to_responses = defaultdict(list)
        conversations_processed = 0

        for example in dataset:
            messages = example.get("messages", [])

            # Extract first user message (prompt) for grouping
            first_user_msg = None
            for msg in messages:
                if msg.get("role") == "user":
                    first_user_msg = msg.get("content", "").strip()
                    break

            if not first_user_msg:
                continue

            # Check that conversation has at least one assistant response
            has_assistant = any(msg.get("role") == "assistant" for msg in messages)
            if not has_assistant:
                continue

            # Format the full conversation with clear role separators
            conversation_text = []
            for msg in messages:
                role = msg.get("role", "").upper()
                content = msg.get("content", "").strip()
                if role and content:
                    # Use distinctive separators that won't get lost in the content
                    conversation_text.append(f"=== {role} ===\n{content}")

            full_conversation = "\n\n".join(conversation_text)

            # Extract categories for filtering
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

            # Group full conversation by prompt
            prompt_to_responses[first_user_msg].append(full_conversation)

            conversations_processed += 1
            if conversations_processed >= self.sample_size:
                break

        # Filter by min_group_size
        prompt_to_responses = {
            prompt: responses
            for prompt, responses in prompt_to_responses.items()
            if len(responses) >= self.min_group_size
        }

        # Deduplicate responses per prompt
        prompt_to_responses = {
            prompt: sorted(set(responses))
            for prompt, responses in prompt_to_responses.items()
        }

        logger.info(
            f"Grouped {conversations_processed} conversations into "
            f"{len(prompt_to_responses)} unique prompts "
            f"(min_group_size={self.min_group_size})"
        )

        if not prompt_to_responses:
            logger.warning("No prompts found matching criteria")
            return []

        # Select which prompt(s) to use
        prompts = list(prompt_to_responses.keys())

        if self.prompt_id is not None:
            # Use prompt by index - return responses for this specific prompt
            if self.prompt_id >= len(prompts):
                raise ValueError(
                    f"prompt_id {self.prompt_id} out of range "
                    f"(only {len(prompts)} prompts available)"
                )
            selected_prompt = prompts[self.prompt_id]
            logger.info(f"Selected prompt: {selected_prompt[:100]}...")
            documents = prompt_to_responses[selected_prompt]
        elif self.prompt_text is not None:
            # Use prompt by text match - return responses for this specific prompt
            matches = [p for p in prompts if self.prompt_text.lower() in p.lower()]
            if not matches:
                raise ValueError(f"No prompt found matching '{self.prompt_text}'")
            selected_prompt = matches[0]
            if len(matches) > 1:
                logger.warning(
                    f"Multiple prompts match '{self.prompt_text}', using first"
                )
            logger.info(f"Selected prompt: {selected_prompt[:100]}...")
            documents = prompt_to_responses[selected_prompt]
        else:
            # No prompt specified - return responses from multiple prompts to meet max_docs
            logger.info(
                "No prompt specified, collecting responses from multiple prompts"
            )
            documents = []
            for prompt in prompts:
                documents.extend(prompt_to_responses[prompt])
                max_docs = self.config.get("max_docs")
                if max_docs and len(documents) >= max_docs:
                    break

        # Apply max_docs limit
        max_docs = self.config.get("max_docs")
        if max_docs is not None and len(documents) > max_docs:
            # Deterministically sample max_docs from the documents
            sampled_documents = deterministic_sample(
                documents, k=max_docs, seed_base="infinite_chats_max_docs"
            )
            documents = sampled_documents

        logger.debug(f"Returning {len(documents)} full conversation documents")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Documents are simple strings (full conversation text).

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

        Note: Documents are strings with no metadata extracted.
        """
        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)
