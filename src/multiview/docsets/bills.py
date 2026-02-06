"""Legislative bills with topic labels from InstructLF.

Loads US legislative bills with topic annotations from GitHub (allenai/instructLF).
Auto-clones repo to ~/.cache/multiview/instructLF/ on first use, pulls updates on subsequent runs.

Dataset from: https://github.com/allenai/instructLF
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

from multiview.constants import (
    INSTRUCTLF_BILLS_DATA,
    INSTRUCTLF_CACHE_DIR,
    INSTRUCTLF_REPO_URL,
)
from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class BillsDocSet(BaseDocSet):
    """Legislative bills with topic annotations from InstructLF.

    Dataset contains US legislative bills with topic and subtopic labels.
    Each document is a bill summary or tokenized text, labeled with policy topic/subtopic.

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        text_field (str): Which field to use as text - "summary" or "tokenized_text" (default: "summary")

    Usage:
        tasks:
          - document_set: bills
            criterion: topic
            triplet_style: prelabeled
            config:
              max_docs: 300
              text_field: summary
    """

    DATASET_PATH = str(INSTRUCTLF_BILLS_DATA)
    DESCRIPTION = "Legislative bills with topic labels from InstructLF"
    KNOWN_CRITERIA = ["topic", "subtopic"]
    DATASET_NAME = "bills"

    def __init__(self, config: dict | None = None):
        """Initialize Bills dataset.

        Ensures the InstructLF git repo is cloned before loading.

        Config params:
            max_docs: Maximum documents to load (optional)
            text_field: "summary" or "tokenized_text" (default: "summary")
        """
        super().__init__(config)
        self._ensure_instructlf_repo_cloned()
        self.PRECOMPUTED_ANNOTATIONS = {}

    def _ensure_instructlf_repo_cloned(self) -> None:
        """Ensure InstructLF repo is cloned and up to date."""
        if not INSTRUCTLF_CACHE_DIR.exists():
            logger.info(f"Cloning InstructLF repo to {INSTRUCTLF_CACHE_DIR}")
            try:
                INSTRUCTLF_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", INSTRUCTLF_REPO_URL, str(INSTRUCTLF_CACHE_DIR)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug("Successfully cloned InstructLF repo")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to clone InstructLF repo: {e.stderr}"
                ) from e
        else:
            # Pull latest changes
            logger.debug(f"Pulling latest for InstructLF at {INSTRUCTLF_CACHE_DIR}")
            try:
                subprocess.run(
                    ["git", "-C", str(INSTRUCTLF_CACHE_DIR), "pull"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to pull InstructLF updates: {e.stderr}")
                # Don't fail - continue with existing data

        # Verify data file exists
        if not INSTRUCTLF_BILLS_DATA.exists():
            raise RuntimeError(f"Bills data not found at {INSTRUCTLF_BILLS_DATA}")

    def load_documents(self) -> list[Any]:
        """Load legislative bills with topic annotations.

        Returns:
            List of document dicts with "text", "topic", and "subtopic" fields
        """
        logger.info(f"Loading Bills data from {INSTRUCTLF_BILLS_DATA}")

        documents = []
        max_docs = self.config.get("max_docs")
        text_field = self.config.get("text_field", "summary")

        # Load JSONL file
        with open(INSTRUCTLF_BILLS_DATA, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)

                    # Extract text based on configured field
                    text = record.get(text_field, "")
                    if not text or not text.strip():
                        # Fallback to summary if tokenized_text is empty
                        if text_field == "tokenized_text":
                            text = record.get("summary", "")
                        if not text or not text.strip():
                            logger.debug(f"Skipping record {line_num} with empty text")
                            continue

                    # Extract topic and subtopic
                    topic = record.get("topic", "Unknown")
                    subtopic = record.get("subtopic", "Unknown")

                    doc = {
                        "text": text,
                        "topic": topic,
                        "subtopic": subtopic,
                    }

                    documents.append(doc)

                    # Check max_docs limit
                    if max_docs and len(documents) >= max_docs:
                        break

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON at line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(documents)} bills")

        # Build precomputed annotations for both criteria
        self._build_precomputed_annotations(documents)

        return documents

    def _build_precomputed_annotations(self, documents: list[dict]) -> None:
        """Build precomputed annotations mapping for topic and subtopic.

        Args:
            documents: List of document dicts with "text", "topic", and "subtopic" fields
        """
        topic_annotations = {}
        subtopic_annotations = {}

        for doc in documents:
            text = doc["text"]
            topic_annotations[text] = {"prelabel": doc["topic"]}
            subtopic_annotations[text] = {"prelabel": doc["subtopic"]}

        self.PRECOMPUTED_ANNOTATIONS["topic"] = topic_annotations
        self.PRECOMPUTED_ANNOTATIONS["subtopic"] = subtopic_annotations

        logger.debug(
            f"Built precomputed annotations for {len(topic_annotations)} documents "
            f"(2 criteria: topic, subtopic)"
        )

    def get_document_text(self, document: Any) -> str:
        """Extract text from document.

        Args:
            document: Document dict or string

        Returns:
            Document text (bill summary or tokenized text)
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document)

    def get_known_criterion_value(self, document: Any, criterion: str) -> Any:
        """Get the known criterion value for a document.

        Args:
            document: Document dict
            criterion: Criterion name (e.g., "topic" or "subtopic")

        Returns:
            Criterion value (topic/subtopic label or None)
        """
        if isinstance(document, dict) and criterion in document:
            return document[criterion]
        return None
