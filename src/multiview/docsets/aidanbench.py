"""AidanBench document_set loader.

Loads open-ended question answers from GitHub (aidanmclaughlin/AidanBench).
Auto-clones repo to ~/.cache/multiview/AidanBench/ on first use, pulls updates on subsequent runs.

Structure: Nested JSON models[model][temp][question][answers]
Each question is treated as a separate DocumentSet (answers not comparable across questions).

Selection: Use question_id (index) or question_text (search) to specify which question.
Use AidanBenchDocSet.list_questions() to discover available questions.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections import defaultdict
from typing import Any

from multiview.constants import (
    AIDANBENCH_CACHE_DIR,
    AIDANBENCH_REPO_URL,
    AIDANBENCH_RESULTS_PATH,
)
from multiview.docsets.base import BaseDocSet
from multiview.utils.sampling_utils import deterministic_sample

logger = logging.getLogger(__name__)


class AidanBenchDocSet(BaseDocSet):
    """AidanBench open-ended question answers document_set.

    IMPORTANT: Each question is treated as a separate DocumentSet.
    Specify which question to load via config['question_id'] or config['question_text'].

    Config parameters:
        question_id (int, optional): Index of question to load (e.g., 0, 1, 2)
        question_text (str, optional): Text to search for in questions (partial match)
        max_docs (int, optional): Maximum answers to return for selected question
        If neither question_id/question_text provided, loads first question.
    """

    DATASET_PATH = str(AIDANBENCH_RESULTS_PATH)
    DESCRIPTION = "AidanBench open-ended question answers (per-question)"
    KNOWN_CRITERIA = []  # Documents are strings; no metadata extracted

    def __init__(self, config: dict | None = None):
        """Initialize AidanBench dataset.

        Ensures the git repo is cloned before loading.

        Config params:
            question_id: Index of question to load (e.g., 0, 1, 2)
            question_text: Text of question to load (partial match)
            If neither provided, loads first question.
        """
        super().__init__(config)
        self._ensure_repo_cloned()
        self.question_id = config.get("question_id") if config else None
        self.question_text = config.get("question_text") if config else None

    def _ensure_repo_cloned(self) -> None:
        """Ensure AidanBench repo is cloned and up to date."""
        if not AIDANBENCH_CACHE_DIR.exists():
            logger.info(f"Cloning AidanBench repo to {AIDANBENCH_CACHE_DIR}")
            try:
                AIDANBENCH_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "clone", AIDANBENCH_REPO_URL, str(AIDANBENCH_CACHE_DIR)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.debug("Successfully cloned AidanBench repo")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Failed to clone AidanBench repo: {e.stderr}"
                ) from e
        else:
            # Pull latest changes
            logger.debug(f"Pulling latest for AidanBench at {AIDANBENCH_CACHE_DIR}")
            try:
                subprocess.run(
                    ["git", "-C", str(AIDANBENCH_CACHE_DIR), "pull"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to pull AidanBench updates: {e.stderr}")
                # Don't fail - continue with existing data

        # Verify results.json exists
        if not AIDANBENCH_RESULTS_PATH.exists():
            raise RuntimeError(
                f"AidanBench results.json not found at {AIDANBENCH_RESULTS_PATH}"
            )

    def load_documents(self) -> list[Any]:
        """Load unique answers for a SPECIFIC question from AidanBench.

        Returns:
            List of unique answer strings for the selected question
        """
        logger.info(f"Loading AidanBench from {AIDANBENCH_RESULTS_PATH}")

        # Load JSON file
        try:
            with open(AIDANBENCH_RESULTS_PATH, encoding="utf-8") as f:
                raw_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise RuntimeError(f"Failed to load AidanBench results.json: {e}") from e

        # Extract unique answers per question
        # Structure: models[model][temp][question][answers]
        question_to_answers = defaultdict(list)

        for model in sorted(raw_data.get("models", {}).keys()):
            for temp in sorted(raw_data["models"][model].keys()):
                for question in sorted(raw_data["models"][model][temp].keys()):
                    answers = raw_data["models"][model][temp][question]
                    for answer_obj in answers:
                        if isinstance(answer_obj, dict):
                            answer_text = answer_obj.get("answer", "").strip()
                            if answer_text:
                                question_to_answers[question].append(answer_text)

        # Deduplicate answers per question
        question_to_answers = {
            q: sorted(set(answers)) for q, answers in question_to_answers.items()
        }

        logger.debug(f"Loaded {len(question_to_answers)} questions total")

        if not question_to_answers:
            logger.warning("No questions found in AidanBench data")
            return []

        # Select which question to use
        questions = list(question_to_answers.keys())
        if self.question_id is not None:
            # Use question by index
            if self.question_id >= len(questions):
                raise ValueError(
                    f"question_id {self.question_id} out of range "
                    f"(only {len(questions)} questions)"
                )
            selected_question = questions[self.question_id]
        elif self.question_text is not None:
            # Use question by text match
            matches = [q for q in questions if self.question_text.lower() in q.lower()]
            if not matches:
                raise ValueError(
                    f"No question found matching '{self.question_text}'"
                )
            selected_question = matches[0]
            if len(matches) > 1:
                logger.warning(
                    f"Multiple questions match '{self.question_text}', using first"
                )
        else:
            # Default to first question
            selected_question = questions[0]
            logger.info("No question specified, using first question")

        logger.info(f"Selected question: {selected_question[:100]}...")

        # Get answers for this question only
        documents = question_to_answers[selected_question]

        # Apply max_docs limit
        max_docs = self.config.get("max_docs")
        if max_docs is not None and len(documents) > max_docs:
            # Deterministically sample max_docs from the documents
            sampled_documents = deterministic_sample(
                documents, k=max_docs, seed_base="aidanbench_max_docs"
            )
            documents = sampled_documents

        logger.debug(
            f"Returning {len(documents)} answer documents for selected question"
        )
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Documents are simple strings (answer text).

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

    @staticmethod
    def list_questions() -> list[str]:
        """List all available questions in AidanBench.

        Helper method for discovering which questions are available.

        Returns:
            List of question strings
        """
        # Ensure repo is downloaded
        temp_docset = AidanBenchDocSet(config={})  # Trigger download

        # Load and extract questions
        with open(AIDANBENCH_RESULTS_PATH, encoding="utf-8") as f:
            raw_data = json.load(f)

        questions = set()
        for model in raw_data.get("models", {}).values():
            for temp_data in model.values():
                questions.update(temp_data.keys())

        return sorted(questions)
