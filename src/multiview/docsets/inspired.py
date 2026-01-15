"""Inspired movie recommendation dialogues from InstructLF.

Loads movie recommendation conversations from GitHub (allenai/instructLF).
Auto-clones repo to ~/.cache/multiview/instructLF/ on first use, pulls updates on subsequent runs.

Dataset from: https://github.com/allenai/instructLF
"""

from __future__ import annotations

import ast
import logging
import subprocess
from typing import Any

import pandas as pd

from multiview.constants import (
    INSTRUCTLF_CACHE_DIR,
    INSTRUCTLF_INSPIRED_TEST,
    INSTRUCTLF_INSPIRED_TRAIN,
    INSTRUCTLF_REPO_URL,
)
from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import INSPIRED_CRITERIA

logger = logging.getLogger(__name__)


class InspiredDocSet(BaseDocSet):
    """Inspired movie recommendation dialogues from InstructLF.

    Dataset contains conversations where users discuss movie preferences and receive recommendations.
    Each document is a chat dialogue (full_situation), labeled with recommended movie(s).

    Config parameters:
        split (str): "train" or "test" (default: "train")
        max_docs (int, optional): Maximum documents to load
        flatten_movies (bool): If True, create one doc per movie; if False, one doc per chat (default: False)

    Usage:
        tasks:
          - document_set: inspired
            criterion: movie_recommendation
            triplet_style: prelabeled
            config:
              split: train
              max_docs: 200
    """

    DATASET_PATH = str(INSTRUCTLF_INSPIRED_TRAIN)
    DESCRIPTION = "Movie recommendation dialogues from InstructLF"
    KNOWN_CRITERIA = ["movie_recommendation"]
    CRITERION_METADATA = INSPIRED_CRITERIA

    def __init__(self, config: dict | None = None):
        """Initialize Inspired dataset.

        Ensures the InstructLF git repo is cloned before loading.

        Config params:
            split: "train" or "test" (default: "train")
            max_docs: Maximum documents to load (optional)
            flatten_movies: Create one doc per movie recommendation (default: False)
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

        # Verify data files exist
        if not INSTRUCTLF_INSPIRED_TRAIN.exists():
            raise RuntimeError(
                f"Inspired train data not found at {INSTRUCTLF_INSPIRED_TRAIN}"
            )
        if not INSTRUCTLF_INSPIRED_TEST.exists():
            raise RuntimeError(
                f"Inspired test data not found at {INSTRUCTLF_INSPIRED_TEST}"
            )

    def load_documents(self) -> list[Any]:
        """Load Inspired movie recommendation dialogues.

        Returns:
            List of documents. Each document is either:
            - dict with "text" (chat dialogue) and "movie_recommendation" fields if not flattened
            - dict with "text" and "movie_recommendation" (single movie) if flattened
        """
        # Determine which CSV to load
        split = self.config.get("split", "train")
        csv_path = (
            INSTRUCTLF_INSPIRED_TRAIN if split == "train" else INSTRUCTLF_INSPIRED_TEST
        )

        logger.info(f"Loading Inspired {split} data from {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path)

        # Parse and create documents
        documents = []
        max_docs = self.config.get("max_docs")
        flatten_movies = self.config.get("flatten_movies", False)

        # Determine column names based on split (train vs test have different formats)
        if "full_situation" in df.columns:
            # Training format
            chat_col = "full_situation"
            movies_col = "movies"
        elif "test_inputs" in df.columns:
            # Test format
            chat_col = "test_inputs"
            movies_col = "test_outputs"
        else:
            raise ValueError(
                f"Unknown CSV format. Expected columns 'full_situation'/'movies' or 'test_inputs'/'test_outputs', got {df.columns.tolist()}"
            )

        for idx, row in df.iterrows():
            chat = row[chat_col]
            movies_str = row[movies_col]

            # Skip if chat is empty
            if pd.isna(chat) or not chat.strip():
                continue

            # Parse movie list from string representation
            try:
                movies = ast.literal_eval(movies_str)
                if not isinstance(movies, list):
                    movies = [movies]
            except (ValueError, SyntaxError):
                logger.warning(f"Failed to parse movies at row {idx}: {movies_str}")
                continue

            # Skip if no movies
            if not movies:
                continue

            # Create documents based on flatten_movies setting
            if flatten_movies:
                # Create one document per movie
                for movie in movies:
                    documents.append({"text": chat, "movie_recommendation": str(movie)})
            else:
                # Create one document per chat with all movies
                # Use first movie as the label for simplicity
                movie_label = movies[0] if len(movies) == 1 else movies
                documents.append(
                    {
                        "text": chat,
                        "movie_recommendation": str(movie_label)
                        if isinstance(movie_label, str)
                        else str(movies[0]),
                    }
                )

            # Check max_docs limit
            if max_docs and len(documents) >= max_docs:
                break

        logger.info(f"Loaded {len(documents)} documents from Inspired {split} split")

        # Build precomputed annotations
        self._build_precomputed_annotations(documents)

        return documents

    def _build_precomputed_annotations(self, documents: list[dict]) -> None:
        """Build precomputed annotations mapping for movie recommendations.

        Args:
            documents: List of document dicts with "text" and "movie_recommendation" fields
        """
        annotations = {}
        for doc in documents:
            text = doc["text"]
            rec = doc["movie_recommendation"]
            annotations[text] = {"criterion_value": str(rec)}

        self.PRECOMPUTED_ANNOTATIONS["movie_recommendation"] = annotations
        logger.debug(f"Built precomputed annotations for {len(annotations)} documents")

    def get_document_text(self, document: Any) -> str:
        """Extract text from document.

        Args:
            document: Document dict or string

        Returns:
            Document text (chat dialogue)
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document)

    def get_known_criterion_value(self, document: Any, criterion: str) -> Any:
        """Get the known criterion value for a document.

        Args:
            document: Document dict
            criterion: Criterion name (e.g., "movie_recommendation")

        Returns:
            Criterion value (movie name or None)
        """
        if isinstance(document, dict) and criterion in document:
            return document[criterion]
        return None
