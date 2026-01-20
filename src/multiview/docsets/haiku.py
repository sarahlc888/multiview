"""Haiku document set loader.

Loads English haiku poems and analyzes what they evoke beneath the surface,
focusing on philosophical depth, imagery, and implied meanings.
"""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import HAIKU_CRITERIA

logger = logging.getLogger(__name__)


class HaikuDocSet(BaseDocSet):
    """English haiku poems dataset.

    Focuses on analyzing the deeper meanings, philosophical insights,
    and emotional/conceptual evocations beneath the surface of haiku.
    """

    # Metadata
    DATASET_PATH = "taucris/haiku_333K"  # "statworx/haiku"
    DESCRIPTION = (
        "English haiku poems with analysis of deeper meanings and philosophical themes"
    )
    DOCUMENT_TYPE = "Haiku poem"

    # Criteria that can be extracted deterministically (no LLM needed)
    # word_count is automatically included by base class
    KNOWN_CRITERIA = []

    # Metadata for LM-based criteria (descriptions and schema hints)
    CRITERION_METADATA = HAIKU_CRITERIA
    # Synthesis prompts for LM-based document generation
    SYNTHESIS_CONFIGS = {}

    def load_documents(self) -> list[Any]:
        """Load haiku poems from Hugging Face.

        Returns:
            List of haiku texts
        """
        logger.info(f"Loading haiku from Hugging Face: {self.DATASET_PATH}")

        # Determine if we should use streaming mode
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42, buffer_size=10000).take(max_docs)
        else:
            dataset = load_dataset(self.DATASET_PATH, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        documents = []
        for i, example in enumerate(dataset):
            # Extract the haiku text from the dataset
            # Adjust field name based on actual dataset structure
            haiku_text = example.get("text") or example.get("haiku") or str(example)
            documents.append(haiku_text.strip())

            if not use_streaming and max_docs is not None and i + 1 >= max_docs:
                break

        logger.info(f"Loaded {len(documents)} haiku")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        return document if isinstance(document, str) else ""
