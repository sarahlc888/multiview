"""ArXiv CS papers document set loader.

Loads CS paper abstracts from HuggingFace (mteb/ArxivClassification) for
classification by paper type/methodology.
"""

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import ARXIV_CS_CRITERIA

logger = logging.getLogger(__name__)


class ArxivCSDocSet(BaseDocSet):
    """ArXiv CS papers document set.

    Loads CS paper abstracts from the mteb/ArxivClassification dataset.

    Config parameters:
        max_docs (int, optional): Maximum number of documents to load
        split (str): Dataset split to use (default: "train")

    Implementation notes:
        - Uses streaming mode when max_docs < 100 for efficiency
        - Contains full paper text/abstracts (can be quite long)
    """

    # Metadata
    DATASET_PATH = "mteb/ArxivClassification"
    DESCRIPTION = "ArXiv CS paper abstracts for classification by paper type"

    # Known criteria (only deterministic ones)
    KNOWN_CRITERIA = []  # word_count auto-included by base class

    # Metadata for LM-based criteria
    CRITERION_METADATA = ARXIV_CS_CRITERIA

    def load_documents(self) -> list[Any]:
        """Load ArXiv papers from HuggingFace.

        Returns:
            List of paper abstract strings
        """
        logger.info(f"Loading ArXiv CS papers from HuggingFace: {self.DATASET_PATH}")

        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")
        use_streaming = max_docs is not None and max_docs < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_docs={max_docs} < 100)")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42)
        else:
            dataset = load_dataset(self.DATASET_PATH, split=split)
            if max_docs is not None:
                dataset = dataset.shuffle(seed=42)

        # Load documents
        documents = []
        for example in dataset:
            text = example.get("text", "")
            if text:
                documents.append(text)

            # Respect max_docs
            if max_docs is not None and len(documents) >= max_docs:
                break

        logger.debug(f"Loaded {len(documents)} ArXiv CS papers")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A single document (text string)

        Returns:
            The text content of the document
        """
        return document if isinstance(document, str) else ""
