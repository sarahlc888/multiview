"""ArXiv CS papers document set loader.

Loads CS paper abstracts from HuggingFace (librarian-bots/arxiv-metadata-snapshot)
filtered by cs.ai category.
"""

import logging
from typing import Any

from multiview.docsets.arxiv_utils import ARXIV_DATASET_PATH, load_arxiv_abstracts
from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import ARXIV_CS_CRITERIA

logger = logging.getLogger(__name__)


class ArxivCSDocSet(BaseDocSet):
    """ArXiv CS papers document set.

    Loads CS paper abstracts from the librarian-bots/arxiv-metadata-snapshot dataset,
    filtered by cs.ai category.

    Config parameters:
        max_docs (int, optional): Maximum number of documents to load
        split (str): Dataset split to use (default: "train")
    """

    # Metadata
    DATASET_PATH = ARXIV_DATASET_PATH
    DESCRIPTION = "ArXiv CS AI paper abstracts filtered by cs.ai category"
    DOCUMENT_TYPE = "Abstract for a computer science research paper"

    # Known criteria (only deterministic ones)
    KNOWN_CRITERIA = []  # word_count auto-included by base class

    # Metadata for LM-based criteria
    CRITERION_METADATA = ARXIV_CS_CRITERIA

    def load_documents(self) -> list[Any]:
        """Load ArXiv papers from HuggingFace.

        Returns:
            List of paper abstract strings
        """
        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "train")

        # Load abstracts using shared utility
        documents = []
        for example in load_arxiv_abstracts(
            category_filter="cs.AI",
            max_abstracts=max_docs,
            split=split,
            seed=42,
        ):
            # Extract abstract text
            abstract = example.get("abstract", "")
            if abstract:
                documents.append(abstract)

        logger.info(f"Loaded {len(documents)} ArXiv CS AI papers")
        return self._deduplicate(documents)

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A single document (text string)

        Returns:
            The text content of the document
        """
        return document if isinstance(document, str) else ""
