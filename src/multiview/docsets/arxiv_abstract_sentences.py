"""ArXiv CS abstract sentences document set loader.

Loads CS paper abstracts from HuggingFace and splits them into individual sentences.
Each sentence becomes its own document, with metadata tracking which abstract it came from.
"""

import logging
from typing import Any

from multiview.docsets.arxiv_utils import (
    ARXIV_DATASET_PATH,
    load_arxiv_abstracts,
    split_into_sentences,
)
from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class ArxivAbstractSentencesDocSet(BaseDocSet):
    """ArXiv CS abstract sentences document set.

    Loads CS paper abstracts from the librarian-bots/arxiv-metadata-snapshot dataset,
    filtered by cs.ai category, and splits each abstract into individual sentences.
    Each sentence becomes a separate document with metadata about its source abstract.

    Config parameters:
        max_docs (int, optional): Maximum number of documents (sentences) to load
        max_abstracts (int, optional): Maximum number of abstracts to process before splitting
        split (str): Dataset split to use (default: "train")

    Note: If max_docs is specified, it applies to the total number of sentences.
          If max_abstracts is specified, it limits the number of abstracts before splitting.
    """

    # Metadata
    DATASET_PATH = ARXIV_DATASET_PATH
    DESCRIPTION = "ArXiv CS AI paper abstracts split into individual sentences"
    DOCUMENT_TYPE = "sentence from an academic paper abstract"

    # Known criteria (prelabeled)
    KNOWN_CRITERIA = []  # source_abstract handled via precomputed annotations

    # Metadata for LM-based criteria
    DATASET_NAME = "arxiv_abstract_sentences"

    def __init__(self, config: dict | None = None):
        """Initialize ArXiv abstract sentences dataset.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        self.PRECOMPUTED_ANNOTATIONS = {}

    def load_documents(self) -> list[Any]:
        """Load ArXiv papers from HuggingFace and split into sentences.

        Returns:
            List of sentence strings (metadata stored in precomputed annotations)
        """
        # Get config params
        max_docs = self.config.get("max_docs")
        max_abstracts = self.config.get("max_abstracts")
        split = self.config.get("split", "train")

        # Load abstracts and split into sentences
        # Store metadata separately for precomputed annotations
        sentences_with_metadata = []
        abstracts_processed = 0

        for example in load_arxiv_abstracts(
            category_filter="cs.AI",
            max_abstracts=max_abstracts,
            split=split,
            seed=42,
        ):
            # Extract abstract
            abstract = example.get("abstract", "").strip()
            abstract_id = example.get("id", f"abstract_{abstracts_processed}")

            # Split abstract into sentences
            sentences = split_into_sentences(abstract)

            # Store sentence with metadata
            for sentence_idx, sentence in enumerate(sentences):
                metadata = {
                    "text": sentence,
                    "source_abstract": abstract_id,
                    "abstract_id": abstract_id,
                    "sentence_idx": sentence_idx,
                }
                sentences_with_metadata.append(metadata)

                # Check max_docs limit
                if max_docs is not None and len(sentences_with_metadata) >= max_docs:
                    break

            abstracts_processed += 1

            # Check max_docs limit after processing each abstract
            if max_docs is not None and len(sentences_with_metadata) >= max_docs:
                break

        logger.info(
            f"Loaded {len(sentences_with_metadata)} sentences from {abstracts_processed} ArXiv CS AI abstracts"
        )

        # Build precomputed annotations from metadata
        self._build_precomputed_annotations(sentences_with_metadata)

        # Return just the text strings
        return [item["text"] for item in sentences_with_metadata]

    def _build_precomputed_annotations(self, documents: list[dict]) -> None:
        """Build precomputed annotations mapping for source_abstract.

        Args:
            documents: List of document dicts with "text" and "source_abstract" fields
        """
        source_abstract_annotations = {}

        for doc in documents:
            text = doc["text"]
            source_abstract_annotations[text] = {"prelabel": doc["source_abstract"]}

        self.PRECOMPUTED_ANNOTATIONS["source_abstract"] = source_abstract_annotations

        logger.debug(
            f"Built precomputed annotations for {len(source_abstract_annotations)} sentences "
            f"(1 criterion: source_abstract)"
        )

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document string (sentence text)

        Returns:
            The text content of the document (single sentence)
        """
        return str(document)
