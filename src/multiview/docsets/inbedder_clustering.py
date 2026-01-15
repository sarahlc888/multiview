"""Base class for InBedder clustering datasets.

Provides unified support for InBedder benchmark datasets that use the clustering format:
- text: The document text
- cluster: The cluster label/category
- split: The dataset split

Subclasses only need to specify dataset path, description, and criterion metadata.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class InBedderClusteringDocSet(BaseDocSet):
    """Base class for InBedder clustering datasets.

    Datasets that use this format:
    - NYTClustering (topic/location subsets)
    - RateMyProfClustering
    - FeedbacksClustering
    - FewRelClustering
    - FewNerdClustering
    - FewEventClustering

    Each dataset has text+cluster format where cluster is the criterion value.

    Config parameters:
        subset (str, optional): Dataset subset/configuration
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")

    Subclasses must define:
        DATASET_PATH (str): HuggingFace dataset path
        DESCRIPTION (str): Dataset description
        KNOWN_CRITERIA (list[str]): List of criterion names
        CRITERION_METADATA (dict): Metadata for LM-based criteria
        SUBSETS (list[str], optional): Valid subset names (if applicable)
    """

    # Subclasses must override these
    DATASET_PATH: str = ""
    DESCRIPTION: str = ""
    KNOWN_CRITERIA: list[str] = []
    CRITERION_METADATA: dict = {}
    SUBSETS: list[str] | None = None  # None means no subsets

    def __init__(self, config: dict | None = None):
        """Initialize InBedderClusteringDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}

    def load_documents(self) -> list[Any]:
        """Load clustering dataset from HuggingFace.

        Loads documents with cluster labels for prelabeled triplets.

        Returns:
            List of document dicts: {"text": str, "{criterion}": str}
        """
        # Get config params
        subset = self.config.get("subset")
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "test")

        # Validate subset if applicable
        if self.SUBSETS is not None:
            if not subset:
                raise ValueError(
                    f"subset is required for {self.__class__.__name__}. "
                    f"Valid options: {', '.join(self.SUBSETS)}. "
                    f"Set via config: {{'subset': '{self.SUBSETS[0]}'}}"
                )
            if subset not in self.SUBSETS:
                raise ValueError(
                    f"Invalid subset '{subset}' for {self.__class__.__name__}. "
                    f"Valid options: {', '.join(self.SUBSETS)}"
                )

        logger.info(
            f"Loading {self.__class__.__name__} from HuggingFace: {self.DATASET_PATH} "
            f"(subset={subset}, split={split})"
        )

        # Use streaming for large datasets or when max_docs is small
        use_streaming = max_docs is not None and max_docs < 1000

        if use_streaming:
            logger.debug(f"Using streaming mode with max_docs={max_docs}")
            if subset:
                dataset = load_dataset(
                    self.DATASET_PATH, subset, split=split, streaming=True
                )
            else:
                dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
            dataset = dataset.shuffle(seed=42)
        else:
            logger.debug(f"Loading full dataset split: {split}")
            if subset:
                dataset = load_dataset(self.DATASET_PATH, subset, split=split)
            else:
                dataset = load_dataset(self.DATASET_PATH, split=split)
            dataset = dataset.shuffle(seed=42)

        # Determine which criterion we're using
        criterion = self._get_criterion_from_subset(subset)

        # Extract documents
        documents = []
        doc_count = 0

        for idx, example in enumerate(dataset):
            try:
                # Extract fields
                text = example.get("text", "").strip()
                cluster = example.get("cluster", "").strip()

                # Skip if invalid
                if not text or not cluster:
                    logger.debug(f"Skipping row {idx}: missing text or cluster")
                    continue

                # Add document with cluster label
                documents.append(
                    {
                        "text": text,
                        criterion: cluster,
                    }
                )
                doc_count += 1

                # Check max_docs limit
                if max_docs and doc_count >= max_docs:
                    logger.debug(f"Reached max_docs limit: {max_docs}")
                    break

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed row at index {idx}: {e}")
                continue

        logger.info(
            f"Loaded {len(documents)} documents " f"(subset={subset}, split={split})"
        )

        # Build precomputed annotations
        self._build_precomputed_annotations(documents, criterion)

        return documents

    def _get_criterion_from_subset(self, subset: str | None) -> str:
        """Get the criterion name from the subset.

        Default implementation: subset name is the criterion name,
        or the first known criterion if no subset.

        Subclasses can override for custom logic.

        Args:
            subset: The dataset subset

        Returns:
            The criterion name
        """
        if subset:
            return subset
        if self.KNOWN_CRITERIA:
            return self.KNOWN_CRITERIA[0]
        raise ValueError("No criterion available")

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document (dict or string)

        Returns:
            Text content
        """
        if isinstance(document, dict):
            return document.get("text", "")
        return str(document) if document else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - Any criterion in KNOWN_CRITERIA: the cluster label

        Args:
            document: A document
            criterion: The criterion name

        Returns:
            Criterion value or None
        """
        if criterion in self.KNOWN_CRITERIA:
            if isinstance(document, dict):
                return document.get(criterion)
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations(
        self, documents: list[dict], criterion: str
    ) -> None:
        """Build precomputed annotations from loaded documents.

        Creates a mapping: {document_text: {"criterion_value": cluster_label}}

        Args:
            documents: List of document dicts with 'text' and criterion fields
            criterion: The criterion name
        """
        annotations = {}
        cluster_counts = {}

        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text")
                label = doc.get(criterion)

                if text and label:
                    annotations[text] = {"criterion_value": label}

                    # Track cluster distribution
                    cluster_counts[label] = cluster_counts.get(label, 0) + 1

        self.PRECOMPUTED_ANNOTATIONS[criterion] = annotations

        logger.info(
            f"Built precomputed annotations for {criterion}: "
            f"{len(annotations)} documents across {len(cluster_counts)} clusters"
        )
        logger.debug(f"Cluster distribution: {cluster_counts}")
