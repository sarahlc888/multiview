"""Task representation for benchmarking."""

import json
import logging
from pathlib import Path

from multiview.benchmark.annotation_utils import (
    annotate_with_known_criterion,
    annotate_with_lm,
)
from multiview.benchmark.document_sets import DOCSETS
from multiview.benchmark.triplets.triplet_utils import (
    create_lm_triplets,
    create_random_triplets,
)

logger = logging.getLogger(__name__)


class Task:
    """A benchmark task with a specific document_set and criterion.

    Each task loads a document_set and creates triplets based on a criterion.
    """

    def __init__(self, config: dict):
        """Initialize task from config.

        Args:
            config: Configuration dict with keys:
                - document_set: Dataset name (e.g., 'gsm8k', 'crossword_clues')
                - criterion: Criterion for triplet creation (e.g., 'arithmetic', 'clue_type')
                - max_docs: Maximum number of documents to load (default: None)
                - max_triplets: Maximum number of triplets to create (default: None)
                - triplet_style: How to sample triplets (default: 'lm')
                - Any other params are stored but may not be used yet
        """
        self.config = config

        # Extract required params
        self.document_set_name = config["document_set"]
        self.criterion_name = config["criterion"]

        # Extract optional params with defaults
        self.max_docs = config.get("max_docs")
        self.max_triplets = config.get("max_triplets")
        self.triplet_style = config.get("triplet_style", "lm")

        # Get the document_set class from registry
        if self.document_set_name not in DOCSETS:
            raise ValueError(
                f"Unknown document_set: {self.document_set_name}. "
                f"Available document_sets: {list(DOCSETS.keys())}"
            )

        document_set_cls = DOCSETS[self.document_set_name]
        # Pass split if provided in config, otherwise let dataset use its own default
        dataset_config = {"max_docs": self.max_docs}
        if "split" in config:
            dataset_config["split"] = config["split"]
        self.document_set = document_set_cls(config=dataset_config)

        self.documents = None
        self.document_annotations = None  # List of dicts with criterion values
        self.triplets = None

        # Warn if criterion is provided but meaningless
        if self.triplet_style == "random":
            logger.warning(
                f"criterion '{self.criterion_name}' is required but meaningless "
                f"for triplet_style='random'"
            )

    def load_documents(self):
        """Load documents from the document_set."""
        logger.info(f"Loading documents for {self.document_set_name}...")
        # Document sets now handle max_docs internally
        self.documents = self.document_set.load_documents()
        logger.debug(f"Loaded {len(self.documents)} documents")

    def annotate_documents(self):
        """Annotate documents with criterion values.

        Wrapper function for methods defined in annotation_utils.
        """
        if self.documents is None:
            raise RuntimeError("Must call load_documents() before annotate_documents()")

        logger.info(f"Annotating documents for criterion: {self.criterion_name}...")

        if self.criterion_name in self.document_set.KNOWN_CRITERIA:
            self.document_annotations = annotate_with_known_criterion(
                self.documents, self.document_set, self.criterion_name
            )
        else:
            self.document_annotations = annotate_with_lm(
                self.documents, self.criterion_name
            )

    def create_triplets(self):
        """Create triplets based on the triplet_style.

        Wrapper function for methods defined in triplet_utils."""
        if self.documents is None:
            raise RuntimeError("Must call load_documents() before create_triplets()")

        logger.info(f"Creating triplets for {self.document_set_name}...")
        logger.info(f"Triplet style: {self.triplet_style}")

        # Create triplets based on style
        if self.triplet_style == "random":
            self.triplets = create_random_triplets(
                self.documents,
                max_triplets=self.max_triplets,
            )
        elif self.triplet_style == "lm":
            self.triplets = create_lm_triplets(
                self.documents,
                max_triplets=self.max_triplets,
            )
        else:
            raise ValueError(f"Unknown triplet_style: {self.triplet_style}")

        logger.info(f"Created {len(self.triplets)} triplets")

    def get_task_name(self) -> str:
        """Get the name of this task.

        Returns:
            Task name in format: {document_set}__{criterion}
        """
        return f"{self.document_set_name}__{self.criterion_name}"

    def save_triplets(self, output_dir: str | Path) -> None:
        """Save triplets to a JSONL file.

        Args:
            output_dir: Output directory path (e.g., outputs/run_name/triplets)
        """
        if self.triplets is None:
            raise RuntimeError("Must call create_triplets() before save_triplets()")

        output_dir = Path(output_dir)
        task_name = self.get_task_name()
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save triplets as JSONL
        output_file = task_dir / "triplets.jsonl"
        with open(output_file, "w") as f:
            for i, (anchor, positive, negative) in enumerate(self.triplets):
                triplet_data = {
                    "triplet_id": i,
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                }
                f.write(json.dumps(triplet_data) + "\n")

        logger.info(f"Saved {len(self.triplets)} triplets to {output_file}")

    def save_doc_annotations(self, output_dir: str | Path) -> None:
        raise NotImplementedError("Saving document annotations not implemented yet")
