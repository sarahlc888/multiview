"""Task representation for benchmarking."""

import json
import logging
from pathlib import Path

from multiview.benchmark.annotations import (
    annotate_with_known_criterion,
    annotate_with_lm,
    annotate_with_lm_all,
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
            config: Configuration dict with the following keys:

                Core Configuration:
                - document_set (str, required): Dataset name (e.g., 'gsm8k', 'crossword_clues')
                - criterion (str, required): Criterion for triplet creation (e.g., 'arithmetic', 'clue_type')
                - criterion_description (str, optional): Description of what the criterion means
                - max_docs (int, optional): Maximum number of documents to load
                - max_triplets (int, optional): Maximum number of triplets to create

                Annotation Configuration (for triplet_style="lm_all"):
                - n_schema_samples (int, default=10): Documents to sample for schema generation
                - category_schema_hint (str, optional): Hint for category schema generation
                - tag_schema_hint (str, optional): Hint for tag schema generation
                - summary_guidance_hint (str, optional): Hint for what to include in summaries
                - summary_format_hint (str, optional): Hint for summary format/structure
                - include_annotation_debug (bool, default=False): Include debug/reasoning in annotations

                Triplet Configuration:
                - triplet_style (str, default="lm"): "random", "lm", or "lm_all"
                  * "random": Random triplet sampling
                  * "lm": LM-based triplet selection with candidate filtering
                  * "lm_all": Uses rich multi-faceted annotations (categories + tags + summaries)
                - candidate_strategy (str, default="multi"): "bm25", "embedding", "jaccard", or "multi"
                - use_spurious_hard_negs (bool, default=True): Include spurious hard negatives
                - embedding_preset (str, default="hf_qwen3_embedding_8b"): Embedding model for candidates
                - lm_judge_preset (str, optional): LM judge preset for triplet selection
                - triplet_example (dict/str, optional): Example triplet for LM judge guidance
        """
        self.config = config

        # Extract required params
        self.document_set_name = config["document_set"]
        self.criterion_name = config["criterion"]

        # Extract optional params with defaults
        self.max_docs = config.get("max_docs")
        self.max_triplets = config.get("max_triplets")
        self.triplet_style = config.get("triplet_style", "lm")
        self.add_synthetic_docs = config.get("add_synthetic_docs", False)
        self.num_synthetic_per_doc = config.get("num_synthetic_per_doc", 2)

        # Get the document_set class from registry
        if self.document_set_name not in DOCSETS:
            raise ValueError(
                f"Unknown document_set: {self.document_set_name}. "
                f"Available document_sets: {list(DOCSETS.keys())}"
            )

        document_set_cls = DOCSETS[self.document_set_name]
        self.document_set = document_set_cls(
            config={
                "max_docs": self.max_docs,
                "split": "train",  # Could be configurable later
            }
        )

        self.documents = None
        self.document_annotations = None  # List of dicts with criterion values
        self.triplets = None  # List of (anchor_id, positive_id, negative_id) tuples
        self.synthesis_anchor_indices = (
            None  # Indices of docs used as synthesis anchors (backward compat)
        )
        self.synthesis_metadata = None  # Full synthesis metadata dict

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

    def augment_with_synthetic_documents(self):
        """Generate synthetic documents using LM-based synthesis."""
        if not self.add_synthetic_docs:
            return

        if self.documents is None:
            raise RuntimeError(
                "Must call load_documents() before augment_with_synthetic_documents()"
            )

        logger.info("Generating synthetic documents...")

        # Import here to avoid circular dependency
        from multiview.benchmark import synthesis_utils

        synthetic_docs, synthesis_metadata = synthesis_utils.synthesize_documents(
            documents=self.documents,
            document_set=self.document_set,
            criterion_name=self.criterion_name,
            num_synthetic_per_doc=self.num_synthetic_per_doc,
        )

        if synthetic_docs:
            logger.info(f"Added {len(synthetic_docs)} synthetic documents")
            self.documents.extend(synthetic_docs)
            self.synthesis_metadata = synthesis_metadata
            # Backward compatibility: extract anchor_indices from metadata
            self.synthesis_anchor_indices = synthesis_metadata.get("anchor_indices", [])
            logger.info(
                f"Stored synthesis metadata with {len(self.synthesis_anchor_indices)} anchor indices"
            )
        else:
            logger.info("No synthetic documents generated")

    def annotate_documents(self):
        """Annotate documents with criterion values.

        Wrapper function for methods defined in annotation_utils.
        """
        if self.documents is None:
            raise RuntimeError("Must call load_documents() before annotate_documents()")

        logger.info(f"Annotating documents for criterion: {self.criterion_name}...")

        # Infer annotation mode from triplet_style
        # lm_all → use union_all.py (categories + tags + summaries)
        # random / lm → use simple annotation
        needs_rich_annotation = self.triplet_style in ["lm_all", "lm_multifaceted"]

        if self.criterion_name in self.document_set.KNOWN_CRITERIA:
            # Known criterion - deterministic extraction
            self.document_annotations = annotate_with_known_criterion(
                self.documents, self.document_set, self.criterion_name
            )
        elif needs_rich_annotation:
            # Get criterion metadata from document_set (if available)
            criterion_metadata = self.document_set.get_criterion_metadata(
                self.criterion_name
            )

            # Use config values if provided, otherwise fall back to metadata
            criterion_description = self.config.get(
                "criterion_description"
            ) or criterion_metadata.get("description")
            category_schema_hint = self.config.get(
                "category_schema_hint"
            ) or criterion_metadata.get("category_schema_hint")
            tag_schema_hint = self.config.get(
                "tag_schema_hint"
            ) or criterion_metadata.get("tag_schema_hint")
            summary_guidance_hint = self.config.get(
                "summary_guidance_hint"
            ) or criterion_metadata.get("summary_guidance_hint")
            summary_format_hint = self.config.get(
                "summary_format_hint"
            ) or criterion_metadata.get("summary_format_hint")

            # Rich "all" LM annotation (union_all.py - combines categories, tags, summaries)
            self.document_annotations = annotate_with_lm_all(
                documents=self.documents,
                criterion=self.criterion_name,
                criterion_description=criterion_description,
                n_schema_samples=self.config.get("n_schema_samples", 10),
                category_schema_hint=category_schema_hint,
                tag_schema_hint=tag_schema_hint,
                summary_guidance_hint=summary_guidance_hint,
                summary_format_hint=summary_format_hint,
                include_debug=self.config.get("include_annotation_debug", False),
                cache_alias_prefix=f"{self.get_task_name()}_annotation",
            )
        else:
            # Simple LM annotation (existing)
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
        elif self.triplet_style in ["lm", "lm_all", "lm_multifaceted"]:
            # LM-based with candidate selection (lm_all uses rich annotations)
            self.triplets = create_lm_triplets(
                documents=self.documents,
                annotations=self.document_annotations,
                max_triplets=self.max_triplets,
                candidate_strategy=self.config.get("candidate_strategy", "multi"),
                use_spurious_hard_negs=self.config.get("use_spurious_hard_negs", True),
                embedding_preset=self.config.get(
                    "embedding_preset", "hf_qwen3_embedding_8b"
                ),
                lm_judge_preset=self.config.get(
                    "lm_judge_preset", "triplet_selection_gemini"
                ),
                criterion=self.criterion_name,
                criterion_description=self.config.get("criterion_description"),
                cache_alias_prefix=f"{self.get_task_name()}_triplets",
                triplet_example=self.config.get("triplet_example"),
                anchor_indices=self.synthesis_anchor_indices,
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

        # Save triplets as JSONL (document IDs, not text)
        output_file = task_dir / "triplets.jsonl"
        with open(output_file, "w") as f:
            for i, (anchor_id, positive_id, negative_id) in enumerate(self.triplets):
                triplet_data = {
                    "triplet_id": i,
                    "anchor_id": anchor_id,
                    "positive_id": positive_id,
                    "negative_id": negative_id,
                }
                f.write(json.dumps(triplet_data) + "\n")

        logger.info(f"Saved {len(self.triplets)} triplets to {output_file}")

    def save_doc_annotations(self, output_dir: str | Path) -> None:
        """Save document annotations to a JSONL file.

        Args:
            output_dir: Output directory path (e.g., outputs/run_name/annotations)
        """
        if self.document_annotations is None:
            raise RuntimeError(
                "Must call annotate_documents() before save_doc_annotations()"
            )

        output_dir = Path(output_dir)
        task_name = self.get_task_name()
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # Save annotations as JSONL
        output_file = task_dir / "annotations.jsonl"
        with open(output_file, "w") as f:
            for i, (doc, annotation) in enumerate(
                zip(self.documents, self.document_annotations, strict=False)
            ):
                annotation_data = {
                    "doc_id": i,
                    "document": doc,
                    **annotation,
                }
                f.write(json.dumps(annotation_data) + "\n")

        logger.info(
            f"Saved {len(self.document_annotations)} annotations to {output_file}"
        )

    def validate_synthetic_annotations(self, output_dir: str | Path) -> dict:
        """Validate synthetic document annotations against source documents.

        Computes Jaccard similarities for tags and spurious_tags between
        synthetic documents and their anchor documents. Generates validation
        reports showing quality of synthesis.

        Args:
            output_dir: Output directory path (e.g., outputs/run_name/validation)

        Returns:
            Dict with validation statistics
        """
        if not self.synthesis_metadata or not self.document_annotations:
            logger.warning("No synthesis metadata or annotations available for validation")
            return {}

        from multiview.benchmark.validation import validate_synthesis

        return validate_synthesis(
            documents=self.documents,
            annotations=self.document_annotations,
            synthesis_metadata=self.synthesis_metadata,
            output_dir=output_dir,
            task_name=self.get_task_name(),
        )
