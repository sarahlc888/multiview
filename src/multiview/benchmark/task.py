"""Task representation for benchmarking."""

import logging

from multiview.benchmark.annotations import (
    annotate_with_known_criterion,
    annotate_with_lm_all,
)
from multiview.benchmark.triplets.quality_assurance import (
    filter_triplets_by_quality,
    rate_triplet_quality,
)
from multiview.benchmark.triplets.triplet_utils import (
    create_lm_triplets,
    create_random_triplets,
)
from multiview.benchmark.triplets.utils import build_triplet_dicts
from multiview.docsets import DOCSETS

logger = logging.getLogger(__name__)

TRIPLET_STYLE_RANDOM = "random"
TRIPLET_STYLE_LM = "lm"
TRIPLET_STYLE_LM_ALL = "lm_all"

LM_TRIPLET_STYLES = {TRIPLET_STYLE_LM, TRIPLET_STYLE_LM_ALL}
RICH_ANNOTATION_STYLES = {TRIPLET_STYLE_LM_ALL}


class Task:
    """A benchmark task with a specific document_set and criterion.

    Each task loads a document_set and creates triplets based on a criterion.
    """

    def __init__(self, config: dict):
        """Initialize task from config.

        Required keys:
        - document_set: Dataset name (e.g., "gsm8k")
        - criterion: Criterion name (e.g., "arithmetic")

        Common optional keys:
        - triplet_style: "random" | "lm" | "lm_all"
        - max_docs, max_triplets, criterion_description
        - candidate_strategy, use_spurious_hard_negs, embedding_preset, lm_judge_preset
        - lm_all annotation hints: n_schema_samples, *_schema_hint, summary_*_hint
        """
        self.config = config

        # Extract required params
        self.document_set_name = config["document_set"]
        self.criterion_name = config["criterion"]

        # Extract optional params with defaults
        self.max_docs = config.get("max_docs")
        self.max_triplets = config.get("max_triplets")
        self.triplet_style = config.get("triplet_style", TRIPLET_STYLE_LM)
        self.add_synthetic_docs = config.get("add_synthetic_docs", False)
        # Synthetic doc synthesis configuration:
        self.num_synthetic_docs = config.get("num_synthetic_docs", 0)

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
        self.triplets = None  # List of (anchor_id, positive_id, negative_id) tuples
        self.triplet_quality_ratings = (
            None  # List of quality ratings (1-4) for each triplet
        )
        self.synthesis_anchor_indices = (
            None  # Indices of docs used as synthesis anchors (backward compat)
        )
        self.synthesis_metadata = None  # Full synthesis metadata dict

        # Warn if criterion is provided but meaningless
        if self.triplet_style == TRIPLET_STYLE_RANDOM:
            logger.warning(
                f"criterion '{self.criterion_name}' is required but meaningless "
                f"for triplet_style='random'"
            )

    def _require_documents(self, *, caller: str) -> None:
        if self.documents is None:
            raise RuntimeError(f"Must call load_documents() before {caller}()")

    def _require_triplets(self, *, caller: str) -> None:
        if self.triplets is None:
            raise RuntimeError(f"Must call create_triplets() before {caller}()")

    def _criterion_metadata(self) -> dict:
        return self.document_set.get_criterion_metadata(self.criterion_name) or {}

    def _resolved_criterion_hints(self) -> dict:
        meta = self._criterion_metadata()
        return {
            "criterion_description": self.config.get("criterion_description")
            or meta.get("description"),
            "category_schema_hint": self.config.get("category_schema_hint")
            or meta.get("category_schema_hint"),
            "tag_schema_hint": self.config.get("tag_schema_hint")
            or meta.get("tag_schema_hint"),
            "summary_guidance_hint": self.config.get("summary_guidance_hint")
            or meta.get("summary_guidance_hint"),
            "summary_format_hint": self.config.get("summary_format_hint")
            or meta.get("summary_format_hint"),
        }

    def _resolved_criterion_description(self) -> str | None:
        return self._resolved_criterion_hints()["criterion_description"]

    def _triplet_dicts(self) -> list[dict]:
        return build_triplet_dicts(self.documents, self.triplets)

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
        from multiview.benchmark.synthesis import synthesis_utils

        # Resolve desired synthetic doc count (absolute)
        if self.num_synthetic_docs <= 0:
            logger.info("num_synthetic_docs <= 0; skipping synthetic generation")
            return

        synthetic_docs, synthesis_metadata = synthesis_utils.synthesize_documents(
            documents=self.documents,
            document_set=self.document_set,
            criterion_name=self.criterion_name,
            num_synthetic_docs=self.num_synthetic_docs,
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
        self._require_documents(caller="annotate_documents")

        logger.info(f"Annotating documents for criterion: {self.criterion_name}...")

        if self.criterion_name in self.document_set.KNOWN_CRITERIA:
            self.document_annotations = annotate_with_known_criterion(
                self.documents, self.document_set, self.criterion_name
            )
        elif self.triplet_style in RICH_ANNOTATION_STYLES:
            hints = self._resolved_criterion_hints()
            self.document_annotations = annotate_with_lm_all(
                documents=self.documents,
                criterion=self.criterion_name,
                criterion_description=hints["criterion_description"],
                n_schema_samples=self.config.get("n_schema_samples", 10),
                category_schema_hint=hints["category_schema_hint"],
                tag_schema_hint=hints["tag_schema_hint"],
                summary_guidance_hint=hints["summary_guidance_hint"],
                summary_format_hint=hints["summary_format_hint"],
                include_debug=self.config.get("include_annotation_debug", False),
                cache_alias_prefix=f"{self.get_task_name()}_annotation",
            )
        else:
            raise ValueError(
                f"Unknown criterion '{self.criterion_name}' for document_set "
                f"'{self.document_set_name}'. Simple LM annotation is not implemented. "
                f"Use triplet_style='{TRIPLET_STYLE_LM_ALL}' (rich annotation) or choose a "
                "known criterion."
            )

    def create_triplets(self):
        """Create triplets based on the triplet_style.

        Wrapper function for methods defined in triplet_utils."""
        self._require_documents(caller="create_triplets")

        logger.info(f"Creating triplets for {self.document_set_name}...")
        logger.info(f"Triplet style: {self.triplet_style}")

        if self.triplet_style == TRIPLET_STYLE_RANDOM:
            self.triplets = create_random_triplets(
                self.documents,
                max_triplets=self.max_triplets,
            )
        elif self.triplet_style in LM_TRIPLET_STYLES:
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
                criterion_description=self._resolved_criterion_description(),
                cache_alias_prefix=f"{self.get_task_name()}_triplets",
                triplet_example=self.config.get("triplet_example"),
                anchor_indices=self.synthesis_anchor_indices,
            )
        else:
            raise ValueError(f"Unknown triplet_style: {self.triplet_style}")

        logger.info(f"Created {len(self.triplets)} triplets")

    def rate_triplet_quality(
        self,
        lm_judge_preset: str = "lmjudge_quality_rating_gemini",
        min_quality: int | None = None,
    ) -> dict:
        """Rate the quality of triplets using an LM judge and optionally filter.

        This method rates each triplet on a 1-4 scale:
        1 = invalid, 2 = ambiguous, 3 = trivial, 4 = ideal

        Args:
            lm_judge_preset: Preset for quality rating LM judge
            min_quality: If provided, filter triplets to keep only those with
                quality >= min_quality (1-4). If None, no filtering is applied.

        Returns:
            Dict with quality rating statistics

        Example:
            >>> task.create_triplets()
            >>> stats = task.rate_triplet_quality(min_quality=3)
            >>> print(f"Kept {stats['n_kept']} ideal/trivial triplets")
        """
        self._require_documents(caller="rate_triplet_quality")
        self._require_triplets(caller="rate_triplet_quality")

        logger.info("Rating triplet quality...")

        triplet_dicts = self._triplet_dicts()

        criterion_description = self._resolved_criterion_description()

        has_annotations = self.document_annotations is not None
        if has_annotations and "with_annotation" not in lm_judge_preset:
            lm_judge_preset = "lmjudge_quality_rating_with_annotation_gemini"
            logger.info(f"Using annotation-aware preset: {lm_judge_preset}")

        cache_alias = f"{self.get_task_name()}_quality_rating"
        results = rate_triplet_quality(
            triplets=triplet_dicts,
            criterion=self.criterion_name,
            criterion_description=criterion_description,
            lm_judge_preset=lm_judge_preset,
            cache_alias=cache_alias,
            annotations=self.document_annotations if has_annotations else None,
        )

        self.triplet_quality_ratings = results["ratings"]

        if min_quality is not None:
            logger.info(f"Filtering triplets with quality >= {min_quality}")
            triplets_with_ratings = results["triplets_with_ratings"]
            filtered_triplets, filter_stats = filter_triplets_by_quality(
                triplets_with_ratings, min_quality=min_quality
            )

            self.triplets = [
                (t["anchor_id"], t["positive_id"], t["negative_id"])
                for t in filtered_triplets
            ]
            self.triplet_quality_ratings = [
                t["quality_rating"] for t in filtered_triplets
            ]

            logger.info(
                f"Filtered triplets: {len(self.triplets)} remaining "
                f"(removed {filter_stats['n_filtered']})"
            )

            return {**results, **filter_stats}

        return results

    def get_task_name(self) -> str:
        """Get the name of this task.

        Returns:
            Task name in format: {document_set}__{criterion}
        """
        return f"{self.document_set_name}__{self.criterion_name}"
