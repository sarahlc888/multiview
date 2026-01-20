"""Task representation for benchmarking.


--------------------------------------------------
Automatic semantic similarity evaluation pipeline
--------------------------------------------------
Given a set of documents and a criterion, create triplets (anchor, pos, neg) where anchor is more similar to pos than neg based on the criteria.

Overall workflow:
- Accept a corpus and a specific criteria/aspect as input
- Use a strong LM to generate some text elaborating on the criteria and how it relates to the corpus.
- Use these annotations to construct triplets

High quality data has 2 features

(1) Criteria should be unambiguous. Even a relatively weak generative LM like qwen 3 8b should be able to correctly identify the positive item >90% of the time. This is a non-negotiable requirement.
(2) Triplets should be non-trivial (hard negatives). Ideally, triplets should be constructed so that anchor and negative have some superficial similarities that a model must learn to be invariant to.

An additional consideration is scalability.
The pipeline should be designed so that task complexity is scalable with the strength of an LM judge.
We want to be able to scale task complexity simply by using a stronger LM judge.
"""

from __future__ import annotations

import logging

from multiview.benchmark.annotations import (
    annotate_with_known_criterion,
    annotate_with_lm_all,
    annotate_with_lm_category,
    annotate_with_lm_summary_dict,
    annotate_with_lm_summary_sentence,
    annotate_with_lm_tags,
    annotate_with_precomputed,
)
from multiview.benchmark.triplets.pairwise_similarity import (
    create_pairwise_similarity_triplets,
)
from multiview.benchmark.triplets.triplet_utils import (
    create_lm_triplets,
    create_lm_triplets_category,
    create_lm_triplets_summary_dict,
    create_lm_triplets_summary_sentence,
    create_lm_triplets_tags,
    create_prelabeled_triplets,
    create_random_triplets,
)
from multiview.benchmark.triplets.utils import build_triplet_dicts
from multiview.docsets import DOCSETS
from multiview.docsets.d5_applic import D5ApplicabilityDocSet
from multiview.docsets.d5_triplets import create_d5_applicability_triplets
from multiview.docsets.instructstsb import InstructSTSBDocSet
from multiview.docsets.intent_emotion import (
    IntentEmotionDocSet,
    create_intent_emotion_triplets,
)
from multiview.docsets.kgc_base import KGCBaseDocSet, create_kgc_triplets

logger = logging.getLogger(__name__)

TRIPLET_STYLE_RANDOM = "random"
TRIPLET_STYLE_PRELABELED = "prelabeled"
TRIPLET_STYLE_LM = "lm"
TRIPLET_STYLE_LM_ALL = "lm_all"
TRIPLET_STYLE_LM_CATEGORY = "lm_category"
TRIPLET_STYLE_LM_TAGS = "lm_tags"
TRIPLET_STYLE_LM_SUMMARY_DICT = "lm_summary_dict"
TRIPLET_STYLE_LM_SUMMARY_SENTENCE = "lm_summary_sentence"

LM_TRIPLET_STYLES = {
    TRIPLET_STYLE_LM,
    TRIPLET_STYLE_LM_ALL,
    TRIPLET_STYLE_LM_CATEGORY,
    TRIPLET_STYLE_LM_TAGS,
    TRIPLET_STYLE_LM_SUMMARY_DICT,
    TRIPLET_STYLE_LM_SUMMARY_SENTENCE,
}
RICH_ANNOTATION_STYLES = {
    TRIPLET_STYLE_LM_ALL,
    TRIPLET_STYLE_LM_CATEGORY,
    TRIPLET_STYLE_LM_TAGS,
    TRIPLET_STYLE_LM_SUMMARY_DICT,
    TRIPLET_STYLE_LM_SUMMARY_SENTENCE,
}

# Triplet style abbreviations for config suffix
TRIPLET_STYLE_ABBREV = {
    "random": "rnd",
    "prelabeled": "pre",
    "lm": "hn",  # "lm" is treated as hard_negative style
    "lm_all": "hn",
    "lm_category": "cat",
    "lm_tags": "tag",
    "lm_summary_dict": "sdict",
    "lm_summary_sentence": "ssent",
}


def _make_config_suffix(config: dict) -> str:
    """Create compact config suffix for task directory name.

    Includes triplet_style and max_triplets for easy identification.

    Args:
        config: Task config dict

    Returns:
        Suffix like "hn__300" or "rnd__500"
    """
    parts = []

    # Add triplet style abbreviation
    triplet_style = config.get("triplet_style", "lm")
    style_abbrev = TRIPLET_STYLE_ABBREV.get(triplet_style, triplet_style[:4])
    parts.append(style_abbrev)

    # Add max triplets
    max_triplets = config.get("max_triplets", 0)
    parts.append(str(max_triplets))

    return "__".join(parts)


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
        - run_name: Experiment/run name for cache organization
        - triplet_style: "random" | "lm" | "lm_all"
        - max_docs, max_triplets, criterion_description
        - candidate_strategy, use_spurious_hard_negs, embedding_preset, lm_judge_preset
        - lm_all annotation hints: n_schema_samples, *_schema_hint, summary_*_hint

        Quality rating overrides:
        - quality_rating_preset: Custom preset for rating without annotations
                                  (default: "lmjudge_quality_rating_gemini")
        - quality_rating_preset_with_annotations: Custom preset for rating with annotations
                                                   (default: "lmjudge_quality_rating_with_annotation_gemini")

        Caching configuration:
        - reuse_cached_triplets: Whether to reuse cached triplets (default: True)
                                 Set to False to always regenerate triplets
        - triplet_cache_dir: Directory to check for/save cached triplets
                            (default: outputs/{run_name} if run_name is set)
        - use_cache: Whether to use completion-level caching (default: True)
                    This is for individual LM calls, not triplet reuse
        """
        self.config = config

        # Extract required params
        self.document_set_name = config["document_set"]
        self.criterion_name = config["criterion"]

        # Extract optional params with defaults
        self.run_name = config.get("run_name")
        self.max_docs = config.get("max_docs")
        self.max_triplets = config.get("max_triplets")
        self.triplet_style = config.get("triplet_style", TRIPLET_STYLE_LM)
        # Synthetic doc synthesis configuration:
        self.num_synthetic_docs = config.get("num_synthetic_docs", 0)
        # Schema generation pool size - if not set, use all documents for schema generation
        # Setting this ensures schema generation is stable even when max_docs changes
        self.schema_pool_size = config.get("schema_pool_size")

        # Quality rating preset overrides (optional)
        self.quality_rating_preset = self.config.get("quality_rating_preset")
        self.quality_rating_preset_with_annotations = self.config.get(
            "quality_rating_preset_with_annotations"
        )

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
        # Merge in docset-specific config if provided (e.g., subset for nytclustering)
        if "config" in config:
            dataset_config.update(config["config"])
        self.document_set = document_set_cls(config=dataset_config)

        # Validate IntentEmotionDocSet criterion matches subset
        if isinstance(self.document_set, IntentEmotionDocSet):
            subset = dataset_config.get("subset")
            expected_criterion = (
                "intent_similarity" if subset == "intent" else "emotion_similarity"
            )
            if self.criterion_name != expected_criterion:
                raise ValueError(
                    f"IntentEmotionDocSet: criterion '{self.criterion_name}' doesn't match subset '{subset}'. "
                    f"Expected criterion '{expected_criterion}' for subset '{subset}'"
                )

        self.documents = (
            None  # Raw documents as returned by docset (may be dicts or strings)
        )
        self.document_annotations = None  # List of dicts with criterion values
        self.triplets = None  # List of (anchor_id, positive_id, negative_id) tuples
        self.triplet_quality_ratings = (
            None  # List of quality ratings (1-5) for each triplet
        )
        self.triplet_quality_ratings_with_annotations = None
        self.triplet_quality_ratings_without_annotations = None
        self.triplet_quality_reasoning = (
            None  # List of reasoning traces for quality ratings
        )
        self.triplet_quality_reasoning_with_annotations = None
        self.triplet_quality_reasoning_without_annotations = None
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

        # Resolve criterion metadata and hints (with config overrides)
        meta = self.document_set.get_criterion_metadata(self.criterion_name) or {}

        # Criterion description is REQUIRED - fail fast if missing
        self.criterion_description = self.config.get(
            "criterion_description"
        ) or meta.get("description")
        if not self.criterion_description:
            raise ValueError(
                f"Missing criterion description for '{self.criterion_name}' in document set '{self.document_set_name}'. "
                f"Please add a 'description' field for this criterion in configs/available_criteria.yaml "
                f"under the '{self.document_set_name}' section, or provide 'criterion_description' in the task config."
            )

        # Optional hints (can be None)
        # Resolution order: config override > specific hint > default_hint > None
        default_hint = meta.get("default_hint")
        self.category_schema_hint = (
            self.config.get("category_schema_hint")
            or meta.get("category_schema_hint")
            or default_hint
        )
        self.tag_schema_hint = (
            self.config.get("tag_schema_hint")
            or meta.get("tag_schema_hint")
            or default_hint
        )
        self.summary_hint = (
            self.config.get("summary_hint") or meta.get("summary_hint") or default_hint
        )
        self.triplet_example_hint = self.config.get("triplet_example_hint") or meta.get(
            "triplet_example_hint"
        )

    def load_documents(self):
        """Load documents from the document_set."""
        logger.info(f"Loading documents for {self.document_set_name}...")
        # Document sets now handle max_docs and deduplication internally
        # Store raw documents (may be dicts or strings depending on docset)
        self.documents = self.document_set.load_documents()
        logger.debug(f"Loaded {len(self.documents)} documents")

    def get_schema_pool(self) -> list:
        """Get the document pool to use for schema generation.

        Returns a subset of documents for schema generation. If schema_pool_size
        is set, returns the first N documents. Otherwise returns all documents.

        This ensures schema generation is stable across different max_docs values.
        """
        if self.documents is None:
            raise RuntimeError("Must call load_documents() before get_schema_pool()")

        if self.schema_pool_size is None:
            return self.documents

        pool_size = min(self.schema_pool_size, len(self.documents))
        logger.debug(
            f"Using schema pool of {pool_size} documents "
            f"(schema_pool_size={self.schema_pool_size}, total docs={len(self.documents)})"
        )
        return self.documents[:pool_size]

    def augment_with_synthetic_documents(self):
        """Generate synthetic documents using LM-based synthesis."""
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
            run_name=self.run_name,
        )

        if synthetic_docs:
            logger.info(f"Added {len(synthetic_docs)} synthetic documents")
            self.documents.extend(synthetic_docs)
            self.synthesis_metadata = synthesis_metadata
            # Use unique remix anchor indices
            self.synthesis_anchor_indices = synthesis_metadata.get(
                "remix_anchor_indices", []
            )
            logger.info(
                f"Stored synthesis metadata with {len(self.synthesis_anchor_indices)} remix anchor indices"
            )
        else:
            logger.info("No synthetic documents generated")

    def annotate_documents(self):
        """Annotate documents with criterion values.

        Wrapper function for methods defined in annotation_utils.
        """
        # Skip annotation for datasets with pairwise relationships (like InstructSTSB)
        if isinstance(self.document_set, InstructSTSBDocSet):
            logger.info(
                f"Skipping annotation for {self.document_set_name} (uses pairwise relationships)"
            )
            return

        logger.info(f"Annotating documents for criterion: {self.criterion_name}...")

        # Check for precomputed annotations first
        if self.document_set.has_precomputed_annotations(self.criterion_name):
            logger.info("Using precomputed annotations from dataset")
            self.document_annotations = annotate_with_precomputed(
                self.documents, self.document_set, self.criterion_name
            )
        elif self.criterion_name in self.document_set.KNOWN_CRITERIA:
            self.document_annotations = annotate_with_known_criterion(
                self.documents, self.document_set, self.criterion_name
            )
        elif self.triplet_style in RICH_ANNOTATION_STYLES:
            # Get schema pool for schema generation (if schema_pool_size is set)
            schema_pool = self.get_schema_pool()

            if self.triplet_style == TRIPLET_STYLE_LM_ALL:
                self.document_annotations = annotate_with_lm_all(
                    documents=self.documents,
                    criterion=self.criterion_name,
                    document_type=self.document_set.DOCUMENT_TYPE,
                    criterion_description=self.criterion_description,
                    n_schema_samples=self.config.get("n_schema_samples", 10),
                    category_schema_hint=self.category_schema_hint,
                    tag_schema_hint=self.tag_schema_hint,
                    summary_hint=self.summary_hint,
                    include_debug=self.config.get("include_annotation_debug", False),
                    cache_alias_prefix=f"{self.get_task_name()}_annotation",
                    run_name=self.run_name,
                    schema_documents=schema_pool,
                    category_schema_preset=self.config.get("category_schema_preset"),
                    category_classify_preset=self.config.get(
                        "category_classify_preset"
                    ),
                    tag_schema_preset=self.config.get("tag_schema_preset"),
                    spurious_tag_schema_preset=self.config.get(
                        "spurious_tag_schema_preset"
                    ),
                    tag_apply_preset=self.config.get("tag_apply_preset"),
                    summary_guidance_preset=self.config.get("summary_guidance_preset"),
                    summary_generate_preset=self.config.get("summary_generate_preset"),
                )
            elif self.triplet_style == TRIPLET_STYLE_LM_CATEGORY:
                self.document_annotations = annotate_with_lm_category(
                    documents=self.documents,
                    criterion=self.criterion_name,
                    criterion_description=self.criterion_description,
                    document_type=self.document_set.DOCUMENT_TYPE,
                    n_schema_samples=self.config.get("n_schema_samples", 10),
                    category_schema_hint=self.category_schema_hint,
                    cache_alias_prefix=f"{self.get_task_name()}_annotation",
                    run_name=self.run_name,
                    schema_documents=schema_pool,
                    category_schema_preset=self.config.get("category_schema_preset"),
                    category_classify_preset=self.config.get(
                        "category_classify_preset"
                    ),
                )
            elif self.triplet_style == TRIPLET_STYLE_LM_TAGS:
                self.document_annotations = annotate_with_lm_tags(
                    documents=self.documents,
                    criterion=self.criterion_name,
                    criterion_description=self.criterion_description,
                    document_type=self.document_set.DOCUMENT_TYPE,
                    n_schema_samples=self.config.get("n_schema_samples", 10),
                    tag_schema_hint=self.tag_schema_hint,
                    cache_alias_prefix=f"{self.get_task_name()}_annotation",
                    run_name=self.run_name,
                    schema_documents=schema_pool,
                    tag_schema_preset=self.config.get("tag_schema_preset"),
                    spurious_tag_schema_preset=self.config.get(
                        "spurious_tag_schema_preset"
                    ),
                    tag_apply_preset=self.config.get("tag_apply_preset"),
                )
            elif self.triplet_style == TRIPLET_STYLE_LM_SUMMARY_SENTENCE:
                self.document_annotations = annotate_with_lm_summary_sentence(
                    documents=self.documents,
                    criterion=self.criterion_name,
                    criterion_description=self.criterion_description,
                    document_type=self.document_set.DOCUMENT_TYPE,
                    n_schema_samples=self.config.get("n_schema_samples", 10),
                    summary_hint=self.summary_hint,
                    cache_alias_prefix=f"{self.get_task_name()}_annotation",
                    run_name=self.run_name,
                    schema_documents=schema_pool,
                    summary_guidance_preset=self.config.get("summary_guidance_preset"),
                    summary_generate_preset=self.config.get("summary_generate_preset"),
                )
            elif self.triplet_style == TRIPLET_STYLE_LM_SUMMARY_DICT:
                self.document_annotations = annotate_with_lm_summary_dict(
                    documents=self.documents,
                    criterion=self.criterion_name,
                    criterion_description=self.criterion_description,
                    document_type=self.document_set.DOCUMENT_TYPE,
                    n_schema_samples=self.config.get("n_schema_samples", 10),
                    summary_hint=self.summary_hint,
                    cache_alias_prefix=f"{self.get_task_name()}_annotation",
                    run_name=self.run_name,
                    schema_documents=schema_pool,
                    summary_guidance_preset=self.config.get("summary_guidance_preset"),
                    summary_generate_preset=self.config.get("summary_generate_preset"),
                )
        else:
            raise ValueError(
                f"Unknown criterion '{self.criterion_name}' for document_set "
                f"'{self.document_set_name}'. Simple LM annotation is not implemented. "
                f"Use triplet_style='{TRIPLET_STYLE_LM_ALL}' (rich annotation) or choose a "
                "known criterion."
            )

    def can_use_cached_triplets(self) -> bool:
        """Check if cached triplets are available without loading them.

        This is a lightweight check that validates cache exists and config matches.
        Use this after load_documents() to decide if annotation is needed.

        Returns:
            True if cached triplets can be used, False otherwise
        """
        from multiview.benchmark.artifacts import can_use_cached_triplets

        cache_dir = self._get_triplet_cache_dir()
        if not cache_dir:
            logger.debug("No cache directory configured - cannot use cached triplets")
            return False

        task_name = self.get_task_name()
        logger.debug(f"Checking for cached triplets in: {cache_dir}/{task_name}")

        result = can_use_cached_triplets(
            output_dir=cache_dir,
            task_name=task_name,
            current_config=self.config,
        )

        if result:
            logger.debug(f"Cached triplets found and config matches for {task_name}")
        else:
            logger.debug(f"Cached triplets not available for {task_name}")

        return result

    def try_load_cached_triplets(self, output_dir: str) -> bool:
        """Try to load cached triplets if they match the current config.

        Thin wrapper around artifacts.try_load_cached_triplets().

        Args:
            output_dir: Directory where cached triplets might be saved

        Returns:
            True if cached triplets were loaded successfully, False otherwise
        """
        from multiview.benchmark.artifacts import try_load_cached_triplets

        result = try_load_cached_triplets(
            output_dir=output_dir,
            task_name=self.get_task_name(),
            current_config=self.config,
        )

        if result is not None:
            self.triplets, self.triplet_quality_ratings = result
            return True
        return False

    def _needs_annotations_for_triplet_generation(self) -> bool:
        """Check if current triplet_style requires annotations.

        Returns:
            True if annotations are needed for triplet generation, False otherwise
        """
        # Random doesn't need annotations
        if self.triplet_style == TRIPLET_STYLE_RANDOM:
            return False

        # Prelabeled style: check document set type
        if self.triplet_style == TRIPLET_STYLE_PRELABELED:
            # KGC, IntentEmotion, InstructSTSB, and D5 Applicability have their own data/labels
            if isinstance(
                self.document_set,
                KGCBaseDocSet
                | IntentEmotionDocSet
                | InstructSTSBDocSet
                | D5ApplicabilityDocSet,
            ):
                return False
            # Other prelabeled styles need annotations
            return True

        # LM styles need annotations
        return True

    def _get_triplet_cache_dir(self) -> str | None:
        """Resolve the triplet cache directory from config.

        Returns:
            Cache directory path, or None if triplet cache reuse is disabled
        """
        # Check if triplet cache reuse is enabled (default: True)
        if not self.config.get("reuse_cached_triplets", True):
            return None

        cache_dir = self.config.get("triplet_cache_dir")
        if cache_dir is None and self.run_name:
            cache_dir = f"outputs/{self.run_name}"

        return cache_dir

    def create_triplets(self):
        """Create triplets based on the triplet_style.

        If reuse_cached_triplets is enabled (default), tries to load cached triplets first
        BEFORE requiring annotations. Only requires annotations if cache is unavailable
        and triplet generation needs them.

        Wrapper function that orchestrates caching and delegates to triplet_utils.
        """
        # Try loading from cache FIRST (before expensive annotation)
        cache_dir = self._get_triplet_cache_dir()
        if cache_dir:
            if self.try_load_cached_triplets(cache_dir):
                logger.info("=" * 60)
                logger.info("✓ REUSING CACHED TRIPLETS")
                logger.info("=" * 60)
                return
            logger.info("Cache not usable - generating new triplets")
        elif not self.config.get("reuse_cached_triplets", True):
            logger.info("Triplet cache reuse disabled (reuse_cached_triplets=False)")
        else:
            logger.info(
                "No cache directory configured (set run_name or triplet_cache_dir)"
            )

        # Cache failed - check if we need annotations for generation
        if self._needs_annotations_for_triplet_generation():
            if self.document_annotations is None:
                raise RuntimeError(
                    f"Annotations required for triplet_style='{self.triplet_style}'. "
                    f"Call annotate_documents() before create_triplets()"
                )

        logger.info("=" * 60)
        logger.info(f"GENERATING NEW TRIPLETS: {self.document_set_name}")
        logger.info(f"Triplet style: {self.triplet_style}")
        logger.info("=" * 60)

        # Calculate overshooting factor for quality filtering
        # If quality filtering is enabled, generate more triplets upfront to account for filtering
        rate_quality = self.config.get("rate_triplet_quality", False)
        min_quality = self.config.get("min_triplet_quality", None)

        if rate_quality and min_quality is not None:
            # Overshoot by 3x to ensure we have enough high-quality triplets
            # After filtering, we'll select the top max_triplets by rating
            overshooting_factor = 3.0
            effective_max_triplets = int(self.max_triplets * overshooting_factor)
            logger.info(
                f"Quality filtering enabled (min_quality={min_quality}): "
                f"Generating {effective_max_triplets} triplets (3x overshoot) "
                f"to target {self.max_triplets} after filtering"
            )
        else:
            effective_max_triplets = self.max_triplets

        if self.triplet_style == TRIPLET_STYLE_RANDOM:
            self.triplets = create_random_triplets(
                self.documents,
                max_triplets=effective_max_triplets,
            )
        elif self.triplet_style == TRIPLET_STYLE_PRELABELED:
            # Check document set type to determine triplet generation strategy
            if isinstance(self.document_set, KGCBaseDocSet):
                # KGC triplets: all three share same relation type
                self.triplets = create_kgc_triplets(
                    documents=self.documents,
                    metadata_lookup=self.document_set._triplet_metadata,
                    max_triplets=effective_max_triplets,
                    seed=self.config.get("seed", 42),
                )
            elif isinstance(self.document_set, IntentEmotionDocSet):
                # IntentEmotion: use pre-made triplets from dataset
                self.triplets = create_intent_emotion_triplets(
                    documents=self.documents,
                    metadata_lookup=self.document_set._triplet_metadata,
                    max_triplets=effective_max_triplets,
                    seed=self.config.get("seed", 42),
                )
            elif isinstance(self.document_set, InstructSTSBDocSet):
                # InstructSTSB - use pairwise similarity relationships
                logger.info(
                    "Using pairwise similarity triplet creation for InstructSTSB"
                )
                self.triplets = create_pairwise_similarity_triplets(
                    documents=self.documents,
                    docset=self.document_set,
                    criterion=self.criterion_name,
                    max_triplets=effective_max_triplets,
                    seed=self.config.get("seed", 42),
                )
            elif isinstance(self.document_set, D5ApplicabilityDocSet):
                # D5 Applicability - use custom property-text matching
                logger.info(
                    "Using D5 applicability property-text matching triplet creation"
                )
                self.triplets = create_d5_applicability_triplets(
                    documents=self.documents,
                    docset=self.document_set,
                    max_triplets=effective_max_triplets,
                    selection_strategy=self.config.get(
                        "prelabeled_selection", "hard_negatives"
                    ),
                    seed=self.config.get("seed", 42),
                )
            else:
                # Standard prelabeled triplets using annotations
                # Pass metadata_lookup if available (for is_anchor markers)
                metadata_lookup = getattr(self.document_set, "_triplet_metadata", None)
                self.triplets = create_prelabeled_triplets(
                    documents=self.documents,
                    annotations=self.document_annotations,
                    max_triplets=effective_max_triplets,
                    selection_strategy=self.config.get(
                        "prelabeled_selection", "hard_negatives"
                    ),
                    seed=self.config.get("seed", 42),
                    metadata_lookup=metadata_lookup,
                )
        elif self.triplet_style in LM_TRIPLET_STYLES:
            # Get embedding preset overrides from criterion metadata (e.g., embed_instr)
            embedding_preset_name = self.config.get(
                "embedding_preset", "hf_qwen3_embedding_8b"
            )

            criterion_metadata = (
                self.document_set.get_criterion_metadata(self.criterion_name) or {}
            )
            embedding_preset_overrides = None
            if "embed_instr" in criterion_metadata:
                # For symmetric retrieval (summary-to-summary), use instruction template
                embedding_preset_overrides = {
                    "instruction": criterion_metadata["embed_instr"],
                }

            # Common parameters for all LM triplet styles
            common_params = {
                "documents": self.documents,
                "annotations": self.document_annotations,
                "max_triplets": effective_max_triplets,
                "lm_judge_preset": self.config.get(
                    "lm_judge_preset", "triplet_select_positive_gemini"
                ),
                "lm_judge_preset_negative": self.config.get(
                    "lm_judge_preset_negative", "triplet_select_negative_gemini"
                ),
                "criterion": self.criterion_name,
                "criterion_description": self.criterion_description,
                "cache_alias_prefix": f"{self.get_task_name()}_triplets",
                "triplet_example_hint": self.triplet_example_hint,
                "anchor_indices": self.synthesis_anchor_indices,
                "max_num_candidates": self.config.get("max_num_candidates", 10),
                "run_name": self.run_name,
            }

            if (
                self.triplet_style == TRIPLET_STYLE_LM_ALL
                or self.triplet_style == TRIPLET_STYLE_LM
            ):
                self.triplets = create_lm_triplets(
                    candidate_strategy=self.config.get("candidate_strategy", "multi"),
                    use_spurious_hard_negs=self.config.get(
                        "use_spurious_hard_negs", True
                    ),
                    embedding_preset=embedding_preset_name,
                    embedding_preset_overrides=embedding_preset_overrides,
                    **common_params,
                )
            elif self.triplet_style == TRIPLET_STYLE_LM_CATEGORY:
                self.triplets = create_lm_triplets_category(
                    use_bm25_heuristic=self.config.get(
                        "use_bm25_heuristic", True
                    ),  # Changed default to True
                    **common_params,
                )
            elif self.triplet_style == TRIPLET_STYLE_LM_TAGS:
                self.triplets = create_lm_triplets_tags(**common_params)
            elif self.triplet_style == TRIPLET_STYLE_LM_SUMMARY_SENTENCE:
                self.triplets = create_lm_triplets_summary_sentence(
                    embedding_preset=embedding_preset_name,
                    embedding_preset_overrides=embedding_preset_overrides,
                    **common_params,
                )
            elif self.triplet_style == TRIPLET_STYLE_LM_SUMMARY_DICT:
                self.triplets = create_lm_triplets_summary_dict(
                    embedding_preset=embedding_preset_name,
                    embedding_preset_overrides=embedding_preset_overrides,
                    **common_params,
                )
        else:
            raise ValueError(f"Unknown triplet_style: {self.triplet_style}")

        logger.info(f"Created {len(self.triplets)} triplets")

    def rate_and_filter_quality(
        self,
        min_quality: int | None = None,
        output_dir: str | None = None,
    ) -> dict:
        """Rate triplet quality, optionally compare with/without annotations, filter, and save drops."""
        from multiview.benchmark.triplets.quality_assurance import (
            rate_and_filter_quality_workflow,
        )

        # Run quality workflow (handles rating, comparison, filtering, consistency, and top-N selection)
        # Extract text from documents (handle both dict and string formats)
        document_texts = [
            self.document_set.get_document_text(doc) for doc in self.documents
        ]
        result = rate_and_filter_quality_workflow(
            triplets=build_triplet_dicts(document_texts, self.triplets),
            criterion=self.criterion_name,
            criterion_description=self.criterion_description,
            annotations=self.document_annotations,
            min_quality=min_quality,
            max_triplets=self.max_triplets if min_quality is not None else None,
            cache_alias=f"{self.get_task_name()}_quality_rating",
            run_name=self.run_name,
            quality_rating_preset=self.quality_rating_preset,
            quality_rating_preset_with_annotations=self.quality_rating_preset_with_annotations,
            validate_consistency=self.config.get("validate_consistency", True),
            consistency_min_quality=self.config.get("consistency_min_quality", 3),
            consistency_max_invalid=self.config.get("consistency_max_invalid", 1),
        )

        # Update task state
        self.triplets = [
            (t["anchor_id"], t["positive_id"], t["negative_id"])
            for t in result["kept_triplets"]
        ]
        self.triplet_quality_ratings = result["ratings"]
        self.triplet_quality_reasoning = result["reasoning"]

        # Store comparison results if available
        if "ratings_without_annotations" in result:
            self.triplet_quality_ratings_without_annotations = result[
                "ratings_without_annotations"
            ]
            self.triplet_quality_reasoning_without_annotations = result[
                "reasoning_without_annotations"
            ]
            self.triplet_quality_ratings_with_annotations = result[
                "ratings_with_annotations"
            ]
            self.triplet_quality_reasoning_with_annotations = result[
                "reasoning_with_annotations"
            ]

        # Save dropped triplets
        if result["dropped_triplets"] and (output_dir or self._get_triplet_cache_dir()):
            from multiview.benchmark.artifacts import (
                save_dropped_triplets_from_quality_result,
            )

            save_dir = output_dir or self._get_triplet_cache_dir()
            file_path = save_dropped_triplets_from_quality_result(
                documents=self.documents,
                quality_result=result,
                output_dir=save_dir,
                task_name=self.get_task_name(),
                min_quality=min_quality,
                document_annotations=self.document_annotations,
            )
            logger.info(
                f"✓ Saved {len(result['dropped_triplets'])} dropped triplets to {file_path}"
            )

        if min_quality is not None:
            logger.info(
                f"Filtered: {len(self.triplets)} remaining (removed {result['stats'].get('n_filtered', 0)})"
            )

        return result["stats"]

    def get_task_name(self) -> str:
        """Generate task name with config suffix: {document_set}__{criterion}__{style}__{count}.

        Includes triplet_style and max_triplets for easy identification and to prevent
        different configs from overwriting each other's artifacts.

        Examples:
            - gsm8k__arithmetic__hn__300 (hard_negative, 300 triplets)
            - gsm8k__arithmetic__rnd__500 (random, 500 triplets)
            - dickinson__theme__pre__200 (prelabeled, 200 triplets)

        Returns:
            Task name with config suffix, or base name if use_config_suffix=False
        """
        base_name = f"{self.document_set_name}__{self.criterion_name}"

        # Check if config suffix is enabled (default: True)
        use_config_suffix = self.config.get("use_config_suffix", True)

        if not use_config_suffix:
            # Legacy mode: no suffix
            return base_name

        # Add config suffix
        config_suffix = _make_config_suffix(self.config)

        return f"{base_name}__{config_suffix}"
