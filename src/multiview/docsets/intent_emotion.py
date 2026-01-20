"""IntentEmotion dataset loader from InBedder.

Loads pre-made triplets with intent and emotion similarity from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/IntentEmotion
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import INTENT_EMOTION_CRITERIA

logger = logging.getLogger(__name__)


class IntentEmotionDocSet(BaseDocSet):
    """IntentEmotion pre-made triplets from InBedder.

    Dataset contains 12,320 triplets (6.16k per subset) from BANKING77.
    Two subsets with contrasting similarity criteria:
    - intent: Intent-based similarity
    - emotion: Emotion-based similarity

    Each row contains: anchor, positive, negative
    Uses prelabeled triplet style to directly use the pre-made triplets.
    The task system automatically detects IntentEmotionDocSet and uses the appropriate logic.

    Config parameters:
        subset (str): "intent" or "emotion" (required)
        max_docs (int, optional): Maximum documents to load (counts individual texts, not triplets)
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: intent_emotion
            criterion: intent_similarity
            triplet_style: prelabeled
            config:
              subset: intent
              max_docs: 300  # Will load ~100 triplets (3 docs each)
    """

    DATASET_PATH = "BrandonZYW/IntentEmotion"
    DESCRIPTION = "IntentEmotion triplets from InBedder (intent vs emotion)"

    # Both criteria are known (pre-labeled)
    KNOWN_CRITERIA = ["intent_similarity", "emotion_similarity"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    CRITERION_METADATA = INTENT_EMOTION_CRITERIA

    def __init__(self, config: dict | None = None):
        """Initialize IntentEmotionDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}

        # Store operational metadata for triplet generation
        # Maps document_text -> metadata dict
        self._triplet_metadata: dict[str, dict] = {}

    def load_documents(self) -> list[Any]:
        """Load IntentEmotion triplets from HuggingFace.

        Documents are simple strings (text utterances).
        Metadata for triplet generation is stored in self._triplet_metadata.

        Returns:
            List of document strings
        """
        # Get config params
        subset = self.config.get("subset")
        if subset not in ["intent", "emotion"]:
            raise ValueError(
                f"subset must be 'intent' or 'emotion', got: {subset}. "
                "Set via config: {'subset': 'intent'}"
            )

        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "test")

        logger.info(
            f"Loading IntentEmotion from HuggingFace: {self.DATASET_PATH} "
            f"(subset={subset}, split={split})"
        )

        # Use streaming for large datasets or when max_docs is small
        use_streaming = max_docs is not None and max_docs < 1000

        if use_streaming:
            logger.debug(f"Using streaming mode with max_docs={max_docs}")
            dataset = load_dataset(
                self.DATASET_PATH, subset, split=split, streaming=True
            )
            dataset = dataset.shuffle(seed=42, buffer_size=10000)
        else:
            logger.debug(f"Loading full dataset split: {split}")
            dataset = load_dataset(self.DATASET_PATH, subset, split=split)
            dataset = dataset.shuffle(seed=42)

        # Determine which criterion we're using
        criterion = "intent_similarity" if subset == "intent" else "emotion_similarity"

        # Extract documents from triplets
        documents = []
        metadata_list = []  # Temporary list to store metadata during loading
        doc_count = 0

        for triplet_idx, example in enumerate(dataset):
            try:
                # Extract fields
                anchor = example.get("anchor", "").strip()
                positive = example.get("positive", "").strip()
                negative = example.get("negative", "").strip()

                # Skip if invalid
                if not anchor or not positive or not negative:
                    logger.debug(
                        f"Skipping triplet {triplet_idx}: missing anchor/positive/negative"
                    )
                    continue

                # Create labels for this triplet
                triplet_id = f"triplet_{triplet_idx}"
                label_anchor = f"{triplet_id}_anchor"
                label_positive = f"{triplet_id}_positive"
                label_negative = f"{triplet_id}_negative"

                # Add anchor - just the text
                documents.append(anchor)
                metadata_list.append(
                    {
                        "text": anchor,
                        criterion: label_anchor,
                        "is_anchor": True,
                        "triplet_id": triplet_id,
                    }
                )
                doc_count += 1

                # Add positive - just the text
                documents.append(positive)
                metadata_list.append(
                    {
                        "text": positive,
                        criterion: label_positive,
                        "triplet_id": triplet_id,
                    }
                )
                doc_count += 1

                # Add negative - just the text
                documents.append(negative)
                metadata_list.append(
                    {
                        "text": negative,
                        criterion: label_negative,
                        "triplet_id": triplet_id,
                    }
                )
                doc_count += 1

                # Check max_docs limit (applies to total documents, not triplets)
                if max_docs and doc_count >= max_docs:
                    logger.debug(f"Reached max_docs limit: {max_docs}")
                    break

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    f"Skipping malformed triplet at index {triplet_idx}: {e}"
                )
                continue

        logger.info(
            f"Loaded {len(documents)} documents from {triplet_idx + 1} triplets "
            f"(subset={subset}, split={split})"
        )

        # Build metadata lookups
        self._build_metadata_lookups(metadata_list, criterion)

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document (string)

        Returns:
            Text content
        """
        return str(document) if document else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - intent_similarity: the triplet label (from PRECOMPUTED_ANNOTATIONS)
        - emotion_similarity: the triplet label (from PRECOMPUTED_ANNOTATIONS)

        Note: Documents are strings, so criterion values come from PRECOMPUTED_ANNOTATIONS.
        Use get_precomputed_annotation() to access these.

        Args:
            document: A document (string)
            criterion: The criterion name

        Returns:
            Criterion value or None
        """
        if criterion in ["intent_similarity", "emotion_similarity"]:
            # Criterion values are stored in PRECOMPUTED_ANNOTATIONS
            # The caller should use get_precomputed_annotation() instead
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_metadata_lookups(
        self, metadata_list: list[dict], criterion: str
    ) -> None:
        """Build metadata lookups from loaded documents.

        Creates:
        1. PRECOMPUTED_ANNOTATIONS[criterion]: {document_text: {"prelabel": label}}
        2. self._triplet_metadata: {document_text: metadata_dict}

        Args:
            metadata_list: List of metadata dicts with 'text' and other fields
            criterion: The criterion name (intent_similarity or emotion_similarity)
        """
        annotations = {}

        for meta in metadata_list:
            text = meta.get("text")
            if not text:
                continue

            # Build precomputed annotations for the criterion
            label = meta.get(criterion)
            if label:
                annotations[text] = {"prelabel": label}

            # Store full metadata for triplet generation
            self._triplet_metadata[text] = meta

        self.PRECOMPUTED_ANNOTATIONS[criterion] = annotations

        logger.info(
            f"Built precomputed annotations for {criterion}: "
            f"{len(annotations)} documents"
        )
        logger.info(
            f"Built triplet metadata for {len(self._triplet_metadata)} documents"
        )


def create_intent_emotion_triplets(
    documents: list[str],
    metadata_lookup: dict[str, dict],
    max_triplets: int | None = None,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Create triplets for IntentEmotion evaluation using pre-made triplets.

    The IntentEmotion dataset contains pre-made triplets (anchor, positive, negative).
    This function reconstructs them from the documents using metadata.

    Args:
        documents: List of document texts (strings)
        metadata_lookup: Dict mapping document_text -> metadata dict
        max_triplets: Maximum number of triplets to return
        seed: Random seed for deterministic sampling

    Returns:
        List of (anchor_idx, positive_idx, negative_idx) triplets
    """
    from multiview.utils.sampling_utils import deterministic_sample

    # Group documents by triplet_id
    triplet_map = {}
    for idx, doc_text in enumerate(documents):
        metadata = metadata_lookup.get(doc_text, {})
        triplet_id = metadata.get("triplet_id")
        role = metadata.get("is_anchor")  # True for anchor, False for positive/negative

        if not triplet_id:
            continue

        if triplet_id not in triplet_map:
            triplet_map[triplet_id] = {
                "anchor": None,
                "positive": None,
                "negative": None,
            }

        # Determine role based on position in original triplet
        # Anchor has is_anchor=True, others don't
        if role:
            triplet_map[triplet_id]["anchor"] = idx
        elif (
            triplet_map[triplet_id]["anchor"] is not None
            and triplet_map[triplet_id]["positive"] is None
        ):
            triplet_map[triplet_id]["positive"] = idx
        else:
            triplet_map[triplet_id]["negative"] = idx

    # Build triplets
    triplets = []
    for _triplet_id, indices in triplet_map.items():
        anchor_idx = indices["anchor"]
        pos_idx = indices["positive"]
        neg_idx = indices["negative"]

        # Only include complete triplets
        if anchor_idx is not None and pos_idx is not None and neg_idx is not None:
            triplets.append((anchor_idx, pos_idx, neg_idx))

    # Sample if max_triplets specified
    if max_triplets and len(triplets) > max_triplets:
        triplets = deterministic_sample(
            triplets, max_triplets, f"intent_emotion_{seed}"
        )

    logger.info(
        f"Created {len(triplets)} IntentEmotion triplets from {len(documents)} documents"
    )
    return triplets
