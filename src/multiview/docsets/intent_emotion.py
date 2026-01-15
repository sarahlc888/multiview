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
    Uses intent_emotion triplet style to directly use the pre-made triplets.

    Config parameters:
        subset (str): "intent" or "emotion" (required)
        max_docs (int, optional): Maximum documents to load (counts individual texts, not triplets)
        split (str): Dataset split (default: "test")

    Usage:
        tasks:
          - document_set: intent_emotion
            criterion: intent_similarity
            triplet_style: intent_emotion
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

    def load_documents(self) -> list[Any]:
        """Load IntentEmotion triplets from HuggingFace.

        The dataset contains pre-made triplets (anchor, positive, negative).
        We load them as individual documents and store triplet metadata
        so they can be reconstructed via create_intent_emotion_triplets().

        Returns:
            List of document dicts: {"text": str, "triplet_id": int, "role": str}
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
            dataset = dataset.shuffle(seed=42)
        else:
            logger.debug(f"Loading full dataset split: {split}")
            dataset = load_dataset(self.DATASET_PATH, subset, split=split)
            dataset = dataset.shuffle(seed=42)

        # Determine which criterion we're using
        criterion = "intent_similarity" if subset == "intent" else "emotion_similarity"

        # Extract documents from triplets
        documents = []
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

                # Add anchor (mark as anchor for triplet selection)
                documents.append(
                    {
                        "text": anchor,
                        criterion: label_anchor,
                        "is_anchor": True,
                        "triplet_id": triplet_id,
                    }
                )
                doc_count += 1

                # Add positive
                documents.append(
                    {
                        "text": positive,
                        criterion: label_positive,
                        "triplet_id": triplet_id,
                    }
                )
                doc_count += 1

                # Add negative
                documents.append(
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

        # Build precomputed annotations
        self._build_precomputed_annotations(documents, criterion)

        return documents

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
        - intent_similarity: the triplet label (triplet_X_anchor/positive/negative)
        - emotion_similarity: the triplet label (triplet_X_anchor/positive/negative)

        Args:
            document: A document
            criterion: The criterion name

        Returns:
            Criterion value or None
        """
        if criterion in ["intent_similarity", "emotion_similarity"]:
            if isinstance(document, dict):
                return document.get(criterion)
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_precomputed_annotations(
        self, documents: list[dict], criterion: str
    ) -> None:
        """Build precomputed annotations from loaded documents.

        Creates a mapping: {document_text: {"criterion_value": label}}
        where label is "triplet_X_anchor/positive/negative"

        Args:
            documents: List of document dicts with 'text' and criterion fields
            criterion: The criterion name (intent_similarity or emotion_similarity)
        """
        annotations = {}

        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get("text")
                label = doc.get(criterion)

                if text and label:
                    annotations[text] = {"criterion_value": label}

        self.PRECOMPUTED_ANNOTATIONS[criterion] = annotations

        logger.info(
            f"Built precomputed annotations for {criterion}: "
            f"{len(annotations)} documents"
        )


def create_intent_emotion_triplets(
    documents: list[dict],
    max_triplets: int | None = None,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Create triplets for IntentEmotion evaluation using pre-made triplets.

    The IntentEmotion dataset contains pre-made triplets (anchor, positive, negative).
    This function reconstructs them from the documents.

    Args:
        documents: List of document dicts with triplet_id and role metadata
        max_triplets: Maximum number of triplets to return
        seed: Random seed for deterministic sampling

    Returns:
        List of (anchor_idx, positive_idx, negative_idx) triplets
    """
    from multiview.utils.sampling_utils import deterministic_sample

    # Group documents by triplet_id
    triplet_map = {}
    for idx, doc in enumerate(documents):
        triplet_id = doc.get("triplet_id")
        role = doc.get("is_anchor")  # True for anchor, False for positive/negative

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
