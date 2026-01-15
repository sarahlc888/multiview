"""FB15k-237 Knowledge Graph Completion dataset.

Loads KG triples from Freebase (FB15k-237) from HuggingFace.
Dataset from: https://huggingface.co/datasets/KGraph/FB15k-237

FB15k-237 was created from FB15k by removing inverse relations to eliminate
test leakage. Contains 310,079 triples with 14,505 entities and 237 relation
types from the Freebase knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.criteria_metadata import FB15K237_CRITERIA
from multiview.docsets.kgc_base import KGCBaseDocSet

logger = logging.getLogger(__name__)


class FB15k237DocSet(KGCBaseDocSet):
    """FB15k-237 Knowledge Graph Completion dataset.

    Freebase knowledge graph with 237 relation types covering diverse domains:
    - People (occupation, nationality, place of birth, ...)
    - Organizations (headquarters, industry, founded by, ...)
    - Locations (contains, capital, timezone, ...)
    - Media (genre, language, release date, ...)
    - And many more

    Each row contains a triple in text format: "[head] [relation] [tail]"

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")
        relations (list[str], optional): Only include specific relations
        min_relation_freq (int): Minimum relation frequency (default: 10)

    Usage:
        tasks:
          - document_set: fb15k237
            criterion: relation
            triplet_style: prelabeled
            config:
              max_docs: 500
              split: test
    """

    DATASET_PATH = "KGraph/FB15k-237"
    DESCRIPTION = (
        "Freebase knowledge graph completion (FB15k-237) with 237 relation types"
    )

    # Relation criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["relation"]

    # Metadata for LM-based criteria
    CRITERION_METADATA = FB15K237_CRITERIA

    def load_triples(self) -> list[dict[str, Any]]:
        """Load FB15k-237 triples from HuggingFace.

        Parses text format: "[head] [relation] [tail]"

        Returns:
            List of triple dicts: {"head": str, "relation": str, "tail": str}
        """
        split = self.config.get("split", "test")
        max_docs = self.config.get("max_docs")

        logger.info(
            f"Loading FB15k-237 from HuggingFace: {self.DATASET_PATH} "
            f"(split={split})"
        )

        # Use streaming for efficiency when max_docs is set
        use_streaming = max_docs is not None and max_docs < 10000

        if use_streaming:
            logger.debug(f"Using streaming mode with max_docs={max_docs}")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
        else:
            logger.debug(f"Loading full dataset split: {split}")
            dataset = load_dataset(self.DATASET_PATH, split=split)

        # Parse triples from text format
        triples = []
        for idx, example in enumerate(dataset):
            try:
                # Text format: "[head] [relation] [tail]"
                text = example.get("text", "").strip()

                if not text:
                    logger.debug(f"Skipping empty row at index {idx}")
                    continue

                # Split by whitespace
                parts = text.split()

                if len(parts) != 3:
                    logger.debug(
                        f"Skipping malformed row at index {idx}: "
                        f"expected 3 parts, got {len(parts)}"
                    )
                    continue

                head, relation, tail = parts

                triple = {
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                }
                triples.append(triple)

                # Early exit if we have enough triples
                # (will be further filtered in load_documents)
                if max_docs and len(triples) >= max_docs * 2:
                    logger.debug(f"Loaded {len(triples)} triples (buffered)")
                    break

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed row at index {idx}: {e}")
                continue

        logger.info(f"Loaded {len(triples)} triples (split={split})")

        return triples

    def get_entity_text(self, entity_id: str | int) -> str:
        """Get text representation for an entity.

        FB15k-237 entities are Freebase MIDs (e.g., /m/02mjmr).
        We format them for readability.

        Args:
            entity_id: Freebase MID (e.g., /m/02mjmr)

        Returns:
            Formatted entity text
        """
        # Cache entity texts
        entity_key = str(entity_id)
        if entity_key not in self._entity_texts:
            # Format: "entity /m/02mjmr" or extract last component
            # Freebase MIDs are like: /m/02mjmr, /location/country, etc.
            if "/" in entity_key:
                # Try to extract meaningful name from path
                parts = entity_key.split("/")
                # Use last non-empty component
                name = next((p for p in reversed(parts) if p), entity_key)
                self._entity_texts[entity_key] = f"entity {name}"
            else:
                self._entity_texts[entity_key] = f"entity {entity_key}"

        return self._entity_texts[entity_key]

    def get_relation_text(self, relation: str) -> str:
        """Get human-readable relation text.

        FB15k-237 relations are Freebase paths (e.g., /location/country/capital).
        We format them for readability.

        Args:
            relation: Freebase relation path

        Returns:
            Human-readable relation description
        """
        # Cache relation texts
        if relation not in self._relation_texts:
            # Format: "/location/country/capital" -> "location country capital"
            # Remove leading slash and replace slashes with spaces
            human_readable = relation.lstrip("/").replace("/", " ")
            self._relation_texts[relation] = human_readable

        return self._relation_texts[relation]
