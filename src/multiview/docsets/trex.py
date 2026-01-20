"""T-REx Knowledge Graph Completion dataset.

Loads KG triples from T-REx (Textual Relation Extraction) from HuggingFace.
Dataset from: https://huggingface.co/datasets/relbert/t_rex

T-REx aligns Wikipedia text with Wikidata knowledge graph triples.
Unlike FB15k/WN18RR, entities are actual readable text (e.g., "Albert Einstein",
"Germany") and relations are human-readable (e.g., "[Person] was born in [Place]").

This makes it ideal for text-based evaluation without needing entity ID lookups.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.criteria_metadata import TREX_CRITERIA
from multiview.docsets.kgc_base import KGCBaseDocSet
from multiview.docsets.wikidata_properties import get_property_name

logger = logging.getLogger(__name__)


class TRExDocSet(KGCBaseDocSet):
    """T-REx Knowledge Graph Completion dataset.

    T-REx aligns Wikipedia abstracts with Wikidata triples, providing:
    - Plain text entities (e.g., "Barack Obama", "United States")
    - Human-readable relations (e.g., "[Person] is president of [Country]")
    - Wikipedia text context for each triple

    Contains 1.6M triples with 839 unique relation types covering diverse domains:
    people, places, organizations, events, creative works, etc.

    Each row contains:
    - relation: Human-readable relation template (string)
    - head: Subject entity name (plain text, 1-84 chars)
    - tail: Object entity name (plain text, 1-52 chars)
    - title: Wikipedia article title
    - text: Full Wikipedia abstract text (21-33k chars)

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "validation")
        relations (list[str], optional): Only include specific relations
        min_relation_freq (int): Minimum relation frequency (default: 10)

    Usage:
        tasks:
          - document_set: trex
            criterion: relation
            triplet_style: prelabeled
            config:
              max_docs: 500
              split: validation
    """

    DATASET_PATH = "relbert/t_rex"
    DESCRIPTION = (
        "T-REx knowledge graph with plain text entities from Wikipedia/Wikidata"
    )

    # Relation criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["relation"]

    # Metadata for LM-based criteria
    CRITERION_METADATA = TREX_CRITERIA

    def load_triples(self) -> list[dict[str, Any]]:
        """Load T-REx triples from HuggingFace.

        Returns:
            List of triple dicts: {
                "head": str,
                "relation": str,
                "tail": str,
                "title": str,
                "text": str
            }
        """
        split = self.config.get("split", "validation")
        max_docs = self.config.get("max_docs")

        logger.info(
            f"Loading T-REx from HuggingFace: {self.DATASET_PATH} " f"(split={split})"
        )

        # Use streaming for efficiency when max_docs is set
        use_streaming = max_docs is not None and max_docs < 50000

        if use_streaming:
            logger.debug(f"Using streaming mode with max_docs={max_docs}")
            dataset = load_dataset(self.DATASET_PATH, split=split, streaming=True)
        else:
            logger.debug(f"Loading full dataset split: {split}")
            dataset = load_dataset(self.DATASET_PATH, split=split)

        # Convert to list of dicts
        triples = []
        for idx, example in enumerate(dataset):
            try:
                triple = {
                    "head": example["head"],
                    "relation": example["relation"],
                    "tail": example["tail"],
                    "title": example.get("title", ""),
                    "text": example.get("text", ""),
                }
                triples.append(triple)

                # Early exit for streaming
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

        T-REx entities are already plain text, so just return as-is.

        Args:
            entity_id: Entity name (already plain text)

        Returns:
            Entity text
        """
        return str(entity_id)

    def get_relation_text(self, relation: str) -> str:
        """Get human-readable relation text.

        T-REx relations can be either:
        - P-codes (e.g., "P106") -> inflate to "occupation"
        - Templates (e.g., "[Person] was born in [Place]") -> use as-is

        Args:
            relation: Relation identifier (P-code or template string)

        Returns:
            Human-readable relation text
        """
        # Check if it's a P-code (starts with P followed by digits)
        if relation.startswith("P") and relation[1:].isdigit():
            return get_property_name(relation)

        # Otherwise return as-is (already human-readable)
        return relation
