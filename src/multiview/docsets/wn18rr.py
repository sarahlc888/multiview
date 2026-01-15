"""WN18RR Knowledge Graph Completion dataset.

Loads KG triples from WordNet (WN18RR) from HuggingFace.
Dataset from: https://huggingface.co/datasets/VLyb/WN18RR

WN18RR is derived from WN18, with data removed to eliminate test-set leakage
due to inverse relations. Contains 93,003 triples with 40,943 entities and
11 relation types from the WordNet lexical database.
"""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset

from multiview.docsets.criteria_metadata import WN18RR_CRITERIA
from multiview.docsets.kgc_base import KGCBaseDocSet

logger = logging.getLogger(__name__)


class WN18RRDocSet(KGCBaseDocSet):
    """WN18RR Knowledge Graph Completion dataset.

    WordNet-based knowledge graph with 11 relation types representing
    lexical relationships between words:
    - _hypernym: is-a relationships
    - _derivationally_related_form: word derivations
    - _instance_hypernym: instance-of relationships
    - _member_meronym: part-of relationships
    - _synset_domain_topic_of: topic associations
    - And 6 more relation types

    Each row contains: head (int), relation (str), tail (int)

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")
        relations (list[str], optional): Only include specific relations
        min_relation_freq (int): Minimum relation frequency (default: 10)

    Usage:
        tasks:
          - document_set: wn18rr
            criterion: relation
            triplet_style: prelabeled
            config:
              max_docs: 500
              split: test
    """

    DATASET_PATH = "VLyb/WN18RR"
    DESCRIPTION = "WordNet knowledge graph completion (WN18RR) with 11 relation types"

    # Relation criterion is known (pre-labeled)
    KNOWN_CRITERIA = ["relation"]

    # Metadata for LM-based criteria
    CRITERION_METADATA = WN18RR_CRITERIA

    # Human-readable relation names
    RELATION_NAMES = {
        "_hypernym": "hypernym (is a type of)",
        "_derivationally_related_form": "derivationally related form",
        "_instance_hypernym": "instance hypernym (is an instance of)",
        "_member_meronym": "member meronym (has member)",
        "_synset_domain_topic_of": "domain topic",
        "_has_part": "has part",
        "_member_of_domain_usage": "domain usage",
        "_member_of_domain_region": "domain region",
        "_verb_group": "verb group",
        "_also_see": "also see",
        "_similar_to": "similar to",
    }

    def load_triples(self) -> list[dict[str, Any]]:
        """Load WN18RR triples from HuggingFace.

        Returns:
            List of triple dicts: {"head": int, "relation": str, "tail": int}
        """
        split = self.config.get("split", "test")

        logger.info(
            f"Loading WN18RR from HuggingFace: {self.DATASET_PATH} " f"(split={split})"
        )

        # Load full dataset (it's small - only 93k triples total)
        dataset = load_dataset(self.DATASET_PATH, split=split)

        # Convert to list of dicts
        triples = []
        for example in dataset:
            triple = {
                "head": example["head"],
                "relation": example["relation"],
                "tail": example["tail"],
            }
            triples.append(triple)

        logger.info(f"Loaded {len(triples)} triples (split={split})")

        return triples

    def get_entity_text(self, entity_id: str | int) -> str:
        """Get text representation for an entity.

        WN18RR entities are integer IDs without descriptions in the dataset.
        We return a formatted entity identifier.

        Args:
            entity_id: Entity ID (int)

        Returns:
            Formatted entity text
        """
        # Cache entity texts
        entity_key = str(entity_id)
        if entity_key not in self._entity_texts:
            # Format: "entity_12345"
            self._entity_texts[entity_key] = f"entity_{entity_id}"

        return self._entity_texts[entity_key]

    def get_relation_text(self, relation: str) -> str:
        """Get human-readable relation text.

        Args:
            relation: Relation identifier (e.g., "_hypernym")

        Returns:
            Human-readable relation description
        """
        # Cache relation texts
        if relation not in self._relation_texts:
            # Use human-readable name if available, otherwise use raw relation
            human_readable = self.RELATION_NAMES.get(relation, relation)
            self._relation_texts[relation] = human_readable

        return self._relation_texts[relation]
