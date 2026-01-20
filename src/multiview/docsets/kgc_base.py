"""Base class for Knowledge Graph Completion (KGC) datasets.

KGC tasks are fundamentally asymmetric:
- Given (head, relation), predict tail
- NOT the same as predicting head from (relation, tail)

This base class provides common functionality for loading and structuring
KG triples for asymmetric evaluation in the multiview framework.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections import defaultdict
from typing import Any

from multiview.docsets.base import BaseDocSet

logger = logging.getLogger(__name__)


class KGCBaseDocSet(BaseDocSet):
    """Base class for Knowledge Graph Completion datasets.

    KGC datasets contain triples: (head, relation, tail)

    Key design decisions for asymmetry:
    1. Documents represent (head, relation) -> tail queries
    2. The "criterion" is the relation type
    3. Triplets are: (query, correct_tail, incorrect_tail)
       where query = (head, relation)

    Subclasses must implement:
    - load_triples(): Load raw triples from dataset
    - get_entity_text(): Get text description for an entity ID
    - get_relation_text(): Get text description for a relation

    Config parameters:
        max_docs (int, optional): Maximum documents to load
        split (str): Dataset split (default: "test")
        relations (list[str], optional): Only include specific relations
        min_relation_freq (int): Minimum relation frequency to include (default: 10)
    """

    # Relation is the known criterion for KGC
    KNOWN_CRITERIA = ["relation"]

    def __init__(self, config: dict | None = None):
        """Initialize KGC dataset.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Store entity and relation metadata
        self._entity_texts: dict[str, str] = {}
        self._relation_texts: dict[str, str] = {}
        # Track relation frequencies for filtering
        self._relation_counts: dict[str, int] = defaultdict(int)

        # Initialize precomputed annotations as instance variable
        self.PRECOMPUTED_ANNOTATIONS = {}

        # Store operational metadata for triplet generation
        # Maps document_text -> metadata dict
        self._triplet_metadata: dict[str, dict] = {}

    @abstractmethod
    def load_triples(self) -> list[dict[str, Any]]:
        """Load raw KG triples from dataset.

        Returns:
            List of triple dicts with keys:
                - head: Head entity ID (string or int)
                - relation: Relation type (string)
                - tail: Tail entity ID (string or int)
        """
        pass

    @abstractmethod
    def get_entity_text(self, entity_id: str | int) -> str:
        """Get text description for an entity.

        For datasets with entity descriptions, return the description.
        For datasets with only IDs, return a formatted string.

        Args:
            entity_id: Entity identifier

        Returns:
            Text representation of the entity
        """
        pass

    @abstractmethod
    def get_relation_text(self, relation: str) -> str:
        """Get human-readable text for a relation.

        Args:
            relation: Relation identifier

        Returns:
            Human-readable relation description
        """
        pass

    def load_documents(self) -> list[Any]:
        """Load KG triples and convert to documents.

        Documents are simple strings (entity texts).
        Metadata for triplet generation is stored in self._triplet_metadata.

        Returns:
            List of document strings (entity texts)
        """
        # Load raw triples
        triples = self.load_triples()

        # Count relation frequencies
        for triple in triples:
            self._relation_counts[triple["relation"]] += 1

        # Filter by configuration
        max_docs = self.config.get("max_docs")
        relations_filter = self.config.get("relations")
        min_relation_freq = self.config.get("min_relation_freq", 10)

        # Log relation statistics
        logger.info(
            f"Loaded {len(triples)} triples with "
            f"{len(self._relation_counts)} unique relations"
        )

        # Filter out low-frequency relations
        frequent_relations = {
            rel
            for rel, count in self._relation_counts.items()
            if count >= min_relation_freq
        }
        logger.info(
            f"Keeping {len(frequent_relations)} relations with "
            f"frequency >= {min_relation_freq}"
        )

        # Convert triples to documents
        # For KGC, each triple creates TWO documents: one for head, one for tail
        # This allows us to compare: given head entity + relation, which tail is correct?
        documents = []
        metadata_list = []  # Temporary list to store metadata during loading

        for triple in triples:
            relation = triple["relation"]

            # Apply relation filters
            if relations_filter and relation not in relations_filter:
                continue
            if relation not in frequent_relations:
                continue

            # Get text representations
            head_id = triple["head"]
            tail_id = triple["tail"]

            head_text = self.get_entity_text(head_id)
            relation_text = self.get_relation_text(relation)
            tail_text = self.get_entity_text(tail_id)

            # Add HEAD document (query entity) - just the text
            documents.append(head_text)
            metadata_list.append(
                {
                    "text": head_text,
                    "entity_type": "head",
                    "entity_id": str(head_id),
                    "relation": relation,
                    "relation_text": relation_text,
                    "correct_tail_id": str(tail_id),
                }
            )

            # Add TAIL document (candidate answer) - just the text
            documents.append(tail_text)
            metadata_list.append(
                {
                    "text": tail_text,
                    "entity_type": "tail",
                    "entity_id": str(tail_id),
                    "relation": relation,
                    "relation_text": relation_text,
                    "source_head_id": str(head_id),
                }
            )

            # Check max_docs limit
            if max_docs and len(documents) >= max_docs:
                logger.info(f"Reached max_docs limit: {max_docs}")
                break

        logger.info(f"Created {len(documents)} documents from KG triples")

        # Build metadata lookups
        self._build_metadata_lookups(metadata_list)

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
        - relation: the relation type (from PRECOMPUTED_ANNOTATIONS)

        Note: Documents are strings, so criterion values come from PRECOMPUTED_ANNOTATIONS.
        Use get_precomputed_annotation() to access these.

        Args:
            document: A document (string)
            criterion: The criterion name

        Returns:
            Criterion value or None
        """
        if criterion == "relation":
            # Criterion values are stored in PRECOMPUTED_ANNOTATIONS
            # The caller should use get_precomputed_annotation() instead
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)

    def _build_metadata_lookups(self, metadata_list: list[dict]) -> None:
        """Build metadata lookups from loaded documents.

        Creates:
        1. PRECOMPUTED_ANNOTATIONS["relation"]: {document_text: {"prelabel": relation}}
        2. self._triplet_metadata: {document_text: metadata_dict}

        Args:
            metadata_list: List of metadata dicts with 'text' and other fields
        """
        annotations = {}
        relation_counts = defaultdict(int)

        for meta in metadata_list:
            text = meta.get("text")
            if not text:
                continue

            # Build precomputed annotations for relation criterion
            relation = meta.get("relation")
            if relation:
                annotations[text] = {"prelabel": relation}
                relation_counts[relation] += 1

            # Store full metadata for triplet generation
            self._triplet_metadata[text] = meta

        self.PRECOMPUTED_ANNOTATIONS["relation"] = annotations

        logger.info(
            f"Built precomputed annotations for relation: "
            f"{len(annotations)} documents across {len(relation_counts)} relations"
        )
        logger.info(
            f"Built triplet metadata for {len(self._triplet_metadata)} documents"
        )
        logger.debug(
            f"Top 10 relations: {sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:10]}"
        )

    def get_query_text(self, document: str) -> str:
        """Get the query text for asymmetric evaluation.

        For KGC, the query is (head, relation), not just head alone.

        Args:
            document: Document text (string)

        Returns:
            Query text combining head and relation
        """
        # Look up metadata for this document
        doc_text = self.get_document_text(document)
        metadata = self._triplet_metadata.get(doc_text, {})

        # For head entities, combine with relation
        if metadata.get("entity_type") == "head":
            relation_text = metadata.get("relation_text", "")
            return f"{doc_text} [RELATION: {relation_text}]"

        # For tail entities, just return the text
        return doc_text

    def get_target_text(self, document: str) -> str:
        """Get the target text for asymmetric evaluation.

        For KGC, the target is the tail entity.

        Args:
            document: Document text (string)

        Returns:
            Target (tail) text
        """
        # Documents are already just entity texts
        return self.get_document_text(document)


def create_kgc_triplets(
    documents: list[str],
    metadata_lookup: dict[str, dict],
    max_triplets: int | None = None,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Create triplets for Knowledge Graph Completion evaluation.

    Documents are individual entity texts (heads and tails).
    Task: given a head entity + relation, distinguish correct tail from incorrect tail.

    Triplet structure:
    - Anchor: HEAD entity (e.g., "Francis Palmer Smith")
    - Positive: TAIL entity that is correct for anchor (e.g., "architect")
    - Negative: TAIL entity with same relation but incorrect for anchor (e.g., "playwright")
    - Criterion: The relation (e.g., "occupation")

    Args:
        documents: List of entity texts (strings)
        metadata_lookup: Dict mapping document_text -> metadata dict
        max_triplets: Maximum number of triplets to create
        seed: Random seed for deterministic sampling

    Returns:
        List of (anchor_idx, positive_idx, negative_idx) triplets
    """
    from multiview.utils.sampling_utils import deterministic_sample

    if len(documents) < 3:
        return []

    # Separate HEAD and TAIL entities
    head_indices = []
    tail_by_relation = defaultdict(list)  # {relation: [(tail_idx, tail_id)]}

    for idx, doc_text in enumerate(documents):
        metadata = metadata_lookup.get(doc_text, {})
        entity_type = metadata.get("entity_type")

        if entity_type == "head":
            head_indices.append(idx)
        elif entity_type == "tail":
            relation = metadata.get("relation")
            entity_id = metadata.get("entity_id")
            if relation and entity_id:
                tail_by_relation[relation].append((idx, entity_id))

    if not head_indices:
        return []

    triplets = []
    used_docs = set()

    # For each HEAD entity (anchor)
    for anchor_idx in head_indices:
        if anchor_idx in used_docs:
            continue

        anchor_text = documents[anchor_idx]
        anchor_metadata = metadata_lookup.get(anchor_text, {})
        correct_tail_id = anchor_metadata.get("correct_tail_id")
        anchor_relation = anchor_metadata.get("relation")

        if not correct_tail_id or not anchor_relation:
            continue

        tail_candidates = tail_by_relation.get(anchor_relation, [])
        if len(tail_candidates) < 2:
            continue

        # Find positive: TAIL with entity_id == correct_tail_id
        pos_idx = None
        for tail_idx, tail_id in tail_candidates:
            if tail_id == correct_tail_id and tail_idx not in used_docs:
                pos_idx = tail_idx
                break

        if pos_idx is None:
            continue

        # Find negative candidates: TAIL with same relation but different entity_id
        neg_candidates = [
            tail_idx
            for tail_idx, tail_id in tail_candidates
            if tail_id != correct_tail_id
            and tail_idx not in used_docs
            and tail_idx != pos_idx
        ]

        if not neg_candidates:
            continue

        neg_idx = deterministic_sample(
            neg_candidates, 1, seed_base=f"kgc_neg_{anchor_idx}_{seed}"
        )[0]

        triplets.append((anchor_idx, pos_idx, neg_idx))
        used_docs.update([anchor_idx, pos_idx, neg_idx])

        if max_triplets is not None and len(triplets) >= max_triplets:
            break

    return triplets
