"""InstructSTSB dataset loader from InBedder.

Loads sentence pairs with similarity scores and instructions from Hugging Face.
Dataset from: https://huggingface.co/datasets/BrandonZYW/InstructSTSB
Paper: "Answer is All You Need" (https://arxiv.org/abs/2402.09642)

NOTE: This dataset has instruction-conditioned similarity labels. The same sentence
pair can be similar or dissimilar depending on the instruction/question. For example:
  - S1: "A man is playing a harp."
  - S2: "A man is playing a keyboard."
  - Instruction "What instrument is the man playing?" → dissimilar
  - Instruction "Is the man playing an instrument?" → similar

This loader groups by sentence pairs (documents) and treats repeated instructions
as criteria. Only instructions that appear multiple times across different pairs
are kept as criteria to ensure proper multi-view evaluation.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from datasets import load_dataset

from multiview.docsets.base import BaseDocSet
from multiview.docsets.criteria_metadata import INSTRUCTSTSB_CRITERIA

logger = logging.getLogger(__name__)


class InstructSTSBDocSet(BaseDocSet):
    """InstructSTSB dataset from InBedder.

    Dataset contains 2,758 (instruction, sentence_pair, score) tuples, with 1,378
    unique sentence pairs. This loader groups by sentence pairs (as documents) and
    uses repeated instructions as criteria.

    Structure:
    - Documents: Sentence pairs (1,378 unique pairs)
    - Criteria: Instructions that appear multiple times (e.g., "What is the subject?")
    - Annotations: Binary similarity labels (similar/dissimilar) per (pair, instruction)

    Config parameters:
        max_docs (int, optional): Maximum sentence pairs to load
        split (str): Dataset split (default: "test")
        min_instruction_freq (int): Minimum times an instruction must appear to be
            used as a criterion (default: 5). Lower values = more criteria but sparser.
            Recommended values: 5 (30 criteria, 273 docs), 10 (11 criteria, 173 docs)

    Usage:
        tasks:
          - document_set: instructstsb
            criterion: all  # Use all available instruction-based criteria
            config:
              max_docs: 200
              min_instruction_freq: 5
    """

    DATASET_PATH = "BrandonZYW/InstructSTSB"
    DESCRIPTION = "Instruction-conditioned sentence pair similarity from InBedder"

    # All criteria are known (pre-labeled)
    # 'instructed_similarity' is a catch-all criterion that uses all pairwise relationships
    # Individual instruction-based criteria are dynamically added during load_documents()
    KNOWN_CRITERIA = ["instructed_similarity"]

    # Metadata for LM-based criteria (descriptions, hints, etc.)
    CRITERION_METADATA = INSTRUCTSTSB_CRITERIA

    def __init__(self, config: dict | None = None):
        """Initialize InstructSTSBDocSet.

        Args:
            config: Optional configuration dict
        """
        super().__init__(config)
        # Initialize precomputed annotations as instance variable
        # Will be populated during load_documents()
        self.PRECOMPUTED_ANNOTATIONS = {}
        self._instruction_to_criterion = {}  # Map instruction text to criterion ID
        self._document_relationships = {}  # Map sentence text -> {instruction -> {similar: set, dissimilar: set}}

    def load_documents(self) -> list[Any]:
        """Load InstructSTSB sentences from HuggingFace.

        Extracts individual sentences as documents (plain strings). Pairwise similarity
        relationships are stored separately in self._document_relationships for triplet generation.

        Returns:
            List of document strings (sentence texts)
        """
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "test")
        min_instruction_freq = self.config.get("min_instruction_freq", 5)

        logger.info(
            f"Loading InstructSTSB from HuggingFace: {self.DATASET_PATH} "
            f"(split={split}, min_instruction_freq={min_instruction_freq})"
        )

        # Load full dataset (no streaming, need to see all data to group properly)
        logger.debug(f"Loading full dataset split: {split}")
        dataset = load_dataset(self.DATASET_PATH, split=split)

        # First pass: count instruction frequencies
        instruction_counts = Counter()
        for example in dataset:
            instruction = example.get("instruction", "").strip()
            if instruction:
                instruction_counts[instruction] += 1

        # Filter to instructions that appear at least min_instruction_freq times
        valid_instructions = {
            instr
            for instr, count in instruction_counts.items()
            if count >= min_instruction_freq
        }

        logger.info(
            f"Found {len(valid_instructions)} instructions appearing "
            f">={min_instruction_freq} times (out of {len(instruction_counts)} total)"
        )

        # Second pass: extract individual sentences and build similarity relationships
        # Map: sentence -> {instruction -> {similar: [sentences], dissimilar: [sentences]}}
        sentence_to_relationships = defaultdict(
            lambda: defaultdict(lambda: {"similar": set(), "dissimilar": set()})
        )

        for example in dataset:
            instruction = example.get("instruction", "").strip()
            sentence1 = example.get("sentence1", "").strip()
            sentence2 = example.get("sentence2", "").strip()
            score = example.get("score")

            # Skip if invalid or instruction not in valid set
            if not sentence1 or not sentence2 or score is None:
                continue
            if instruction not in valid_instructions:
                continue

            # Add bidirectional relationships
            if score == 1:  # Similar
                sentence_to_relationships[sentence1][instruction]["similar"].add(
                    sentence2
                )
                sentence_to_relationships[sentence2][instruction]["similar"].add(
                    sentence1
                )
            else:  # Dissimilar
                sentence_to_relationships[sentence1][instruction]["dissimilar"].add(
                    sentence2
                )
                sentence_to_relationships[sentence2][instruction]["dissimilar"].add(
                    sentence1
                )

        # Store relationships separately and create document list (plain strings)
        documents = []
        for sentence, relationships in sentence_to_relationships.items():
            documents.append(sentence)  # Store as plain string
            self._document_relationships[sentence] = (
                relationships  # Store relationships separately
            )

        logger.info(
            f"Extracted {len(documents)} unique sentences "
            f"with {len(valid_instructions)} criteria"
        )

        # Apply max_docs limit if specified
        if max_docs and len(documents) > max_docs:
            logger.debug(f"Limiting to {max_docs} documents")
            # Limit both documents and relationships
            documents = documents[:max_docs]
            self._document_relationships = {
                k: v for k, v in self._document_relationships.items() if k in documents
            }

        # Build KNOWN_CRITERIA from valid instructions
        self._build_criteria_mapping(valid_instructions)

        # Build precomputed annotations (for similarity-based retrieval)
        self._build_precomputed_annotations(documents)

        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document.

        Args:
            document: A document (string)

        Returns:
            Text content
        """
        return document if isinstance(document, str) else str(document)

    def _build_criteria_mapping(self, valid_instructions: set[str]) -> None:
        """Build mapping from instruction text to criterion IDs.

        Creates simplified criterion IDs from instruction text and populates
        KNOWN_CRITERIA class variable.

        Args:
            valid_instructions: Set of instruction strings to use as criteria
        """
        # Sort for consistency
        sorted_instructions = sorted(valid_instructions)

        # Create mapping: instruction text -> criterion ID
        # Use simplified IDs based on index for now
        self._instruction_to_criterion = {}
        self.KNOWN_CRITERIA = ["instructed_similarity"]  # Keep the catch-all criterion

        for idx, instruction in enumerate(sorted_instructions):
            # Create a readable criterion ID from instruction
            # Take first few words, lowercase, replace spaces with underscores
            words = instruction.lower().replace("?", "").split()[:5]
            criterion_id = f"instr_{idx:03d}_{'_'.join(words)[:40]}"

            self._instruction_to_criterion[instruction] = criterion_id
            self.KNOWN_CRITERIA.append(criterion_id)

        # Store reverse mapping for display/debugging
        self._criterion_to_instruction = {
            v: k for k, v in self._instruction_to_criterion.items()
        }

        logger.info(
            f"Created {len(self.KNOWN_CRITERIA)} criteria from instructions (including 'instructed_similarity')"
        )
        logger.debug(f"Sample criteria: {self.KNOWN_CRITERIA[:5]}")

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        For InstructSTSB, documents don't have fixed labels - they have pairwise
        similarity relationships. This method is not used for triplet generation.

        Args:
            document: A document
            criterion: The criterion name (criterion ID)

        Returns:
            None (pairwise relationships are used instead)
        """
        # This dataset uses pairwise relationships, not document labels
        # Triplets are generated using get_similar_documents and get_dissimilar_documents
        return None

    def get_similar_documents(
        self, document: Any, criterion: str, all_documents: list
    ) -> list:
        """Get documents that are similar to the given document under the criterion.

        Args:
            document: The anchor document (string)
            criterion: The criterion ID (or 'instructed_similarity' for all relationships)
            all_documents: List of all documents to search from

        Returns:
            List of similar documents
        """
        # Look up relationships for this document
        relationships = self._document_relationships.get(document, {})
        if not relationships:
            return []

        # For 'instructed_similarity' criterion, aggregate across all instructions
        if criterion == "instructed_similarity":
            similar_texts = set()
            for rel_data in relationships.values():
                similar_texts.update(rel_data.get("similar", set()))
        else:
            instruction = self._criterion_to_instruction.get(criterion)
            if not instruction:
                return []
            similar_texts = relationships.get(instruction, {}).get("similar", set())

        # Find documents with matching texts
        similar_docs = [doc for doc in all_documents if doc in similar_texts]
        return similar_docs

    def get_dissimilar_documents(
        self, document: Any, criterion: str, all_documents: list
    ) -> list:
        """Get documents that are dissimilar to the given document under the criterion.

        For InstructSTSB, we use explicitly annotated negatives from ANY criterion:
        sentences that were marked with score=0 (dissimilar) under any instruction.
        This ensures high quality negatives (all explicitly annotated) while providing
        sufficient data. We exclude sentences marked as similar under the current criterion.

        Args:
            document: The anchor document (string)
            criterion: The criterion ID (or 'instructed_similarity' for all relationships)
            all_documents: List of all documents to search from

        Returns:
            List of dissimilar documents (explicitly marked as dissimilar somewhere)
        """
        # Look up relationships for this document
        relationships = self._document_relationships.get(document, {})
        if not relationships:
            return []

        # For 'instructed_similarity' criterion, get similar texts from all instructions
        if criterion == "instructed_similarity":
            similar_texts = set()
            for rel_data in relationships.values():
                similar_texts.update(rel_data.get("similar", set()))
        else:
            instruction = self._criterion_to_instruction.get(criterion)
            if not instruction:
                return []
            similar_texts = relationships.get(instruction, {}).get("similar", set())

        # Get all sentences explicitly marked as DISSIMILAR under ANY instruction
        all_dissimilar_texts = set()
        for _instr, rel_data in relationships.items():
            all_dissimilar_texts.update(rel_data.get("dissimilar", set()))

        # Exclude sentences that are similar under the current criterion
        valid_dissimilar_texts = all_dissimilar_texts - similar_texts

        # Find documents with matching texts
        dissimilar_docs = [
            doc for doc in all_documents if doc in valid_dissimilar_texts
        ]
        return dissimilar_docs

    def _build_precomputed_annotations(self, documents: list[str]) -> None:
        """Build precomputed annotations from loaded documents.

        For InstructSTSB, we track pairwise similarity relationships. Negatives
        are explicitly annotated (score=0) in the dataset.

        Args:
            documents: List of document strings
        """
        # Log statistics about relationships
        criterion_stats = defaultdict(
            lambda: {"docs_with_similar": 0, "total_similar_pairs": 0}
        )

        for doc_text in documents:
            # Get relationships for this document
            relationships = self._document_relationships.get(doc_text, {})
            if not relationships:
                continue

            for instruction, rel_data in relationships.items():
                criterion_id = self._instruction_to_criterion.get(instruction)
                if not criterion_id:
                    continue

                similar_count = len(rel_data.get("similar", set()))
                if similar_count > 0:
                    criterion_stats[criterion_id]["docs_with_similar"] += 1
                    criterion_stats[criterion_id]["total_similar_pairs"] += (
                        similar_count
                    )

        # Log statistics
        logger.info(
            f"Built similarity relationships: {len(self.KNOWN_CRITERIA)} criteria, "
            f"{len(documents)} documents (using explicitly annotated negatives)"
        )

        # Log per-criterion stats (skip 'instructed_similarity' as it's not instruction-based)
        instruction_criteria = [
            c for c in self.KNOWN_CRITERIA if c != "instructed_similarity"
        ]
        for criterion_id in instruction_criteria[
            :5
        ]:  # Show first 5 instruction-based criteria
            stats = criterion_stats[criterion_id]
            instruction = self._criterion_to_instruction[criterion_id]
            logger.debug(
                f"  {criterion_id}: {stats['docs_with_similar']} docs with similar pairs, "
                f"{stats['total_similar_pairs']} total similar pairs "
                f"- \"{instruction[:60]}...\""
            )
