"""Utilities for triplet creation and sampling.

This module contains the main triplet creation logic:
    - create_random_triplets(): Simple random triplet sampling
    - create_lm_triplets(): LM-based triplet creation with candidate selection
      * Uses candidate_selection.py for filtering candidate positives/negatives
      * Uses LM judge to select final positive/negative from candidates
    - select_triplet(): LM judge that selects positive/negative from candidate sets

Related modules:
    - candidate_selection.py: BM25, embedding, Jaccard, spurious selection strategies
    - utils.py: Helper functions (jaccard_similarity, format_annotation_for_display, etc.)
"""

import logging
import random

from multiview.benchmark.triplets.utils import (
    extract_active_tags,
    format_annotation_for_display,
    jaccard_similarity,
)
from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def create_random_triplets(
    documents: list[dict] | list[str],
    max_triplets: int | None = None,
) -> list[tuple[int, int, int]]:
    """Create random triplets from documents.

    Args:
        documents: List of documents (used only to determine size)
        max_triplets: Maximum number of triplets to create (None = unlimited)

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets as document indices
    """
    if len(documents) < 3:
        return []

    triplets = []
    num_triplets = max_triplets if max_triplets is not None else len(documents)

    for _ in range(num_triplets):
        # Sample 3 random distinct indices
        sampled_indices = random.sample(range(len(documents)), 3)
        triplets.append(tuple(sampled_indices))

    return triplets


def select_triplet(
    anchor_idx: int,
    candidate_indices: list[int],
    documents: list[str],
    annotations: list[dict],
    criterion: str,
    criterion_description: str,
    triplet_example: dict | str | None = None,
    cache_alias: str | None = None,
) -> tuple[int, int]:
    """Use LM judge to select positive and negative from candidates.

    Args:
        anchor_idx: Index of anchor document
        candidate_indices: List of candidate document indices
        documents: List of all documents
        annotations: List of all annotations
        criterion: Criterion name
        criterion_description: Criterion description
        triplet_example: Optional example guidance (dict or string)
        cache_alias: Cache alias for LM calls

    Returns:
        Tuple of (positive_idx, negative_idx)
    """

    if len(candidate_indices) < 2:
        # Not enough candidates, return first two
        return candidate_indices[0], candidate_indices[-1]

    # Get anchor info
    anchor_doc = documents[anchor_idx]
    anchor_ann = annotations[anchor_idx]

    # Format anchor annotation
    anchor_annotation = format_annotation_for_display(anchor_ann)

    # Compute Jaccard similarities for all candidates
    anchor_tags = extract_active_tags(anchor_ann, "tags")
    anchor_spurious = extract_active_tags(anchor_ann, "spurious_tags")

    # Format candidates with scores
    candidates_text_parts = []
    for i, cand_idx in enumerate(candidate_indices):
        cand = documents[cand_idx]
        cand_ann = annotations[cand_idx]

        # Compute true and spurious Jaccard similarity
        cand_tags = extract_active_tags(cand_ann, "tags")
        cand_spurious = extract_active_tags(cand_ann, "spurious_tags")

        true_sim = jaccard_similarity(anchor_tags, cand_tags)
        spurious_sim = jaccard_similarity(anchor_spurious, cand_spurious)

        cand_annotation = format_annotation_for_display(cand_ann)

        cand_text = f"[{i+1}] {cand}\n"
        cand_text += f"Annotation: {cand_annotation}\n"
        cand_text += f"True Tag Similarity: {true_sim:.2f} | Spurious Tag Similarity: {spurious_sim:.2f}"

        candidates_text_parts.append(cand_text)

    candidates_text = "\n\n".join(candidates_text_parts)

    # Format triplet example section
    triplet_example_section = ""
    if triplet_example:
        if isinstance(triplet_example, dict):
            anchor_ex = triplet_example.get("anchor", "")
            pos_ex = triplet_example.get("pos", "")
            neg_ex = triplet_example.get("neg", "")
            triplet_example_section = f"""TRIPLET EXAMPLE GUIDANCE:
This example illustrates what makes a good triplet for this criterion:

EXAMPLE ANCHOR: {anchor_ex}
EXAMPLE POSITIVE: {pos_ex}
EXAMPLE NEGATIVE: {neg_ex}

"""
        else:
            triplet_example_section = f"TRIPLET EXAMPLE GUIDANCE:\n{triplet_example}\n"

    # Prepare inputs for LM judge
    inputs = {
        "criterion": [criterion],
        "criterion_description": [criterion_description or criterion],
        "triplet_example_section": [triplet_example_section],
        "anchor_doc": [anchor_doc],
        "anchor_annotation": [anchor_annotation],
        "candidates": [candidates_text],
    }

    # Run LM judge
    results = run_inference(
        inputs=inputs,
        config="triplet_selection_gemini",
        cache_alias=cache_alias,
        verbose=False,
    )

    response = results[0] if results else ""

    # Parse response for CHOSEN POSITIVE and CHOSEN NEGATIVE
    positive_idx = None
    negative_idx = None

    for line in response.split("\n"):
        line_upper = line.upper().strip()
        if "CHOSEN POSITIVE:" in line_upper or "POSITIVE:" in line_upper:
            try:
                num = int("".join(c for c in line if c.isdigit()))
                if 1 <= num <= len(candidate_indices):
                    positive_idx = candidate_indices[num - 1]
            except ValueError:
                pass
        elif "CHOSEN NEGATIVE:" in line_upper or "NEGATIVE:" in line_upper:
            try:
                num = int("".join(c for c in line if c.isdigit()))
                if 1 <= num <= len(candidate_indices):
                    negative_idx = candidate_indices[num - 1]
            except ValueError:
                pass

    # Fallback: use heuristic if parsing fails
    if positive_idx is None:
        positive_idx = candidate_indices[0]
        logger.warning("Failed to parse positive selection, using first candidate")
    if negative_idx is None:
        negative_idx = candidate_indices[-1]
        logger.warning("Failed to parse negative selection, using last candidate")
    if positive_idx == negative_idx and len(candidate_indices) > 1:
        # Ensure they're different
        negative_idx = (
            candidate_indices[1]
            if positive_idx == candidate_indices[0]
            else candidate_indices[0]
        )

    return positive_idx, negative_idx


def create_lm_triplets(
    documents: list[str],
    annotations: list[dict] | None = None,
    max_triplets: int | None = None,
    candidate_strategy: str = "multi",
    use_spurious_hard_negs: bool = True,
    embedding_preset: str = "hf_qwen3_embedding_8b",
    lm_judge_preset: str = "triplet_selection_gemini",
    criterion: str | None = None,
    criterion_description: str | None = None,
    cache_alias_prefix: str | None = None,
    triplet_example: dict | str | None = None,
    anchor_indices: list[int] | None = None,
) -> list[tuple[int, int, int]]:
    """Create triplets using language model judge with candidate selection.

    This function:
    1. For each anchor document, selects a pool of candidates using various strategies
    2. Uses an LM judge to compare anchor with candidates
    3. Selects the most similar (positive) and least similar (negative) from the pool

    Args:
        documents: List of document strings
        annotations: Optional list of rich annotation dicts (required for multi strategy)
        max_triplets: Maximum number of triplets to create (None = unlimited)
        candidate_strategy: Strategy for candidate selection:
            - "bm25": BM25 similarity over summaries
            - "embedding": Embedding similarity
            - "jaccard": Jaccard similarity over tags
            - "multi": Combine all strategies
        use_spurious_hard_negs: If True, add spurious hard negatives to candidate pool
        embedding_preset: Preset for embedding model (if using embedding strategy)
        lm_judge_preset: Preset for LM judge
        criterion: Criterion name (for LM judge prompt)
        criterion_description: Description of criterion (for LM judge prompt)
        cache_alias_prefix: Prefix for cache aliases
        anchor_indices: Optional list of document indices to use as anchors.
            If provided, uses these indices as anchors (useful for synthesis).
            If None, uses sequential indices 0, 1, 2, ..., num_triplets-1.

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets as document indices
    """
    from multiview.benchmark.triplets.candidate_selection import (
        merge_candidate_pools,
        select_candidates_bm25,
        select_candidates_embedding,
        select_candidates_jaccard,
        select_spurious_hard_negatives,
    )

    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Determine anchor indices to use
    if anchor_indices is not None:
        # Use provided anchor indices (e.g., from synthesis)
        anchors_to_process = anchor_indices
        # Respect max_triplets if provided
        if max_triplets is not None:
            anchors_to_process = anchors_to_process[:max_triplets]
    else:
        # Default: sequential indices
        num_triplets = max_triplets if max_triplets is not None else len(documents)
        num_triplets = min(num_triplets, len(documents))
        anchors_to_process = list(range(num_triplets))

    # Check if annotations are needed
    needs_annotations = (
        candidate_strategy in ["multi", "jaccard", "bm25", "embedding"]
        or use_spurious_hard_negs
    )
    if needs_annotations and (
        annotations is None or len(annotations) != len(documents)
    ):
        raise ValueError(
            f"Annotations required for candidate_strategy='{candidate_strategy}' or use_spurious_hard_negs=True"
        )

    triplets = []

    logger.info(
        f"Creating {len(anchors_to_process)} triplets using LM judge with {candidate_strategy} strategy"
    )

    # Process each anchor
    for i, anchor_idx in enumerate(anchors_to_process):
        logger.debug(
            f"Processing anchor {i + 1}/{len(anchors_to_process)} (doc_idx={anchor_idx})"
        )

        # Step 1: Select candidate pool
        if candidate_strategy == "bm25":
            candidates = select_candidates_bm25(
                documents, annotations, anchor_idx, k=20, use_summary=True
            )
            candidate_indices = [idx for idx, score in candidates]

        elif candidate_strategy == "embedding":
            cache_alias = (
                f"{cache_alias_prefix}_embedding" if cache_alias_prefix else None
            )
            candidates = select_candidates_embedding(
                documents,
                annotations,
                anchor_idx,
                k=20,
                embedding_preset=embedding_preset,
                use_summary=True,
                cache_alias=cache_alias,
            )
            candidate_indices = [idx for idx, score in candidates]

        elif candidate_strategy == "jaccard":
            candidates = select_candidates_jaccard(
                annotations, anchor_idx, k=20, use_spurious=False
            )
            candidate_indices = [idx for idx, score in candidates]

        elif candidate_strategy == "multi":
            # Combine multiple strategies
            bm25_candidates = select_candidates_bm25(
                documents, annotations, anchor_idx, k=10, use_summary=True
            )
            cache_alias = (
                f"{cache_alias_prefix}_embedding" if cache_alias_prefix else None
            )
            emb_candidates = select_candidates_embedding(
                documents,
                annotations,
                anchor_idx,
                k=10,
                embedding_preset=embedding_preset,
                use_summary=True,
                cache_alias=cache_alias,
            )
            jaccard_candidates = select_candidates_jaccard(
                annotations, anchor_idx, k=10, use_spurious=False
            )

            candidate_indices = merge_candidate_pools(
                bm25_candidates, emb_candidates, jaccard_candidates, deduplicate=True
            )

        else:
            raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")

        # Add spurious hard negatives if requested
        if use_spurious_hard_negs and annotations is not None:
            spurious_candidates = select_spurious_hard_negatives(
                annotations, anchor_idx, k=5
            )
            spurious_indices = [idx for idx, score in spurious_candidates]
            candidate_indices = list(set(candidate_indices + spurious_indices))

        # Ensure we have enough candidates
        if len(candidate_indices) < 2:
            logger.warning(f"Not enough candidates for anchor {anchor_idx}, skipping")
            continue

        # Step 2: Use LM judge to select positive and negative from candidates
        judge_cache_alias = (
            f"{cache_alias_prefix}_judge" if cache_alias_prefix else None
        )

        positive_idx, negative_idx = select_triplet(
            anchor_idx=anchor_idx,
            candidate_indices=candidate_indices,
            documents=documents,
            annotations=annotations,
            criterion=criterion or "similarity",
            criterion_description=criterion_description
            or criterion
            or "general similarity",
            triplet_example=triplet_example,
            cache_alias=judge_cache_alias,
        )

        # Create triplet (IDs only, not text)
        triplet = (anchor_idx, positive_idx, negative_idx)
        triplets.append(triplet)

    logger.info(f"Created {len(triplets)} triplets")
    return triplets
