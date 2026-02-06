"""Utilities for triplet creation and sampling.

This module contains the main triplet creation logic:
    - create_random_triplets(): Simple random triplet sampling
    - create_lm_triplets(): LM-based triplet creation with candidate selection
      * Positive candidates: high similarity on summaries/tags (BM25/embedding/Jaccard)
      * Negative candidates: high similarity on raw docs, low similarity on tags (hard negatives)
      * Uses two-stage LM judge: select_positive_batch() then select_negative_batch()
    - select_positive_batch(): LM judge that selects positives from candidates
    - select_negative_batch(): LM judge that selects negatives from candidates (sees positive)

Related modules:
    - candidate_selection.py: BM25, embedding, Jaccard retrieval strategies
    - utils.py: Helper functions (jaccard_similarity, format_annotation_for_display, etc.)
"""

from __future__ import annotations

import logging

import numpy as np

from multiview.benchmark.annotations import extract_image, extract_text
from multiview.benchmark.triplets.utils import (
    annotation_final_summary,
    extract_active_tags,
    format_annotation_for_display,
    jaccard_similarity,
)
from multiview.inference.inference import run_inference
from multiview.utils.bm25_utils import compute_bm25_matrix
from multiview.utils.prompt_utils import read_or_return
from multiview.utils.sampling_utils import deterministic_sample

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

    for i in range(num_triplets):
        # Sample 3 random distinct indices deterministically
        sampled_indices = deterministic_sample(
            list(range(len(documents))), 3, seed_base=f"random_triplet_{i}"
        )
        triplets.append(tuple(sampled_indices))

    return triplets


def _select_with_bm25(
    anchor_idx: int,
    documents: list[str],
    candidate_indices: list[int],
    num_to_select: int,
    prefer_low_score: bool,
    seed: int,
) -> list[int]:
    """Select candidates using BM25 scores.

    Args:
        anchor_idx: Index of the anchor document
        documents: List of all document texts
        candidate_indices: List of candidate document indices to select from
        num_to_select: Number of candidates to select
        prefer_low_score: If True, prefer LOW BM25 scores (hard positives)
                         If False, prefer HIGH BM25 scores (hard negatives)
        seed: Random seed for tie-breaking

    Returns:
        List of selected candidate indices

    Note:
        - Hard positives: Documents that share a criterion but have low text similarity
        - Hard negatives: Documents that don't share a criterion but have high text similarity
    """
    from multiview.utils.bm25_utils import compute_bm25_scores

    if len(candidate_indices) == 0:
        return []

    if len(candidate_indices) <= num_to_select:
        return candidate_indices

    # Compute BM25 scores from anchor to all documents
    bm25_scores = compute_bm25_scores(documents, anchor_idx)

    # Extract scores for candidate indices
    candidate_scores = [(idx, bm25_scores[idx]) for idx in candidate_indices]

    # Sort by score
    if prefer_low_score:
        # For hard positives: lowest scores first
        candidate_scores.sort(key=lambda x: (x[1], x[0]))  # Use idx as tiebreaker
    else:
        # For hard negatives: highest scores first
        candidate_scores.sort(key=lambda x: (-x[1], x[0]))  # Use idx as tiebreaker

    # Select top num_to_select
    selected = [idx for idx, _ in candidate_scores[:num_to_select]]

    return selected


def create_prelabeled_triplets(
    documents: list[str] | list[dict],
    annotations: list[dict],
    max_triplets: int | None = None,
    selection_strategy: str = "hard_negatives",
    allow_multi_class: bool = True,
    seed: int = 42,
    metadata_lookup: dict[str, dict] | None = None,
) -> list[tuple[int, int, int]]:
    """Create triplets based on pre-existing criterion labels.

    For datasets with known gold labels, creates triplets where:
    - Positives: Documents sharing at least one prelabel with anchor
    - Negatives: Documents sharing NO prelabels with anchor

    Default uses "hard_negatives" strategy (BM25-based selection)

    Args:
        documents: List of document texts (strings) or dicts (will extract text if dicts)
        annotations: List of annotation dicts with "prelabel" field
        max_triplets: Max number of triplets (None = use all documents as anchors)
        selection_strategy: "random" or "hard_negatives" (BM25-based)
        allow_multi_class: Allow documents to have multiple prelabels
        seed: Random seed for deterministic sampling
        metadata_lookup: Optional dict mapping document_text -> metadata dict (for is_anchor markers)

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets

    Raises:
        ValueError: If documents don't have sufficient criterion diversity
    """
    from collections import defaultdict

    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Extract text from documents (handle both string and dict formats for backward compatibility)
    document_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            document_texts.append(doc.get("text", str(doc)))
        elif isinstance(doc, str):
            document_texts.append(doc)
        else:
            document_texts.append(str(doc))

    documents = document_texts

    # Build classmap: {prelabel: [doc_indices]}
    classmap = defaultdict(list)
    doc_to_classes = {}  # doc_idx -> [prelabels]

    for idx, ann in enumerate(annotations):
        prelabel = ann.get("prelabel")
        if prelabel is None:
            logger.warning(f"Document {idx} has no prelabel, skipping")
            continue

        # Support multi-class: prelabel can be a list or single value
        if isinstance(prelabel, list):
            values = prelabel
        elif isinstance(prelabel, str) and ", " in prelabel:
            # Handle comma-separated multi-label strings (e.g., "Creative Content, Writing Genres")
            values = [v.strip() for v in prelabel.split(",")]
        else:
            values = [prelabel]

        doc_to_classes[idx] = values
        for value in values:
            classmap[value].append(idx)

    if len(classmap) < 2:
        logger.warning(
            f"Need at least 2 distinct prelabels to create triplets, found {len(classmap)}"
        )
        return []

    logger.info(
        f"Building prelabeled triplets: {len(doc_to_classes)} documents, "
        f"{len(classmap)} distinct criterion values"
    )

    # Helper: check if two doc indices share any criterion value
    def shares_criterion(idx1: int, idx2: int) -> bool:
        if idx1 not in doc_to_classes or idx2 not in doc_to_classes:
            return False
        classes1 = set(doc_to_classes[idx1])
        classes2 = set(doc_to_classes[idx2])
        return len(classes1 & classes2) > 0

    # Select anchor documents
    # If documents have is_anchor markers (via metadata_lookup), only use those as anchors
    marked_anchor_indices = []
    if metadata_lookup:
        for idx in doc_to_classes.keys():
            doc_text = documents[idx]
            metadata = metadata_lookup.get(doc_text, {})
            if metadata.get("is_anchor", False):
                marked_anchor_indices.append(idx)

    if marked_anchor_indices:
        logger.info(
            f"Found {len(marked_anchor_indices)} marked anchors, using only those"
        )
        anchor_indices = marked_anchor_indices
    else:
        anchor_indices = list(doc_to_classes.keys())

    # Shuffle anchors for variety when cycling
    anchor_indices = deterministic_sample(
        anchor_indices,
        len(anchor_indices),
        seed_base=f"criterion_anchors_shuffle_{seed}",
    )

    triplets = []
    # Track which positives and negatives have been used for each anchor
    used_pos_per_anchor = {idx: set() for idx in anchor_indices}
    used_neg_per_anchor = {idx: set() for idx in anchor_indices}
    # Track how many triplets we've created per anchor
    triplets_per_anchor = dict.fromkeys(anchor_indices, 0)

    # Cycle through anchors until we hit max_triplets
    anchor_cycle_idx = 0
    max_attempts = len(anchor_indices) * 100  # Prevent infinite loops
    attempts = 0

    while len(triplets) < (max_triplets or float("inf")) and attempts < max_attempts:
        attempts += 1
        anchor_idx = anchor_indices[anchor_cycle_idx % len(anchor_indices)]
        anchor_cycle_idx += 1
        # Find positive candidates: share at least one criterion value, not anchor itself
        # Exclude already-used positives for this anchor
        pos_candidates = [
            idx
            for idx in doc_to_classes.keys()
            if idx != anchor_idx
            and idx not in used_pos_per_anchor[anchor_idx]
            and shares_criterion(anchor_idx, idx)
        ]

        # Find negative candidates: share NO criterion values
        # Exclude already-used negatives for this anchor
        neg_candidates = [
            idx
            for idx in doc_to_classes.keys()
            if idx != anchor_idx
            and idx not in used_neg_per_anchor[anchor_idx]
            and not shares_criterion(anchor_idx, idx)
        ]

        if len(pos_candidates) == 0:
            # No more valid positives for this anchor, skip
            continue

        if len(neg_candidates) == 0:
            # No more valid negatives for this anchor, skip
            continue

        # Select positive and negative based on strategy
        # Use triplet count for this anchor to vary selections
        triplet_num = triplets_per_anchor[anchor_idx]

        if selection_strategy == "random":
            # Random selection - vary seed by triplet number to get different candidates
            pos_idx = deterministic_sample(
                pos_candidates,
                1,
                seed_base=f"criterion_pos_{anchor_idx}_{triplet_num}_{seed}",
            )[0]
            neg_idx = deterministic_sample(
                neg_candidates,
                1,
                seed_base=f"criterion_neg_{anchor_idx}_{triplet_num}_{seed}",
            )[0]

        elif selection_strategy == "hard_negatives":
            # BM25-based selection - candidates are already filtered to exclude used ones
            pos_idx = _select_with_bm25(
                anchor_idx=anchor_idx,
                documents=documents,
                candidate_indices=pos_candidates,
                num_to_select=1,
                prefer_low_score=True,  # Hard positives: low BM25
                seed=seed + triplet_num,  # Vary seed to get different results
            )[0]

            neg_idx = _select_with_bm25(
                anchor_idx=anchor_idx,
                documents=documents,
                candidate_indices=neg_candidates,
                num_to_select=1,
                prefer_low_score=False,  # Hard negatives: high BM25
                seed=seed + triplet_num,  # Vary seed to get different results
            )[0]

        else:
            raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

        # Validate all three indices are distinct
        if anchor_idx == pos_idx or anchor_idx == neg_idx or pos_idx == neg_idx:
            logger.error(
                f"CRITICAL: Skipping invalid triplet with non-distinct indices: "
                f"anchor={anchor_idx}, positive={pos_idx}, negative={neg_idx}"
            )
            continue

        # Add triplet and mark pos/neg as used for this anchor
        triplets.append((anchor_idx, pos_idx, neg_idx))
        used_pos_per_anchor[anchor_idx].add(pos_idx)
        used_neg_per_anchor[anchor_idx].add(neg_idx)
        triplets_per_anchor[anchor_idx] += 1

    # Log statistics
    num_anchors_used = sum(1 for count in triplets_per_anchor.values() if count > 0)
    avg_triplets_per_anchor = (
        len(triplets) / num_anchors_used if num_anchors_used > 0 else 0
    )
    max_triplets_for_anchor = (
        max(triplets_per_anchor.values()) if triplets_per_anchor else 0
    )

    logger.info(
        f"Created {len(triplets)} prelabeled triplets from {num_anchors_used} anchors "
        f"(avg: {avg_triplets_per_anchor:.1f} triplets/anchor, max: {max_triplets_for_anchor})"
    )

    if len(triplets) < (max_triplets or 0):
        logger.warning(
            f"Only created {len(triplets)}/{max_triplets} requested triplets. "
            f"Exhausted available unique combinations."
        )

    return triplets


def _format_candidates_text(
    candidate_indices: list[int],
    documents: list[str],
    annotations: list[dict],
    anchor_idx: int,
    anchor_ann: dict,
    bm25_scores: np.ndarray | None = None,
    include_spurious_tags: bool = False,
) -> str:
    """Format candidates with similarity scores for display.

    Args:
        candidate_indices: List of candidate document indices
        documents: List of all documents
        annotations: List of all annotations
        anchor_idx: Index of anchor document (for BM25 computation)
        anchor_ann: Anchor document annotation
        bm25_scores: Optional precomputed BM25 scores for the anchor
        include_spurious_tags: If True, include spurious tags in annotations

    Returns:
        Formatted candidates text
    """
    from multiview.utils.bm25_utils import compute_bm25_scores

    # Check if annotations have tags (for lm_tags triplet style)
    has_tags = "tags" in anchor_ann or "spurious_tags" in anchor_ann

    if has_tags:
        anchor_tags = extract_active_tags(anchor_ann, "tags")
        anchor_spurious = extract_active_tags(anchor_ann, "spurious_tags")

    if bm25_scores is None:
        bm25_scores = compute_bm25_scores(documents, anchor_idx)

    candidates_text_parts = []
    for i, cand_idx in enumerate(candidate_indices):
        cand_doc = documents[cand_idx]
        cand_ann = annotations[cand_idx]

        # Extract text from document (image-only docs have text="<image>")
        cand_text_content = extract_text(cand_doc)

        # Get BM25 lexical similarity score
        lexical_sim = bm25_scores[cand_idx]

        cand_annotation = format_annotation_for_display(
            cand_ann, include_spurious=include_spurious_tags
        )

        cand_text = f"[Document {i+1}]\n{cand_text_content}\n\n"
        cand_text += f"[Annotation {i+1}]\n{cand_annotation}\n"

        # Determine if this is an image-only document (no real text for lexical similarity)
        is_image_only = cand_text_content.strip() == "<image>"

        # Only show tag-based similarity scores if annotations have tags
        if has_tags:
            # Compute true and spurious Jaccard similarity
            cand_tags = extract_active_tags(cand_ann, "tags")
            cand_spurious = extract_active_tags(cand_ann, "spurious_tags")

            true_sim = jaccard_similarity(anchor_tags, cand_tags)
            spurious_sim = jaccard_similarity(anchor_spurious, cand_spurious)

            # For image-only docs, skip lexical similarity (not meaningful)
            if is_image_only:
                cand_text += f"Similarity to anchor: Criterion={true_sim:.2f} | Spurious={spurious_sim:.2f}"
            else:
                cand_text += f"Similarity to anchor: Criterion={true_sim:.2f} | Spurious={spurious_sim:.2f} | Lexical={lexical_sim:.2f}"
        elif not is_image_only:
            # Only show lexical similarity for text documents
            cand_text += f"Similarity to anchor: Lexical={lexical_sim:.2f}"

        candidates_text_parts.append(cand_text)

    return "\n\n".join(candidates_text_parts)


def _format_triplet_example_section(
    triplet_example_hint: dict | str | None, selection_type: str = "positive"
) -> str:
    """Format the triplet example section for prompts.

    Args:
        triplet_example_hint: Dict with anchor/pos/neg examples or string guidance
        selection_type: "positive" or "negative" to use appropriate template

    Returns:
        Formatted example section string
    """
    if not triplet_example_hint:
        return ""

    if isinstance(triplet_example_hint, dict):
        anchor_ex = triplet_example_hint.get("anchor", "")
        pos_ex = triplet_example_hint.get("pos", "")
        neg_ex = triplet_example_hint.get("neg", "")

        # Use different templates for positive vs negative selection
        if selection_type == "negative":
            template_path = "prompts/triplet/triplet_example_guidance_for_negative.txt"
        else:
            template_path = "prompts/triplet/triplet_example_guidance_for_positive.txt"

        section_template = read_or_return(template_path)
        return section_template.format(
            anchor_ex=anchor_ex,
            pos_ex=pos_ex,
            neg_ex=neg_ex,
        )
    else:
        section_template = read_or_return(
            "prompts/triplet/triplet_example_guidance_text.txt"
        )
        return section_template.format(triplet_example=triplet_example_hint)


def _coerce_selected_num(selected_num: object) -> object:
    if isinstance(selected_num, list) and len(selected_num) == 1:
        return selected_num[0]
    return selected_num


def select_positive_batch(
    anchor_indices: list[int],
    candidate_indices_by_anchor: list[list[int]],
    documents: list[str],
    annotations: list[dict],
    criterion: str,
    criterion_description: str,
    triplet_example_hint: dict | str | None = None,
    lm_judge_preset: str = "triplet_select_positive_gemini",
    bm25_scores_by_anchor: dict[int, np.ndarray] | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> tuple[list[int | None], list[bool], list[bool]]:
    """Use LM judge to select positives for multiple anchors in one batch.

    Returns:
        tuple of (positive_indices, parse_successes, abstentions):
        - positive_indices: Selected positive index for each anchor (None if abstained or failed)
        - parse_successes: Whether the LM response was successfully parsed
        - abstentions: Whether the LM chose to abstain (return 0)
    """
    if not anchor_indices:
        return [], [], []

    # Separate single-candidate from multi-candidate cases
    positive_indices = []
    parse_successes = []
    abstentions = []
    multi_anchor_indices = []
    multi_candidate_lists = []
    multi_positions = []  # Track original positions for mapping results back

    for i, (anchor_idx, candidate_indices) in enumerate(
        zip(anchor_indices, candidate_indices_by_anchor, strict=False)
    ):
        if len(candidate_indices) == 1:
            # Auto-select the only candidate (no LM judge needed)
            positive_indices.append(candidate_indices[0])
            parse_successes.append(True)
            abstentions.append(False)
        else:
            # Multi-candidate case: needs LM judge
            multi_anchor_indices.append(anchor_idx)
            multi_candidate_lists.append(candidate_indices)
            multi_positions.append(i)
            # Add placeholders for now
            positive_indices.append(None)
            parse_successes.append(False)
            abstentions.append(False)

    # If no multi-candidate cases, return early
    if not multi_anchor_indices:
        num_auto = len(anchor_indices)
        logger.debug(
            f"Auto-selected {num_auto} single-candidate positives (no LM judge needed)"
        )
        return positive_indices, parse_successes, abstentions

    # Log the optimization
    num_auto = len(anchor_indices) - len(multi_anchor_indices)
    if num_auto > 0:
        logger.debug(
            f"Auto-selected {num_auto} single-candidate positives, "
            f"calling LM judge for {len(multi_anchor_indices)} multi-candidate cases"
        )

    # Build prompts only for multi-candidate cases
    triplet_example_section = _format_triplet_example_section(
        triplet_example_hint, selection_type="positive"
    )

    # Check if annotations have tags to determine if we should show similarity note
    has_tags = len(annotations) > 0 and (
        "tags" in annotations[0] or "spurious_tags" in annotations[0]
    )
    similarity_note = (
        '- NOTE: Each candidate is annotated with  "Criterion" (similarity on what matters), "Spurious" (similarity on surface features), and "Lexical" (BM25 word overlap)\n'
        if has_tags
        else ""
    )

    anchor_docs = []
    all_images = []  # List of image lists (one per prompt)
    anchor_annotations = []
    candidates_texts = []

    for anchor_idx, candidate_indices in zip(
        multi_anchor_indices, multi_candidate_lists, strict=False
    ):
        anchor_doc = documents[anchor_idx]
        # Extract text and image separately for proper formatting
        anchor_text = extract_text(anchor_doc)
        anchor_image = extract_image(anchor_doc)

        # If document has an image, use <image> placeholder to mark where it should go
        if anchor_image and not anchor_text:
            anchor_text = "<image>"

        anchor_docs.append(anchor_text)

        # Collect all images for this prompt: anchor + candidates
        prompt_images = []
        if anchor_image:
            prompt_images.append(anchor_image)

        # Add candidate images
        for cand_idx in candidate_indices:
            cand_image = extract_image(documents[cand_idx])
            if cand_image:
                prompt_images.append(cand_image)

        # Add to all_images (list of lists)
        all_images.append(prompt_images if prompt_images else None)

        anchor_ann = annotations[anchor_idx]
        anchor_annotations.append(format_annotation_for_display(anchor_ann))
        bm25_scores = (
            bm25_scores_by_anchor.get(anchor_idx) if bm25_scores_by_anchor else None
        )
        candidates_texts.append(
            _format_candidates_text(
                candidate_indices,
                documents,
                annotations,
                anchor_idx,
                anchor_ann,
                bm25_scores=bm25_scores,
            )
        )

    inputs = {
        "criterion": [criterion] * len(multi_anchor_indices),
        "criterion_description": [criterion_description] * len(multi_anchor_indices),
        "triplet_example_section": [triplet_example_section]
        * len(multi_anchor_indices),
        "similarity_note": [similarity_note] * len(multi_anchor_indices),
        "anchor_doc": anchor_docs,
        "anchor_annotation": anchor_annotations,
        "candidates": candidates_texts,
    }

    # Add images if any prompt has images (list of lists format)
    if any(imgs for imgs in all_images):
        inputs["images"] = all_images

    logger.debug(
        f"Running LM judge for positive selection (batch_size={len(multi_anchor_indices)}, "
        f"cache_alias={cache_alias})"
    )
    results = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )

    # Map LM results back to original positions
    for candidate_indices, selected_num, orig_pos in zip(
        multi_candidate_lists, results, multi_positions, strict=False
    ):
        selected_num = _coerce_selected_num(selected_num)
        if isinstance(selected_num, int) and selected_num == 0:
            # LM chose to abstain
            logger.debug(
                f"LM judge abstained on positive selection for anchor at position {orig_pos}"
            )
            positive_indices[orig_pos] = None
            parse_successes[orig_pos] = True
            abstentions[orig_pos] = True
        elif isinstance(selected_num, int) and 1 <= selected_num <= len(
            candidate_indices
        ):
            positive_indices[orig_pos] = candidate_indices[selected_num - 1]
            parse_successes[orig_pos] = True
            abstentions[orig_pos] = False
        else:
            logger.error(
                "CRITICAL: Failed to parse positive selection from LM response. "
                f"Got: {selected_num}. Skipping triplet."
            )
            positive_indices[orig_pos] = None
            parse_successes[orig_pos] = False
            abstentions[orig_pos] = False

    return positive_indices, parse_successes, abstentions


def select_negative_batch(
    anchor_indices: list[int],
    positive_indices: list[int],
    candidate_indices_by_anchor: list[list[int]],
    documents: list[str],
    annotations: list[dict],
    criterion: str,
    criterion_description: str,
    triplet_example_hint: dict | str | None = None,
    lm_judge_preset: str = "triplet_select_negative_gemini",
    bm25_scores_by_anchor: dict[int, np.ndarray] | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> tuple[list[int | None], list[bool], list[bool]]:
    """Use LM judge to select negatives for multiple anchors in one batch.

    Returns:
        tuple of (negative_indices, parse_successes, abstentions):
        - negative_indices: Selected negative index for each anchor (None if abstained or failed)
        - parse_successes: Whether the LM response was successfully parsed
        - abstentions: Whether the LM chose to abstain (return 0)
    """
    if not anchor_indices:
        return [], [], []

    # Separate single-candidate from multi-candidate cases
    negative_indices = []
    parse_successes = []
    abstentions = []
    multi_anchor_indices = []
    multi_positive_indices = []
    multi_candidate_lists = []
    multi_positions = []  # Track original positions for mapping results back

    for i, (anchor_idx, positive_idx, candidate_indices) in enumerate(
        zip(anchor_indices, positive_indices, candidate_indices_by_anchor, strict=False)
    ):
        if len(candidate_indices) == 1:
            # Auto-select the only candidate (no LM judge needed)
            negative_idx = candidate_indices[0]
            # Apply the same logic as in the original code
            if negative_idx == positive_idx and len(candidate_indices) > 1:
                negative_idx = (
                    candidate_indices[1]
                    if candidate_indices[0] == positive_idx
                    else candidate_indices[0]
                )
            negative_indices.append(negative_idx)
            parse_successes.append(True)
            abstentions.append(False)
        else:
            # Multi-candidate case: needs LM judge
            multi_anchor_indices.append(anchor_idx)
            multi_positive_indices.append(positive_idx)
            multi_candidate_lists.append(candidate_indices)
            multi_positions.append(i)
            # Add placeholders for now
            negative_indices.append(None)
            parse_successes.append(False)
            abstentions.append(False)

    # If no multi-candidate cases, return early
    if not multi_anchor_indices:
        num_auto = len(anchor_indices)
        logger.debug(
            f"Auto-selected {num_auto} single-candidate negatives (no LM judge needed)"
        )
        return negative_indices, parse_successes, abstentions

    # Log the optimization
    num_auto = len(anchor_indices) - len(multi_anchor_indices)
    if num_auto > 0:
        logger.debug(
            f"Auto-selected {num_auto} single-candidate negatives, "
            f"calling LM judge for {len(multi_anchor_indices)} multi-candidate cases"
        )

    # Build prompts only for multi-candidate cases
    triplet_example_section = _format_triplet_example_section(
        triplet_example_hint, selection_type="negative"
    )

    # Check if annotations have tags to determine if we should show similarity note
    has_tags = len(annotations) > 0 and (
        "tags" in annotations[0] or "spurious_tags" in annotations[0]
    )
    similarity_note = (
        '- NOTE: Each candidate is annotated with  "Criterion" (similarity on what matters), "Spurious" (similarity on surface features), and "Lexical" (BM25 word overlap)\n'
        if has_tags
        else ""
    )

    anchor_docs = []
    all_images = []  # List of image lists (one per prompt)
    anchor_annotations = []
    positive_docs = []
    positive_annotations = []
    candidates_texts = []

    for anchor_idx, positive_idx, candidate_indices in zip(
        multi_anchor_indices,
        multi_positive_indices,
        multi_candidate_lists,
        strict=False,
    ):
        # Extract anchor text and image
        anchor_doc = documents[anchor_idx]
        anchor_text = extract_text(anchor_doc)
        anchor_image = extract_image(anchor_doc)

        # If document has an image, use <image> placeholder to mark where it should go
        if anchor_image and not anchor_text:
            anchor_text = "<image>"

        anchor_docs.append(anchor_text)

        anchor_ann = annotations[anchor_idx]
        anchor_annotations.append(
            format_annotation_for_display(anchor_ann, include_spurious=True)
        )

        # Extract positive text and image
        positive_doc = documents[positive_idx]
        positive_text = extract_text(positive_doc)
        positive_image = extract_image(positive_doc)

        # If document has an image, use <image> placeholder to mark where it should go
        if positive_image and not positive_text:
            positive_text = "<image>"

        positive_docs.append(positive_text)

        positive_ann = annotations[positive_idx]
        positive_annotations.append(
            format_annotation_for_display(positive_ann, include_spurious=True)
        )

        # Collect all images for this prompt: anchor + positive + candidates
        prompt_images = []
        if anchor_image:
            prompt_images.append(anchor_image)
        if positive_image:
            prompt_images.append(positive_image)

        # Add candidate images
        for cand_idx in candidate_indices:
            cand_image = extract_image(documents[cand_idx])
            if cand_image:
                prompt_images.append(cand_image)

        # Add to all_images (list of lists)
        all_images.append(prompt_images if prompt_images else None)

        bm25_scores = (
            bm25_scores_by_anchor.get(anchor_idx) if bm25_scores_by_anchor else None
        )
        candidates_texts.append(
            _format_candidates_text(
                candidate_indices,
                documents,
                annotations,
                anchor_idx,
                anchor_ann,
                bm25_scores=bm25_scores,
                include_spurious_tags=True,
            )
        )

    inputs = {
        "criterion": [criterion] * len(multi_anchor_indices),
        "criterion_description": [criterion_description] * len(multi_anchor_indices),
        "triplet_example_section": [triplet_example_section]
        * len(multi_anchor_indices),
        "similarity_note": [similarity_note] * len(multi_anchor_indices),
        "anchor_doc": anchor_docs,
        "anchor_annotation": anchor_annotations,
        "positive_doc": positive_docs,
        "positive_annotation": positive_annotations,
        "candidates": candidates_texts,
    }

    # Add images if any prompt has images (list of lists format)
    if any(imgs for imgs in all_images):
        inputs["images"] = all_images

    logger.debug(
        f"Running LM judge for negative selection (batch_size={len(multi_anchor_indices)}, "
        f"cache_alias={cache_alias})"
    )
    results = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )

    # Map LM results back to original positions
    for positive_idx, candidate_indices, selected_num, orig_pos in zip(
        multi_positive_indices,
        multi_candidate_lists,
        results,
        multi_positions,
        strict=False,
    ):
        selected_num = _coerce_selected_num(selected_num)
        if isinstance(selected_num, int) and selected_num == 0:
            # LM chose to abstain
            logger.debug(
                f"LM judge abstained on negative selection for anchor at position {orig_pos}"
            )
            negative_indices[orig_pos] = None
            parse_successes[orig_pos] = True
            abstentions[orig_pos] = True
        elif isinstance(selected_num, int) and 1 <= selected_num <= len(
            candidate_indices
        ):
            negative_idx = candidate_indices[selected_num - 1]
            if negative_idx == positive_idx and len(candidate_indices) > 1:
                negative_idx = (
                    candidate_indices[1]
                    if candidate_indices[0] == positive_idx
                    else candidate_indices[0]
                )
            negative_indices[orig_pos] = negative_idx
            parse_successes[orig_pos] = True
            abstentions[orig_pos] = False
        else:
            logger.error(
                "CRITICAL: Failed to parse negative selection from LM response. "
                f"Got: {selected_num}. Skipping triplet."
            )
            negative_indices[orig_pos] = None
            parse_successes[orig_pos] = False
            abstentions[orig_pos] = False

    return negative_indices, parse_successes, abstentions


def create_lm_triplets(
    documents: list[str],
    annotations: list[dict] | None = None,
    max_triplets: int | None = None,
    candidate_strategy: str = "multi",
    use_spurious_hard_negs: bool = True,
    embedding_preset: str = "hf_qwen3_embedding_8b",
    embedding_preset_overrides: dict | None = None,
    lm_judge_preset: str = "triplet_select_positive_gemini",
    lm_judge_preset_negative: str = "triplet_select_negative_gemini",
    criterion: str | None = None,
    criterion_description: str | None = None,
    cache_alias_prefix: str | None = None,
    triplet_example_hint: dict | str | None = None,
    anchor_indices: list[int] | None = None,
    max_num_candidates: int = 10,
    run_name: str | None = None,
) -> list[tuple[int, int, int]]:
    """Create triplets using language model judge with candidate selection.

    This function:
    1. For each anchor, selects positive candidates (high similarity on summaries/tags)
    2. Uses LM judge to select best positive
    3. Selects negative candidates (high BM25/low Jaccard on docs, spurious hard negs)
    4. Uses LM judge to select best negative (creates hard negatives)

    IMPORTANT: To avoid evaluation bias, embeddings are ONLY used for positive candidate
    selection (on summaries), NOT for negative selection. Negative candidates use BM25,
    Jaccard, and spurious tags to avoid circular dependency with embedding evaluation.

    Args:
        documents: List of document strings
        annotations: Optional list of rich annotation dicts (required for multi strategy)
        max_triplets: Maximum number of triplets to create (None = unlimited)
        candidate_strategy: Strategy for candidate selection:
            - "bm25": BM25 similarity (summaries for pos, raw docs for neg)
            - "embedding": Embedding similarity on summaries for pos, BM25 for neg
            - "jaccard": Jaccard similarity over tags (high for pos, low for neg)
            - "multi": Combine BM25+embedding+Jaccard for pos, BM25+Jaccard+spurious for neg
        use_spurious_hard_negs: DEPRECATED - now ignored, negative strategy handles this
        embedding_preset: Preset for embedding model (used ONLY for positive candidates)
        embedding_preset_overrides: Optional overrides for embedding preset (e.g., custom instruction)
        lm_judge_preset: Preset for LM judge when selecting positives
        lm_judge_preset_negative: Preset for LM judge when selecting negatives
        criterion: Criterion name (for LM judge prompt and instruction-tuned embeddings)
        criterion_description: Description of criterion (for LM judge prompt)
        cache_alias_prefix: Prefix for cache aliases
        triplet_example_hint: Optional example guidance (dict or string)
        anchor_indices: Optional list of document indices to use as anchors.
            If provided, uses these indices as anchors (useful for synthesis).
            If None, uses sequential indices 0, 1, 2, ..., num_triplets-1.
        max_num_candidates: Maximum number of candidates to show LM judge per selection
            (default: 10). For multi strategy, this is the k value per sub-strategy.
        run_name: Optional experiment/run name for cache organization

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets as document indices
    """
    from multiview.benchmark.triplets.candidate_selection import (
        merge_candidate_pools,
        select_candidates_jaccard,
        select_spurious_hard_negatives,
    )

    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Determine anchor indices to use
    if anchor_indices is not None:
        # Use provided anchor indices from synthesis (should be unique remix_anchor_indices)
        anchors_to_process = anchor_indices.copy()

        # If max_triplets is specified and we need MORE triplets than provided anchors,
        # randomly sample additional documents (excluding the provided anchors)
        if max_triplets is not None and len(anchors_to_process) < max_triplets:
            # Calculate how many more we need
            num_additional = max_triplets - len(anchors_to_process)

            # Find all available document indices (excluding anchors already used)
            used_anchors = set(anchors_to_process)
            available_indices = [
                i for i in range(len(documents)) if i not in used_anchors
            ]

            # Randomly sample additional indices (can be original or synthetic docs)
            # Use deterministic sampling for reproducibility
            num_to_add = min(num_additional, len(available_indices))
            if num_to_add > 0:
                additional_anchors = deterministic_sample(
                    available_indices,
                    num_to_add,
                    seed_base="lm_triplets_additional_anchors",
                )
                anchors_to_process.extend(additional_anchors)
                logger.info(
                    f"Extended anchor indices from {len(anchor_indices)} synthesis anchors "
                    f"to {len(anchors_to_process)} total anchors (added {num_to_add} random documents) "
                    f"to meet max_triplets={max_triplets}"
                )
        elif max_triplets is not None and len(anchors_to_process) > max_triplets:
            # If we have too many, limit to max_triplets
            anchors_to_process = anchors_to_process[:max_triplets]
            logger.info(
                f"Limited anchor indices from {len(anchor_indices)} to {len(anchors_to_process)} "
                f"to meet max_triplets={max_triplets}"
            )
    else:
        # Default: sequential indices
        num_triplets = max_triplets if max_triplets is not None else len(documents)
        num_triplets = min(num_triplets, len(documents))
        anchors_to_process = list(range(num_triplets))

    # Log final anchor selection
    logger.info(
        f"Selected {len(anchors_to_process)} anchor documents for triplet creation "
        f"(from {len(documents)} total documents)"
    )

    # Check if annotations are needed
    needs_annotations = candidate_strategy in ["multi", "jaccard", "bm25", "embedding"]
    if needs_annotations and (
        annotations is None or len(annotations) != len(documents)
    ):
        raise ValueError(
            f"Annotations required for candidate_strategy='{candidate_strategy}'"
        )

    triplets = []
    parse_failures = 0

    logger.info(
        f"Creating {len(anchors_to_process)} triplets using LM judge with {candidate_strategy} strategy"
    )

    summary_texts = None
    if candidate_strategy in ["bm25", "multi", "embedding"]:
        summary_texts = [annotation_final_summary(ann) for ann in annotations]

    bm25_summary_matrix = None
    bm25_raw_matrix = None
    if candidate_strategy in ["bm25", "multi", "embedding"]:
        bm25_summary_matrix = compute_bm25_matrix(summary_texts)
        bm25_raw_matrix = compute_bm25_matrix(documents)

    embedding_summary = None
    if candidate_strategy in ["embedding", "multi"]:
        summary_cache_alias = (
            f"{cache_alias_prefix}_embedding_summary" if cache_alias_prefix else None
        )
        # Build inputs - include criterion and description if provided (needed for instruction-tuned embeddings)
        inputs = {"document": summary_texts}
        if criterion is not None:
            inputs["criterion"] = criterion
        # Always add criterion_description if criterion is present (even if empty)
        # This ensures instruction templates can always reference {criterion_description}
        if criterion is not None:
            inputs["criterion_description"] = criterion_description or ""

        # Build kwargs for run_inference
        inference_kwargs = {"verbose": False}
        if embedding_preset_overrides:
            inference_kwargs.update(embedding_preset_overrides)

        summary_embeddings = run_inference(
            inputs=inputs,
            config=embedding_preset,
            cache_alias=summary_cache_alias,
            run_name=run_name,
            **inference_kwargs,
        )
        embedding_summary = np.array(summary_embeddings, dtype=float)
        summary_norms = np.linalg.norm(embedding_summary, axis=1, keepdims=True)
        summary_norms[summary_norms == 0] = 1.0
        embedding_summary = embedding_summary / summary_norms

    # NOTE: We intentionally do NOT compute raw document embeddings for negative
    # candidate selection to avoid evaluation bias. Using the same embedding model
    # for both candidate selection and evaluation would create circular dependency.

    candidate_indices_by_anchor: dict[int, list[int]] = {}
    for anchor_idx in anchors_to_process:
        # logger.debug(
        #     f"Processing anchor {i + 1}/{len(anchors_to_process)} (doc_idx={anchor_idx})"
        # )

        # Step 1: Select candidate pool
        if candidate_strategy == "bm25":
            scores = bm25_summary_matrix[anchor_idx].copy()
            scores[anchor_idx] = -np.inf
            top_k_indices = np.argsort(scores, kind="stable")[::-1][
                : max_num_candidates * 2
            ]
            # Filter out anchor in case corpus size < max_num_candidates * 2
            candidate_indices = [int(idx) for idx in top_k_indices if idx != anchor_idx]

        elif candidate_strategy == "embedding":
            similarities = embedding_summary @ embedding_summary[anchor_idx]
            similarities[anchor_idx] = -np.inf
            top_k_indices = np.argsort(similarities, kind="stable")[::-1][
                : max_num_candidates * 2
            ]
            # Filter out anchor in case corpus size < max_num_candidates * 2
            candidate_indices = [int(idx) for idx in top_k_indices if idx != anchor_idx]

        elif candidate_strategy == "jaccard":
            candidates = select_candidates_jaccard(
                annotations, anchor_idx, k=max_num_candidates * 2, use_spurious=False
            )
            candidate_indices = [idx for idx, score in candidates]

        elif candidate_strategy == "multi":
            bm25_scores = bm25_summary_matrix[anchor_idx].copy()
            bm25_scores[anchor_idx] = -np.inf
            bm25_top_indices = np.argsort(bm25_scores, kind="stable")[::-1][
                :max_num_candidates
            ]
            # Filter out anchor in case corpus size < max_num_candidates
            bm25_top_indices = [idx for idx in bm25_top_indices if idx != anchor_idx]
            bm25_candidates = [
                (int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices
            ]

            emb_scores = embedding_summary @ embedding_summary[anchor_idx]
            emb_scores[anchor_idx] = -np.inf
            emb_top_indices = np.argsort(emb_scores, kind="stable")[::-1][
                :max_num_candidates
            ]
            # Filter out anchor in case corpus size < max_num_candidates
            emb_top_indices = [idx for idx in emb_top_indices if idx != anchor_idx]
            emb_candidates = [
                (int(idx), float(emb_scores[idx])) for idx in emb_top_indices
            ]
            jaccard_candidates = select_candidates_jaccard(
                annotations, anchor_idx, k=max_num_candidates, use_spurious=False
            )

            candidate_indices = merge_candidate_pools(
                bm25_candidates, emb_candidates, jaccard_candidates, use_rrf=True
            )
            candidate_indices = candidate_indices[:max_num_candidates]

        else:
            raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")

        if len(candidate_indices) < 2:
            logger.warning(f"Not enough candidates for anchor {anchor_idx}, skipping")
            continue

        candidate_indices_by_anchor[anchor_idx] = candidate_indices

    anchors_for_pos = list(candidate_indices_by_anchor.keys())
    if not anchors_for_pos:
        return []

    judge_cache_alias = (
        f"{cache_alias_prefix}_judge_pos" if cache_alias_prefix else None
    )

    positive_indices, pos_parse_successes, pos_abstentions = select_positive_batch(
        anchor_indices=anchors_for_pos,
        candidate_indices_by_anchor=[
            candidate_indices_by_anchor[idx] for idx in anchors_for_pos
        ],
        documents=documents,
        annotations=annotations,
        criterion=criterion or "similarity",
        criterion_description=criterion_description
        or criterion
        or "general similarity",
        triplet_example_hint=triplet_example_hint,
        lm_judge_preset=lm_judge_preset,
        bm25_scores_by_anchor=None,
        cache_alias=judge_cache_alias,
        run_name=run_name,
    )

    positive_by_anchor: dict[int, int] = {}
    pos_abstention_count = 0
    for anchor_idx, positive_idx, parse_success, abstained in zip(
        anchors_for_pos,
        positive_indices,
        pos_parse_successes,
        pos_abstentions,
        strict=False,
    ):
        if abstained:
            pos_abstention_count += 1
            continue
        if not parse_success or positive_idx is None:
            parse_failures += 1
            continue
        positive_by_anchor[anchor_idx] = positive_idx

    anchors_for_neg = []
    negative_candidates_by_anchor = []
    positive_indices_for_neg = []

    for anchor_idx in anchors_for_pos:
        positive_idx = positive_by_anchor.get(anchor_idx)
        if positive_idx is None:
            continue

        if candidate_strategy == "bm25":
            scores = bm25_raw_matrix[anchor_idx].copy()
            scores[anchor_idx] = -np.inf
            top_k_indices = np.argsort(scores, kind="stable")[::-1][
                : max_num_candidates * 2
            ]
            # Filter out anchor and positive
            negative_candidate_indices = [
                int(idx)
                for idx in top_k_indices
                if idx != anchor_idx and idx != positive_idx
            ]

        elif candidate_strategy == "embedding":
            # Use BM25 on raw docs for negatives (avoid evaluation bias from embeddings)
            scores = bm25_raw_matrix[anchor_idx].copy()
            scores[anchor_idx] = -np.inf
            top_k_indices = np.argsort(scores, kind="stable")[::-1][
                : max_num_candidates * 2
            ]
            # Filter out anchor and positive
            negative_candidate_indices = [
                int(idx)
                for idx in top_k_indices
                if idx != anchor_idx and idx != positive_idx
            ]

        elif candidate_strategy == "jaccard":
            all_jaccard = select_candidates_jaccard(
                annotations, anchor_idx, k=len(documents) - 1, use_spurious=False
            )
            low_jaccard = list(reversed(all_jaccard))[: max_num_candidates * 2]
            # Filter out positive (anchor already filtered by select_candidates_jaccard)
            negative_candidate_indices = [
                idx for idx, score in low_jaccard if idx != positive_idx
            ]

        elif candidate_strategy == "multi":
            # For negatives, use BM25, Jaccard, and spurious (NO embeddings to avoid bias)
            bm25_scores = bm25_raw_matrix[anchor_idx].copy()
            bm25_scores[anchor_idx] = -np.inf
            bm25_top_indices = np.argsort(bm25_scores, kind="stable")[::-1][
                :max_num_candidates
            ]
            # Filter out anchor in case corpus size < max_num_candidates
            bm25_top_indices = [idx for idx in bm25_top_indices if idx != anchor_idx]
            bm25_neg = [(int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices]

            # Get low Jaccard similarity (dissimilar documents)
            all_jaccard = select_candidates_jaccard(
                annotations, anchor_idx, k=len(documents) - 1, use_spurious=False
            )
            low_jaccard = list(reversed(all_jaccard))[:max_num_candidates]

            # Get spurious hard negatives (high spurious sim, low true sim)
            spurious_negs = select_spurious_hard_negatives(
                annotations, anchor_idx, k=max_num_candidates
            )

            # Merge using RRF (no embedding component for negatives)
            negative_candidate_indices = merge_candidate_pools(
                bm25_neg, low_jaccard, spurious_negs, use_rrf=True
            )
            negative_candidate_indices = [
                idx for idx in negative_candidate_indices if idx != positive_idx
            ]
            negative_candidate_indices = negative_candidate_indices[:max_num_candidates]

        else:
            raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")

        if len(negative_candidate_indices) < 1:
            for idx in range(len(documents)):
                if idx != anchor_idx and idx != positive_idx:
                    negative_candidate_indices = [idx]
                    break

        anchors_for_neg.append(anchor_idx)
        positive_indices_for_neg.append(positive_idx)
        negative_candidates_by_anchor.append(negative_candidate_indices)

    neg_cache_alias = f"{cache_alias_prefix}_judge_neg" if cache_alias_prefix else None

    negative_indices, neg_parse_successes, neg_abstentions = select_negative_batch(
        anchor_indices=anchors_for_neg,
        positive_indices=positive_indices_for_neg,
        candidate_indices_by_anchor=negative_candidates_by_anchor,
        documents=documents,
        annotations=annotations,
        criterion=criterion or "similarity",
        criterion_description=criterion_description
        or criterion
        or "general similarity",
        triplet_example_hint=triplet_example_hint,
        lm_judge_preset=lm_judge_preset_negative,
        bm25_scores_by_anchor=None,
        cache_alias=neg_cache_alias,
        run_name=run_name,
    )

    neg_abstention_count = 0
    for anchor_idx, negative_idx, parse_success, abstained in zip(
        anchors_for_neg,
        negative_indices,
        neg_parse_successes,
        neg_abstentions,
        strict=False,
    ):
        if abstained:
            neg_abstention_count += 1
            continue
        if not parse_success or negative_idx is None:
            parse_failures += 1
            continue

        positive_idx = positive_by_anchor[anchor_idx]

        # Validate all three indices are distinct
        if (
            anchor_idx == positive_idx
            or anchor_idx == negative_idx
            or positive_idx == negative_idx
        ):
            logger.error(
                f"CRITICAL: Skipping invalid triplet with non-distinct indices: "
                f"anchor={anchor_idx}, positive={positive_idx}, negative={negative_idx}"
            )
            parse_failures += 1
            continue

        triplets.append((anchor_idx, positive_idx, negative_idx))

    logger.info(f"Created {len(triplets)} triplets")

    total_abstentions = pos_abstention_count + neg_abstention_count
    total_skipped = parse_failures + total_abstentions

    if total_skipped > 0:
        total_attempted = len(triplets) + total_skipped
        logger.warning("=" * 80)
        logger.warning(
            f"Skipped {total_skipped}/{total_attempted} triplets "
            f"({100*total_skipped/total_attempted:.1f}%)"
        )
        if pos_abstention_count > 0:
            logger.warning(
                f"  - {pos_abstention_count} positive abstentions (LM judge found no clear positive match)"
            )
        if neg_abstention_count > 0:
            logger.warning(
                f"  - {neg_abstention_count} negative abstentions (LM judge found no clear hard negative)"
            )
        if parse_failures > 0:
            logger.error(
                f"  - {parse_failures} parse failures (check logs above for details)"
            )
        logger.warning("=" * 80)

    return triplets


def create_lm_triplets_category(
    documents: list[str],
    annotations: list[dict],
    max_triplets: int | None = None,
    lm_judge_preset: str = "triplet_select_positive_gemini",
    lm_judge_preset_negative: str = "triplet_select_negative_gemini",
    criterion: str | None = None,
    criterion_description: str | None = None,
    cache_alias_prefix: str | None = None,
    triplet_example_hint: dict | str | None = None,
    anchor_indices: list[int] | None = None,
    max_num_candidates: int = 10,
    run_name: str | None = None,
    use_bm25_heuristic: bool = False,
) -> list[tuple[int, int, int]]:
    """Create triplets using category annotations (discrete classification).

    This is optimized for category-based triplets where:
    - Positives: Documents with same category
    - Negatives: Documents with different categories + high BM25 (confusable)

    Args:
        documents: List of document strings
        annotations: List of annotation dicts with "category" field
        max_triplets: Maximum number of triplets to create
        lm_judge_preset: Preset for LM judge when selecting positives
        lm_judge_preset_negative: Preset for LM judge when selecting negatives
        criterion: Criterion name
        criterion_description: Description of criterion
        cache_alias_prefix: Prefix for cache aliases
        triplet_example_hint: Optional example guidance
        anchor_indices: Optional list of anchor document indices
        max_num_candidates: Maximum candidates to show LM judge
        run_name: Optional experiment/run name
        use_bm25_heuristic: If True, skip LM judge and use BM25 heuristic for selection:
            - Positive: Pick lowest BM25 score (hardest match within same category)
            - Negative: Pick highest BM25 score (most confusable across categories)
            This is ~10-100x faster but may differ in quality from LM judge.

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets
    """
    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Build category mapping
    doc_to_category: dict[int, str] = {}
    for idx, ann in enumerate(annotations):
        category = ann.get("category")
        if category is not None:
            doc_to_category[idx] = category

    if not doc_to_category:
        raise ValueError("No category annotations found in annotations")

    # Determine anchors
    if anchor_indices is not None:
        anchors_to_process = anchor_indices.copy()
        if max_triplets is not None and len(anchors_to_process) > max_triplets:
            anchors_to_process = anchors_to_process[:max_triplets]
    else:
        num_triplets = max_triplets if max_triplets is not None else len(documents)
        num_triplets = min(num_triplets, len(documents))
        anchors_to_process = list(range(num_triplets))

    logger.info(
        f"Creating category-based triplets for {len(anchors_to_process)} anchors"
    )

    # Precompute BM25 for negative selection (confusable negatives)
    bm25_matrix = compute_bm25_matrix(documents)

    # First pass: collect positive candidates for each anchor
    candidate_indices_by_anchor: dict[int, list[int]] = {}
    for anchor_idx in anchors_to_process:
        if anchor_idx not in doc_to_category:
            logger.warning(f"Anchor {anchor_idx} has no category, skipping")
            continue

        anchor_category = doc_to_category[anchor_idx]

        # Positive candidates: same category, different document
        pos_candidates = [
            idx
            for idx, cat in doc_to_category.items()
            if idx != anchor_idx and cat == anchor_category
        ]

        if len(pos_candidates) < 1:
            logger.warning(
                f"No positive candidates for anchor {anchor_idx} "
                f"(category: {anchor_category}), skipping"
            )
            continue

        # Limit to max_num_candidates by BM25 score (prefer lower similarity = harder positives)
        if len(pos_candidates) > max_num_candidates:
            scores = bm25_matrix[anchor_idx][pos_candidates]
            sorted_indices = np.argsort(
                scores, kind="stable"
            )  # Lower BM25 = harder positives
            pos_candidates = [
                pos_candidates[i] for i in sorted_indices[:max_num_candidates]
            ]

        candidate_indices_by_anchor[anchor_idx] = pos_candidates

    if not candidate_indices_by_anchor:
        logger.warning("No valid anchors with positive candidates")
        return []

    # Select positives using BM25 heuristic or LM judge
    anchors_for_pos = list(candidate_indices_by_anchor.keys())
    positive_by_anchor: dict[int, int] = {}
    parse_failures = 0
    pos_abstention_count = 0
    neg_abstention_count = 0

    if use_bm25_heuristic:
        # BM25 heuristic: pick lowest BM25 score (hardest match within category)
        # Candidates are already sorted by ascending BM25 at line 1212
        for anchor_idx in anchors_for_pos:
            candidates = candidate_indices_by_anchor[anchor_idx]
            if candidates:
                positive_by_anchor[anchor_idx] = candidates[0]
    else:
        # LM judge selection
        judge_cache_alias = (
            f"{cache_alias_prefix}_judge_pos" if cache_alias_prefix else None
        )

        positive_indices, pos_parse_successes, pos_abstentions = select_positive_batch(
            anchor_indices=anchors_for_pos,
            candidate_indices_by_anchor=[
                candidate_indices_by_anchor[idx] for idx in anchors_for_pos
            ],
            documents=documents,
            annotations=annotations,
            criterion=criterion or "category",
            criterion_description=criterion_description,
            triplet_example_hint=triplet_example_hint,
            lm_judge_preset=lm_judge_preset,
            bm25_scores_by_anchor=None,
            cache_alias=judge_cache_alias,
            run_name=run_name,
        )

        pos_abstention_count = 0
        for anchor_idx, positive_idx, parse_success, abstained in zip(
            anchors_for_pos,
            positive_indices,
            pos_parse_successes,
            pos_abstentions,
            strict=False,
        ):
            if abstained:
                pos_abstention_count += 1
                continue
            if not parse_success or positive_idx is None:
                parse_failures += 1
                continue
            positive_by_anchor[anchor_idx] = positive_idx

    # Second pass: select negative candidates
    anchors_for_neg = []
    negative_candidates_by_anchor = []
    positive_indices_for_neg = []

    for anchor_idx in anchors_for_pos:
        positive_idx = positive_by_anchor.get(anchor_idx)
        if positive_idx is None:
            continue

        anchor_category = doc_to_category[anchor_idx]

        # Negative candidates: different category
        neg_candidates = [
            idx
            for idx, cat in doc_to_category.items()
            if idx != anchor_idx and idx != positive_idx and cat != anchor_category
        ]

        if len(neg_candidates) < 1:
            logger.warning(f"No negative candidates for anchor {anchor_idx}, skipping")
            continue

        # Sort by BM25 score (prefer high similarity = confusable negatives)
        if len(neg_candidates) > max_num_candidates:
            scores = bm25_matrix[anchor_idx][neg_candidates]
            sorted_indices = np.argsort(scores, kind="stable")[
                ::-1
            ]  # Higher BM25 = harder negatives
            neg_candidates = [
                neg_candidates[i] for i in sorted_indices[:max_num_candidates]
            ]

        anchors_for_neg.append(anchor_idx)
        positive_indices_for_neg.append(positive_idx)
        negative_candidates_by_anchor.append(neg_candidates)

    # Select negatives using BM25 heuristic or LM judge
    triplets = []

    if use_bm25_heuristic:
        # BM25 heuristic: pick highest BM25 score (most confusable negative)
        # Candidates are already sorted by descending BM25 at line 1281
        for anchor_idx, positive_idx, neg_candidates in zip(
            anchors_for_neg,
            positive_indices_for_neg,
            negative_candidates_by_anchor,
            strict=False,
        ):
            if not neg_candidates:
                parse_failures += 1
                continue

            # Pick first candidate, but if it equals positive, use second candidate
            negative_idx = neg_candidates[0]
            if negative_idx == positive_idx and len(neg_candidates) > 1:
                negative_idx = neg_candidates[1]

            # Ensure all indices are distinct
            if (
                anchor_idx == positive_idx
                or anchor_idx == negative_idx
                or positive_idx == negative_idx
            ):
                logger.error(
                    f"CRITICAL: Skipping invalid triplet with non-distinct indices: "
                    f"anchor={anchor_idx}, positive={positive_idx}, negative={negative_idx}"
                )
                parse_failures += 1
            else:
                triplets.append((anchor_idx, positive_idx, negative_idx))
    else:
        # LM judge selection
        neg_cache_alias = (
            f"{cache_alias_prefix}_judge_neg" if cache_alias_prefix else None
        )

        negative_indices, neg_parse_successes, neg_abstentions = select_negative_batch(
            anchor_indices=anchors_for_neg,
            positive_indices=positive_indices_for_neg,
            candidate_indices_by_anchor=negative_candidates_by_anchor,
            documents=documents,
            annotations=annotations,
            criterion=criterion or "category",
            criterion_description=criterion_description,
            triplet_example_hint=triplet_example_hint,
            lm_judge_preset=lm_judge_preset_negative,
            bm25_scores_by_anchor=None,
            cache_alias=neg_cache_alias,
            run_name=run_name,
        )

        for anchor_idx, negative_idx, parse_success, abstained in zip(
            anchors_for_neg,
            negative_indices,
            neg_parse_successes,
            neg_abstentions,
            strict=False,
        ):
            if abstained:
                neg_abstention_count += 1
                continue
            if not parse_success or negative_idx is None:
                parse_failures += 1
                continue
            triplets.append((anchor_idx, positive_by_anchor[anchor_idx], negative_idx))

    selection_mode = "BM25 heuristic" if use_bm25_heuristic else "LM judge"
    logger.info(
        f"Created {len(triplets)} category-based triplets using {selection_mode}"
    )

    total_abstentions = pos_abstention_count + neg_abstention_count
    total_skipped = parse_failures + total_abstentions

    if total_skipped > 0:
        total_attempted = len(triplets) + total_skipped
        logger.warning("=" * 80)
        logger.warning(
            f"Skipped {total_skipped}/{total_attempted} triplets "
            f"({100*total_skipped/total_attempted:.1f}%)"
        )
        if pos_abstention_count > 0:
            logger.warning(
                f"  - {pos_abstention_count} positive abstentions (LM judge found no clear positive match)"
            )
        if neg_abstention_count > 0:
            logger.warning(
                f"  - {neg_abstention_count} negative abstentions (LM judge found no clear hard negative)"
            )
        if parse_failures > 0:
            failure_type = (
                "selection failures" if use_bm25_heuristic else "parse failures"
            )
            logger.error(f"  - {parse_failures} {failure_type}")
        logger.warning("=" * 80)

    return triplets


def create_lm_triplets_tags(
    documents: list[str],
    annotations: list[dict],
    max_triplets: int | None = None,
    lm_judge_preset: str = "triplet_select_positive_gemini",
    lm_judge_preset_negative: str = "triplet_select_negative_gemini",
    criterion: str | None = None,
    criterion_description: str | None = None,
    cache_alias_prefix: str | None = None,
    triplet_example_hint: dict | str | None = None,
    anchor_indices: list[int] | None = None,
    max_num_candidates: int = 10,
    run_name: str | None = None,
) -> list[tuple[int, int, int]]:
    """Create triplets using tag annotations (multi-label tags).

    This is optimized for tag-based triplets where:
    - Positives: High Jaccard similarity on tags (+ BM25 for hard selection)
    - Negatives: Low tag Jaccard + high spurious Jaccard (spurious hard negatives)

    Args:
        documents: List of document strings
        annotations: List of annotation dicts with "tags" and "spurious_tags" fields
        max_triplets: Maximum number of triplets to create
        lm_judge_preset: Preset for LM judge when selecting positives
        lm_judge_preset_negative: Preset for LM judge when selecting negatives
        criterion: Criterion name
        criterion_description: Description of criterion
        cache_alias_prefix: Prefix for cache aliases
        triplet_example_hint: Optional example guidance
        anchor_indices: Optional list of anchor document indices
        max_num_candidates: Maximum candidates to show LM judge
        run_name: Optional experiment/run name

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets
    """
    from multiview.benchmark.triplets.candidate_selection import (
        merge_candidate_pools,
        select_candidates_jaccard,
        select_spurious_hard_negatives,
    )

    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Determine anchors
    if anchor_indices is not None:
        anchors_to_process = anchor_indices.copy()
        if max_triplets is not None and len(anchors_to_process) > max_triplets:
            anchors_to_process = anchors_to_process[:max_triplets]
    else:
        num_triplets = max_triplets if max_triplets is not None else len(documents)
        num_triplets = min(num_triplets, len(documents))
        anchors_to_process = list(range(num_triplets))

    logger.info(f"Creating tag-based triplets for {len(anchors_to_process)} anchors")

    # Precompute BM25 for hard positive selection
    bm25_matrix = compute_bm25_matrix(documents)

    # First pass: collect positive candidates using Jaccard + BM25
    candidate_indices_by_anchor: dict[int, list[int]] = {}
    for anchor_idx in anchors_to_process:
        # Use multi-strategy: Jaccard + BM25
        jaccard_candidates = select_candidates_jaccard(
            annotations, anchor_idx, k=max_num_candidates, use_spurious=False
        )

        bm25_scores = bm25_matrix[anchor_idx].copy()
        bm25_scores[anchor_idx] = -np.inf
        bm25_top_indices = np.argsort(bm25_scores, kind="stable")[:max_num_candidates]
        # Filter out anchor in case corpus size < max_num_candidates
        bm25_top_indices = [idx for idx in bm25_top_indices if idx != anchor_idx]
        bm25_candidates = [
            (int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices
        ]

        # Merge using RRF
        candidate_indices = merge_candidate_pools(
            bm25_candidates, jaccard_candidates, use_rrf=True
        )
        candidate_indices = candidate_indices[:max_num_candidates]

        if len(candidate_indices) < 1:
            logger.warning(f"No positive candidates for anchor {anchor_idx}, skipping")
            continue

        candidate_indices_by_anchor[anchor_idx] = candidate_indices

    if not candidate_indices_by_anchor:
        logger.warning("No valid anchors with positive candidates")
        return []

    # Select positives using LM judge
    anchors_for_pos = list(candidate_indices_by_anchor.keys())
    judge_cache_alias = (
        f"{cache_alias_prefix}_judge_pos" if cache_alias_prefix else None
    )
    logger.debug(f"Selecting positives with {lm_judge_preset=}...")
    positive_indices, pos_parse_successes, pos_abstentions = select_positive_batch(
        anchor_indices=anchors_for_pos,
        candidate_indices_by_anchor=[
            candidate_indices_by_anchor[idx] for idx in anchors_for_pos
        ],
        documents=documents,
        annotations=annotations,
        criterion=criterion or "tags",
        criterion_description=criterion_description,
        triplet_example_hint=triplet_example_hint,
        lm_judge_preset=lm_judge_preset,
        bm25_scores_by_anchor=None,
        cache_alias=judge_cache_alias,
        run_name=run_name,
    )

    positive_by_anchor: dict[int, int] = {}
    parse_failures = 0
    pos_abstention_count = 0
    for anchor_idx, positive_idx, parse_success, abstained in zip(
        anchors_for_pos,
        positive_indices,
        pos_parse_successes,
        pos_abstentions,
        strict=False,
    ):
        if abstained:
            pos_abstention_count += 1
            continue
        if not parse_success or positive_idx is None:
            parse_failures += 1
            continue
        positive_by_anchor[anchor_idx] = positive_idx

    # Second pass: select negative candidates using spurious hard negatives
    anchors_for_neg = []
    negative_candidates_by_anchor = []
    positive_indices_for_neg = []

    for anchor_idx in anchors_for_pos:
        positive_idx = positive_by_anchor.get(anchor_idx)
        if positive_idx is None:
            continue

        # Get low Jaccard similarity (dissimilar on tags)
        all_jaccard = select_candidates_jaccard(
            annotations, anchor_idx, k=len(documents) - 1, use_spurious=False
        )
        low_jaccard = list(reversed(all_jaccard))[:max_num_candidates]

        # Get spurious hard negatives (high spurious sim, low true sim)
        spurious_negs = select_spurious_hard_negatives(
            annotations, anchor_idx, k=max_num_candidates
        )

        # Get BM25 negatives
        bm25_scores = bm25_matrix[anchor_idx].copy()
        bm25_scores[anchor_idx] = -np.inf
        bm25_top_indices = np.argsort(bm25_scores, kind="stable")[::-1][
            :max_num_candidates
        ]
        # Filter out anchor in case corpus size < max_num_candidates
        bm25_top_indices = [idx for idx in bm25_top_indices if idx != anchor_idx]
        bm25_neg = [(int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices]

        # Merge using RRF
        negative_candidate_indices = merge_candidate_pools(
            bm25_neg, low_jaccard, spurious_negs, use_rrf=True
        )
        negative_candidate_indices = [
            idx for idx in negative_candidate_indices if idx != positive_idx
        ]
        negative_candidate_indices = negative_candidate_indices[:max_num_candidates]

        if len(negative_candidate_indices) < 1:
            logger.warning(f"No negative candidates for anchor {anchor_idx}, skipping")
            continue

        anchors_for_neg.append(anchor_idx)
        positive_indices_for_neg.append(positive_idx)
        negative_candidates_by_anchor.append(negative_candidate_indices)

    # Select negatives using LM judge
    neg_cache_alias = f"{cache_alias_prefix}_judge_neg" if cache_alias_prefix else None

    logger.debug(f"Selecting negatives with {lm_judge_preset_negative=}...")
    negative_indices, neg_parse_successes, neg_abstentions = select_negative_batch(
        anchor_indices=anchors_for_neg,
        positive_indices=positive_indices_for_neg,
        candidate_indices_by_anchor=negative_candidates_by_anchor,
        documents=documents,
        annotations=annotations,
        criterion=criterion or "tags",
        criterion_description=criterion_description,
        triplet_example_hint=triplet_example_hint,
        lm_judge_preset=lm_judge_preset_negative,
        bm25_scores_by_anchor=None,
        cache_alias=neg_cache_alias,
        run_name=run_name,
    )

    # Build final triplets
    triplets = []
    neg_abstention_count = 0
    for anchor_idx, negative_idx, parse_success, abstained in zip(
        anchors_for_neg,
        negative_indices,
        neg_parse_successes,
        neg_abstentions,
        strict=False,
    ):
        if abstained:
            neg_abstention_count += 1
            continue
        if not parse_success or negative_idx is None:
            parse_failures += 1
            continue

        positive_idx = positive_by_anchor[anchor_idx]

        # Validate all three indices are distinct
        if (
            anchor_idx == positive_idx
            or anchor_idx == negative_idx
            or positive_idx == negative_idx
        ):
            logger.error(
                f"CRITICAL: Skipping invalid triplet with non-distinct indices: "
                f"anchor={anchor_idx}, positive={positive_idx}, negative={negative_idx}"
            )
            parse_failures += 1
            continue

        triplets.append((anchor_idx, positive_idx, negative_idx))

    logger.info(f"Created {len(triplets)} tag-based triplets")

    total_abstentions = pos_abstention_count + neg_abstention_count
    total_skipped = parse_failures + total_abstentions

    if total_skipped > 0:
        total_attempted = len(triplets) + total_skipped
        logger.warning("=" * 80)
        logger.warning(
            f"Skipped {total_skipped}/{total_attempted} triplets "
            f"({100*total_skipped/total_attempted:.1f}%)"
        )
        if pos_abstention_count > 0:
            logger.warning(
                f"  - {pos_abstention_count} positive abstentions (LM judge found no clear positive match)"
            )
        if neg_abstention_count > 0:
            logger.warning(
                f"  - {neg_abstention_count} negative abstentions (LM judge found no clear hard negative)"
            )
        if parse_failures > 0:
            logger.error(
                f"  - {parse_failures} parse failures (check logs above for details)"
            )
        logger.warning("=" * 80)

    return triplets


def create_lm_triplets_summary_dict(
    documents: list[str],
    annotations: list[dict],
    max_triplets: int | None = None,
    embedding_preset: str = "hf_qwen3_embedding_8b",
    embedding_preset_overrides: dict | None = None,
    lm_judge_preset: str = "triplet_select_positive_gemini",
    lm_judge_preset_negative: str = "triplet_select_negative_gemini",
    criterion: str | None = None,
    criterion_description: str | None = None,
    cache_alias_prefix: str | None = None,
    triplet_example_hint: dict | str | None = None,
    anchor_indices: list[int] | None = None,
    max_num_candidates: int = 10,
    run_name: str | None = None,
) -> list[tuple[int, int, int]]:
    """Create triplets using dictionary/structured summary annotations.

    This is optimized for summary-based triplets where:
    - Positives: High similarity on summaries (BM25 + embeddings)
    - Negatives: High BM25 on raw docs, low on summaries (semantic hard negatives)

    Args:
        documents: List of document strings
        annotations: List of annotation dicts with "summary" field
        max_triplets: Maximum number of triplets to create
        embedding_preset: Preset for embedding model (on summaries)
        embedding_preset_overrides: Optional overrides for embedding preset
        lm_judge_preset: Preset for LM judge when selecting positives
        lm_judge_preset_negative: Preset for LM judge when selecting negatives
        criterion: Criterion name
        criterion_description: Description of criterion
        cache_alias_prefix: Prefix for cache aliases
        triplet_example_hint: Optional example guidance
        anchor_indices: Optional list of anchor document indices
        max_num_candidates: Maximum candidates to show LM judge
        run_name: Optional experiment/run name

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets
    """
    from multiview.benchmark.triplets.candidate_selection import merge_candidate_pools

    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Determine anchors
    if anchor_indices is not None:
        anchors_to_process = anchor_indices.copy()
        if max_triplets is not None and len(anchors_to_process) > max_triplets:
            anchors_to_process = anchors_to_process[:max_triplets]
    else:
        num_triplets = max_triplets if max_triplets is not None else len(documents)
        num_triplets = min(num_triplets, len(documents))
        anchors_to_process = list(range(num_triplets))

    logger.info(
        f"Creating summary-based triplets for {len(anchors_to_process)} anchors"
    )

    # Extract summary texts
    summary_texts = [annotation_final_summary(ann) for ann in annotations]

    # Precompute BM25 matrices
    bm25_summary_matrix = compute_bm25_matrix(summary_texts)
    bm25_raw_matrix = compute_bm25_matrix(documents)

    # Compute embeddings on summaries
    summary_cache_alias = (
        f"{cache_alias_prefix}_embedding_summary" if cache_alias_prefix else None
    )

    inputs = {"document": summary_texts}
    if criterion is not None:
        inputs["criterion"] = criterion
    # Always add criterion_description if criterion is present (even if empty)
    # This ensures instruction templates can always reference {criterion_description}
    if criterion is not None:
        inputs["criterion_description"] = criterion_description or ""

    inference_kwargs = {"verbose": False}
    if embedding_preset_overrides:
        inference_kwargs.update(embedding_preset_overrides)

    summary_embeddings = run_inference(
        inputs=inputs,
        config=embedding_preset,
        cache_alias=summary_cache_alias,
        run_name=run_name,
        **inference_kwargs,
    )
    embedding_summary = np.array(summary_embeddings, dtype=float)
    summary_norms = np.linalg.norm(embedding_summary, axis=1, keepdims=True)
    summary_norms[summary_norms == 0] = 1.0
    embedding_summary = embedding_summary / summary_norms

    # First pass: collect positive candidates using BM25 + embeddings on summaries
    candidate_indices_by_anchor: dict[int, list[int]] = {}
    for anchor_idx in anchors_to_process:
        # BM25 on summaries
        bm25_scores = bm25_summary_matrix[anchor_idx].copy()
        bm25_scores[anchor_idx] = -np.inf
        bm25_top_indices = np.argsort(bm25_scores, kind="stable")[::-1][
            :max_num_candidates
        ]
        # Filter out anchor in case corpus size < max_num_candidates
        bm25_top_indices = [idx for idx in bm25_top_indices if idx != anchor_idx]
        bm25_candidates = [
            (int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices
        ]

        # Embeddings on summaries
        emb_scores = embedding_summary @ embedding_summary[anchor_idx]
        emb_scores[anchor_idx] = -np.inf
        emb_top_indices = np.argsort(emb_scores, kind="stable")[::-1][
            :max_num_candidates
        ]
        # Filter out anchor in case corpus size < max_num_candidates
        emb_top_indices = [idx for idx in emb_top_indices if idx != anchor_idx]
        emb_candidates = [(int(idx), float(emb_scores[idx])) for idx in emb_top_indices]

        # Merge using RRF
        candidate_indices = merge_candidate_pools(
            bm25_candidates, emb_candidates, use_rrf=True
        )
        candidate_indices = candidate_indices[:max_num_candidates]

        if len(candidate_indices) < 1:
            logger.warning(f"No positive candidates for anchor {anchor_idx}, skipping")
            continue

        candidate_indices_by_anchor[anchor_idx] = candidate_indices

    if not candidate_indices_by_anchor:
        logger.warning("No valid anchors with positive candidates")
        return []

    # Select positives using LM judge
    anchors_for_pos = list(candidate_indices_by_anchor.keys())
    judge_cache_alias = (
        f"{cache_alias_prefix}_judge_pos" if cache_alias_prefix else None
    )

    positive_indices, pos_parse_successes, pos_abstentions = select_positive_batch(
        anchor_indices=anchors_for_pos,
        candidate_indices_by_anchor=[
            candidate_indices_by_anchor[idx] for idx in anchors_for_pos
        ],
        documents=documents,
        annotations=annotations,
        criterion=criterion or "summary",
        criterion_description=criterion_description,
        triplet_example_hint=triplet_example_hint,
        lm_judge_preset=lm_judge_preset,
        bm25_scores_by_anchor=None,
        cache_alias=judge_cache_alias,
        run_name=run_name,
    )

    positive_by_anchor: dict[int, int] = {}
    parse_failures = 0
    pos_abstention_count = 0
    for anchor_idx, positive_idx, parse_success, abstained in zip(
        anchors_for_pos,
        positive_indices,
        pos_parse_successes,
        pos_abstentions,
        strict=False,
    ):
        if abstained:
            pos_abstention_count += 1
            continue
        if not parse_success or positive_idx is None:
            parse_failures += 1
            continue
        positive_by_anchor[anchor_idx] = positive_idx

    # Second pass: select negative candidates (high BM25 on raw docs = semantic hard negatives)
    anchors_for_neg = []
    negative_candidates_by_anchor = []
    positive_indices_for_neg = []

    for anchor_idx in anchors_for_pos:
        positive_idx = positive_by_anchor.get(anchor_idx)
        if positive_idx is None:
            continue

        # Use BM25 on raw docs for negatives (high text similarity but semantically different)
        bm25_scores = bm25_raw_matrix[anchor_idx].copy()
        bm25_scores[anchor_idx] = -np.inf
        bm25_top_indices = np.argsort(bm25_scores, kind="stable")[::-1][
            : max_num_candidates * 2
        ]
        # Filter out anchor (in case corpus size < max_num_candidates * 2) and positive
        negative_candidate_indices = [
            int(idx)
            for idx in bm25_top_indices
            if idx != anchor_idx and idx != positive_idx
        ]
        negative_candidate_indices = negative_candidate_indices[:max_num_candidates]

        if len(negative_candidate_indices) < 1:
            logger.warning(f"No negative candidates for anchor {anchor_idx}, skipping")
            continue

        anchors_for_neg.append(anchor_idx)
        positive_indices_for_neg.append(positive_idx)
        negative_candidates_by_anchor.append(negative_candidate_indices)

    # Select negatives using LM judge
    neg_cache_alias = f"{cache_alias_prefix}_judge_neg" if cache_alias_prefix else None

    negative_indices, neg_parse_successes, neg_abstentions = select_negative_batch(
        anchor_indices=anchors_for_neg,
        positive_indices=positive_indices_for_neg,
        candidate_indices_by_anchor=negative_candidates_by_anchor,
        documents=documents,
        annotations=annotations,
        criterion=criterion or "summary",
        criterion_description=criterion_description,
        triplet_example_hint=triplet_example_hint,
        lm_judge_preset=lm_judge_preset_negative,
        bm25_scores_by_anchor=None,
        cache_alias=neg_cache_alias,
        run_name=run_name,
    )

    # Build final triplets
    triplets = []
    neg_abstention_count = 0
    for anchor_idx, negative_idx, parse_success, abstained in zip(
        anchors_for_neg,
        negative_indices,
        neg_parse_successes,
        neg_abstentions,
        strict=False,
    ):
        if abstained:
            neg_abstention_count += 1
            continue
        if not parse_success or negative_idx is None:
            parse_failures += 1
            continue

        positive_idx = positive_by_anchor[anchor_idx]

        # Validate all three indices are distinct
        if (
            anchor_idx == positive_idx
            or anchor_idx == negative_idx
            or positive_idx == negative_idx
        ):
            logger.error(
                f"CRITICAL: Skipping invalid triplet with non-distinct indices: "
                f"anchor={anchor_idx}, positive={positive_idx}, negative={negative_idx}"
            )
            parse_failures += 1
            continue

        triplets.append((anchor_idx, positive_idx, negative_idx))

    logger.info(f"Created {len(triplets)} summary-based triplets")

    total_abstentions = pos_abstention_count + neg_abstention_count
    total_skipped = parse_failures + total_abstentions

    if total_skipped > 0:
        total_attempted = len(triplets) + total_skipped
        logger.warning("=" * 80)
        logger.warning(
            f"Skipped {total_skipped}/{total_attempted} triplets "
            f"({100*total_skipped/total_attempted:.1f}%)"
        )
        if pos_abstention_count > 0:
            logger.warning(
                f"  - {pos_abstention_count} positive abstentions (LM judge found no clear positive match)"
            )
        if neg_abstention_count > 0:
            logger.warning(
                f"  - {neg_abstention_count} negative abstentions (LM judge found no clear hard negative)"
            )
        if parse_failures > 0:
            logger.error(
                f"  - {parse_failures} parse failures (check logs above for details)"
            )
        logger.warning("=" * 80)

    return triplets


def create_lm_triplets_summary_sentence(
    documents: list[str],
    annotations: list[dict],
    max_triplets: int | None = None,
    embedding_preset: str = "hf_qwen3_embedding_8b",
    embedding_preset_overrides: dict | None = None,
    lm_judge_preset: str = "triplet_select_positive_gemini",
    lm_judge_preset_negative: str = "triplet_select_negative_gemini",
    criterion: str | None = None,
    criterion_description: str | None = None,
    cache_alias_prefix: str | None = None,
    triplet_example_hint: dict | str | None = None,
    anchor_indices: list[int] | None = None,
    max_num_candidates: int = 10,
    run_name: str | None = None,
) -> list[tuple[int, int, int]]:
    """Create triplets using one-sentence summaries.

    This is an alias to create_lm_triplets_summary_dict() - the triplet selection
    logic is identical regardless of summary format.

    Args:
        documents: List of document strings
        annotations: List of annotation dicts with "summary" field (sentence format)
        max_triplets: Maximum number of triplets to create
        embedding_preset: Preset for embedding model (on summaries)
        embedding_preset_overrides: Optional overrides for embedding preset
        lm_judge_preset: Preset for LM judge when selecting positives
        lm_judge_preset_negative: Preset for LM judge when selecting negatives
        criterion: Criterion name
        criterion_description: Description of criterion
        cache_alias_prefix: Prefix for cache aliases
        triplet_example_hint: Optional example guidance
        anchor_indices: Optional list of anchor document indices
        max_num_candidates: Maximum candidates to show LM judge
        run_name: Optional experiment/run name

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets
    """
    return create_lm_triplets_summary_dict(
        documents=documents,
        annotations=annotations,
        max_triplets=max_triplets,
        embedding_preset=embedding_preset,
        embedding_preset_overrides=embedding_preset_overrides,
        lm_judge_preset=lm_judge_preset,
        lm_judge_preset_negative=lm_judge_preset_negative,
        criterion=criterion,
        criterion_description=criterion_description,
        cache_alias_prefix=cache_alias_prefix,
        triplet_example_hint=triplet_example_hint,
        anchor_indices=anchor_indices,
        max_num_candidates=max_num_candidates,
        run_name=run_name,
    )
