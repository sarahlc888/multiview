"""Utilities for triplet creation and sampling.

This module contains the main triplet creation logic:
    - create_random_triplets(): Simple random triplet sampling
    - create_lm_triplets(): LM-based triplet creation with candidate selection
      * Positive candidates: high similarity on summaries/tags (BM25/embedding/Jaccard)
      * Negative candidates: high similarity on raw docs, low similarity on tags (hard negatives)
      * Uses two-stage LM judge: select_positive() then select_negative()
    - select_positive(): LM judge that selects positive from candidates
    - select_negative(): LM judge that selects negative from candidates (sees positive)

Related modules:
    - candidate_selection.py: BM25, embedding, Jaccard retrieval strategies
    - utils.py: Helper functions (jaccard_similarity, format_annotation_for_display, etc.)
"""

import logging

from multiview.benchmark.triplets.utils import (
    extract_active_tags,
    format_annotation_for_display,
    jaccard_similarity,
)
from multiview.inference.inference import run_inference
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


def _format_candidates_text(
    candidate_indices: list[int],
    documents: list[str],
    annotations: list[dict],
    anchor_idx: int,
    anchor_ann: dict,
    include_spurious_tags: bool = False,
) -> str:
    """Format candidates with similarity scores for display.

    Args:
        candidate_indices: List of candidate document indices
        documents: List of all documents
        annotations: List of all annotations
        anchor_idx: Index of anchor document (for BM25 computation)
        anchor_ann: Anchor document annotation
        include_spurious_tags: If True, include spurious tags in annotations

    Returns:
        Formatted candidates text
    """
    from multiview.utils.bm25_utils import compute_bm25_scores

    anchor_tags = extract_active_tags(anchor_ann, "tags")
    anchor_spurious = extract_active_tags(anchor_ann, "spurious_tags")

    # Compute BM25 lexical similarity scores for all documents
    bm25_scores = compute_bm25_scores(documents, anchor_idx)

    candidates_text_parts = []
    for i, cand_idx in enumerate(candidate_indices):
        cand = documents[cand_idx]
        cand_ann = annotations[cand_idx]

        # Compute true and spurious Jaccard similarity
        cand_tags = extract_active_tags(cand_ann, "tags")
        cand_spurious = extract_active_tags(cand_ann, "spurious_tags")

        true_sim = jaccard_similarity(anchor_tags, cand_tags)
        spurious_sim = jaccard_similarity(anchor_spurious, cand_spurious)

        # Get BM25 lexical similarity score
        lexical_sim = bm25_scores[cand_idx]

        cand_annotation = format_annotation_for_display(
            cand_ann, include_spurious=include_spurious_tags
        )

        cand_text = f"[Document {i+1}]\n{cand}\n\n"
        cand_text += f"[Annotation {i+1}]\n{cand_annotation}\n"
        cand_text += f"Similarity to anchor: Criterion={true_sim:.2f} | Spurious={spurious_sim:.2f} | Lexical={lexical_sim:.2f}"

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


def select_positive(
    anchor_idx: int,
    candidate_indices: list[int],
    documents: list[str],
    annotations: list[dict],
    criterion: str,
    criterion_description: str,
    triplet_example_hint: dict | str | None = None,
    lm_judge_preset: str = "triplet_select_positive_gemini",
    cache_alias: str | None = None,
) -> tuple[int, bool]:
    """Use LM judge to select positive from candidates.

    Args:
        anchor_idx: Index of anchor document
        candidate_indices: List of candidate document indices
        documents: List of all documents
        annotations: List of all annotations
        criterion: Criterion name
        criterion_description: Criterion description
        triplet_example_hint: Optional example guidance (dict or string)
        lm_judge_preset: Preset for LM judge
        cache_alias: Cache alias for LM calls

    Returns:
        Tuple of (positive_idx, parse_success)
    """
    if len(candidate_indices) < 1:
        return candidate_indices[0] if candidate_indices else anchor_idx, False

    # Get anchor info
    anchor_doc = documents[anchor_idx]
    anchor_ann = annotations[anchor_idx]
    anchor_annotation = format_annotation_for_display(anchor_ann)

    # Format candidates
    candidates_text = _format_candidates_text(
        candidate_indices, documents, annotations, anchor_idx, anchor_ann
    )

    # Format triplet example section
    triplet_example_section = _format_triplet_example_section(
        triplet_example_hint, selection_type="positive"
    )

    # Prepare inputs for LM judge
    inputs = {
        "criterion": [criterion],
        "criterion_description": [criterion_description or criterion],
        "triplet_example_section": [triplet_example_section],
        "anchor_doc": [anchor_doc],
        "anchor_annotation": [anchor_annotation],
        "candidates": [candidates_text],
    }

    # Run LM judge (preset handles JSON parsing and field extraction)
    logger.debug(
        f"Running LM judge for positive selection (anchor_idx={anchor_idx}, "
        f"candidates={len(candidate_indices)}, cache_alias={cache_alias})"
    )
    results = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        verbose=False,
    )

    # Get parsed selection number (1-indexed)
    selected_num = results[0] if results else None

    # Handle both int and list formats (e.g., 5 or [5])
    if isinstance(selected_num, list) and len(selected_num) == 1:
        selected_num = selected_num[0]

    # Convert to document index
    if isinstance(selected_num, int) and 1 <= selected_num <= len(candidate_indices):
        positive_idx = candidate_indices[selected_num - 1]
        return positive_idx, True
    else:
        logger.error(
            f"CRITICAL: Failed to parse positive selection from LM response. "
            f"Got: {selected_num}. Skipping triplet."
        )
        return None, False


def select_negative(
    anchor_idx: int,
    positive_idx: int,
    candidate_indices: list[int],
    documents: list[str],
    annotations: list[dict],
    criterion: str,
    criterion_description: str,
    triplet_example_hint: dict | str | None = None,
    lm_judge_preset: str = "triplet_select_negative_gemini",
    cache_alias: str | None = None,
) -> tuple[int, bool]:
    """Use LM judge to select negative from candidates.

    Args:
        anchor_idx: Index of anchor document
        positive_idx: Index of already-selected positive document
        candidate_indices: List of candidate document indices (should exclude positive)
        documents: List of all documents
        annotations: List of all annotations
        criterion: Criterion name
        criterion_description: Criterion description
        triplet_example_hint: Optional example guidance (dict or string)
        lm_judge_preset: Preset for LM judge
        cache_alias: Cache alias for LM calls

    Returns:
        Tuple of (negative_idx, parse_success)
    """
    if len(candidate_indices) < 1:
        # Need at least one candidate for negative
        # Find any document that isn't anchor or positive
        for idx in range(len(documents)):
            if idx != anchor_idx and idx != positive_idx:
                return idx, False
        return anchor_idx, False

    # Get anchor and positive info
    anchor_doc = documents[anchor_idx]
    anchor_ann = annotations[anchor_idx]
    # Include spurious tags for negative selection to help identify hard negatives
    anchor_annotation = format_annotation_for_display(anchor_ann, include_spurious=True)

    positive_doc = documents[positive_idx]
    positive_ann = annotations[positive_idx]
    positive_annotation = format_annotation_for_display(
        positive_ann, include_spurious=True
    )

    # Format candidates with spurious tags visible
    candidates_text = _format_candidates_text(
        candidate_indices,
        documents,
        annotations,
        anchor_idx,
        anchor_ann,
        include_spurious_tags=True,
    )

    # Format triplet example section
    triplet_example_section = _format_triplet_example_section(
        triplet_example_hint, selection_type="negative"
    )

    # Prepare inputs for LM judge
    inputs = {
        "criterion": [criterion],
        "criterion_description": [criterion_description or criterion],
        "triplet_example_section": [triplet_example_section],
        "anchor_doc": [anchor_doc],
        "anchor_annotation": [anchor_annotation],
        "positive_doc": [positive_doc],
        "positive_annotation": [positive_annotation],
        "candidates": [candidates_text],
    }

    # Run LM judge (preset handles JSON parsing and field extraction)
    logger.debug(
        f"Running LM judge for negative selection (anchor_idx={anchor_idx}, "
        f"positive_idx={positive_idx}, candidates={len(candidate_indices)}, "
        f"cache_alias={cache_alias})"
    )
    results = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        verbose=False,
    )

    # Get parsed selection number (1-indexed)
    selected_num = results[0] if results else None

    # Handle both int and list formats (e.g., 5 or [5])
    if isinstance(selected_num, list) and len(selected_num) == 1:
        selected_num = selected_num[0]

    # Convert to document index
    if isinstance(selected_num, int) and 1 <= selected_num <= len(candidate_indices):
        negative_idx = candidate_indices[selected_num - 1]
        # Ensure negative is different from positive
        if negative_idx == positive_idx and len(candidate_indices) > 1:
            negative_idx = (
                candidate_indices[1]
                if candidate_indices[0] == positive_idx
                else candidate_indices[0]
            )
        return negative_idx, True
    else:
        logger.error(
            f"CRITICAL: Failed to parse negative selection from LM response. "
            f"Got: {selected_num}. Skipping triplet."
        )
        return None, False


def create_lm_triplets(
    documents: list[str],
    annotations: list[dict] | None = None,
    max_triplets: int | None = None,
    candidate_strategy: str = "multi",
    use_spurious_hard_negs: bool = True,
    embedding_preset: str = "hf_qwen3_embedding_8b",
    lm_judge_preset: str = "triplet_select_positive_gemini",
    lm_judge_preset_negative: str = "triplet_select_negative_gemini",
    criterion: str | None = None,
    criterion_description: str | None = None,
    cache_alias_prefix: str | None = None,
    triplet_example_hint: dict | str | None = None,
    anchor_indices: list[int] | None = None,
    max_num_candidates: int = 10,
) -> list[tuple[int, int, int]]:
    """Create triplets using language model judge with candidate selection.

    This function:
    1. For each anchor, selects positive candidates (high similarity on summaries/tags)
    2. Uses LM judge to select best positive
    3. Selects negative candidates (high similarity on raw docs, low similarity on tags)
    4. Uses LM judge to select best negative (creates hard negatives)

    Args:
        documents: List of document strings
        annotations: Optional list of rich annotation dicts (required for multi strategy)
        max_triplets: Maximum number of triplets to create (None = unlimited)
        candidate_strategy: Strategy for candidate selection:
            - "bm25": BM25 similarity (summaries for pos, raw docs for neg)
            - "embedding": Embedding similarity (summaries for pos, raw docs for neg)
            - "jaccard": Jaccard similarity over tags (high for pos, low for neg)
            - "multi": Combine all strategies
        use_spurious_hard_negs: DEPRECATED - now ignored, negative strategy handles this
        embedding_preset: Preset for embedding model (if using embedding strategy)
        lm_judge_preset: Preset for LM judge when selecting positives
        lm_judge_preset_negative: Preset for LM judge when selecting negatives
        criterion: Criterion name (for LM judge prompt)
        criterion_description: Description of criterion (for LM judge prompt)
        cache_alias_prefix: Prefix for cache aliases
        triplet_example_hint: Optional example guidance (dict or string)
        anchor_indices: Optional list of document indices to use as anchors.
            If provided, uses these indices as anchors (useful for synthesis).
            If None, uses sequential indices 0, 1, 2, ..., num_triplets-1.
        max_num_candidates: Maximum number of candidates to show LM judge per selection
            (default: 10). For multi strategy, this is the k value per sub-strategy.

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
    parse_failures = 0  # Track how many times LM parsing failed

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
                documents,
                annotations,
                anchor_idx,
                k=max_num_candidates * 2,
                use_summary=True,
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
                k=max_num_candidates * 2,
                embedding_preset=embedding_preset,
                use_summary=True,
                cache_alias=cache_alias,
            )
            candidate_indices = [idx for idx, score in candidates]

        elif candidate_strategy == "jaccard":
            candidates = select_candidates_jaccard(
                annotations, anchor_idx, k=max_num_candidates * 2, use_spurious=False
            )
            candidate_indices = [idx for idx, score in candidates]

        elif candidate_strategy == "multi":
            # Combine multiple strategies
            bm25_candidates = select_candidates_bm25(
                documents,
                annotations,
                anchor_idx,
                k=max_num_candidates,
                use_summary=True,
            )
            cache_alias = (
                f"{cache_alias_prefix}_embedding" if cache_alias_prefix else None
            )
            emb_candidates = select_candidates_embedding(
                documents,
                annotations,
                anchor_idx,
                k=max_num_candidates,
                embedding_preset=embedding_preset,
                use_summary=True,
                cache_alias=cache_alias,
            )
            jaccard_candidates = select_candidates_jaccard(
                annotations, anchor_idx, k=max_num_candidates, use_spurious=False
            )

            candidate_indices = merge_candidate_pools(
                bm25_candidates, emb_candidates, jaccard_candidates, use_rrf=True
            )
            # Limit merged pool to max_num_candidates
            candidate_indices = candidate_indices[:max_num_candidates]

        else:
            raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")

        # Ensure we have enough candidates
        if len(candidate_indices) < 2:
            logger.warning(f"Not enough candidates for anchor {anchor_idx}, skipping")
            continue

        # Step 2: Use LM judge to select positive from candidates
        judge_cache_alias = (
            f"{cache_alias_prefix}_judge_pos" if cache_alias_prefix else None
        )

        positive_idx, pos_parse_success = select_positive(
            anchor_idx=anchor_idx,
            candidate_indices=candidate_indices,
            documents=documents,
            annotations=annotations,
            criterion=criterion or "similarity",
            criterion_description=criterion_description
            or criterion
            or "general similarity",
            triplet_example_hint=triplet_example_hint,
            lm_judge_preset=lm_judge_preset,
            cache_alias=judge_cache_alias,
        )

        if not pos_parse_success:
            parse_failures += 1
            continue  # Skip this triplet

        # Step 3: Retrieve negative candidates using DIFFERENT strategy
        # Negatives should be lexically/semantically similar (raw text) but structurally different (tags)
        # This creates hard negatives that are superficially similar but actually dissimilar

        if candidate_strategy == "bm25":
            # Use raw documents (not summaries) and get LOW Jaccard candidates
            neg_candidates = select_candidates_bm25(
                documents,
                annotations,
                anchor_idx,
                k=max_num_candidates * 2,
                use_summary=False,
            )
            negative_candidate_indices = [
                idx for idx, score in neg_candidates if idx != positive_idx
            ]

        elif candidate_strategy == "embedding":
            # Use raw documents (not summaries)
            cache_alias = (
                f"{cache_alias_prefix}_embedding_neg" if cache_alias_prefix else None
            )
            neg_candidates = select_candidates_embedding(
                documents,
                annotations,
                anchor_idx,
                k=max_num_candidates * 2,
                embedding_preset=embedding_preset,
                use_summary=False,  # Use raw documents for negatives
                cache_alias=cache_alias,
            )
            negative_candidate_indices = [
                idx for idx, score in neg_candidates if idx != positive_idx
            ]

        elif candidate_strategy == "jaccard":
            # Invert: get LOW Jaccard similarity documents
            all_jaccard = select_candidates_jaccard(
                annotations, anchor_idx, k=len(documents) - 1, use_spurious=False
            )
            # Take the BOTTOM k (lowest Jaccard similarity)
            low_jaccard = list(reversed(all_jaccard))[: max_num_candidates * 2]
            negative_candidate_indices = [
                idx for idx, score in low_jaccard if idx != positive_idx
            ]

        elif candidate_strategy == "multi":
            # Combine: high BM25/embedding on raw docs + low Jaccard + high spurious similarity
            bm25_neg = select_candidates_bm25(
                documents,
                annotations,
                anchor_idx,
                k=max_num_candidates,
                use_summary=False,
            )
            cache_alias = (
                f"{cache_alias_prefix}_embedding_neg" if cache_alias_prefix else None
            )
            emb_neg = select_candidates_embedding(
                documents,
                annotations,
                anchor_idx,
                k=max_num_candidates,
                embedding_preset=embedding_preset,
                use_summary=False,  # Use raw documents
                cache_alias=cache_alias,
            )
            # Get low Jaccard candidates (low true similarity)
            all_jaccard = select_candidates_jaccard(
                annotations, anchor_idx, k=len(documents) - 1, use_spurious=False
            )
            low_jaccard = list(reversed(all_jaccard))[:max_num_candidates]

            # Get spurious hard negatives (high spurious sim, low true sim)
            spurious_negs = select_spurious_hard_negatives(
                annotations, anchor_idx, k=max_num_candidates
            )

            negative_candidate_indices = merge_candidate_pools(
                bm25_neg, emb_neg, low_jaccard, spurious_negs, use_rrf=True
            )
            # Remove positive if present
            negative_candidate_indices = [
                idx for idx in negative_candidate_indices if idx != positive_idx
            ]
            # Limit to max_num_candidates
            negative_candidate_indices = negative_candidate_indices[:max_num_candidates]

        else:
            raise ValueError(f"Unknown candidate_strategy: {candidate_strategy}")

        # Ensure we have at least one negative candidate
        if len(negative_candidate_indices) < 1:
            # Fallback: find any document that isn't anchor or positive
            for idx in range(len(documents)):
                if idx != anchor_idx and idx != positive_idx:
                    negative_candidate_indices = [idx]
                    break

        # Step 4: Use LM judge to select negative from filtered candidates
        neg_cache_alias = (
            f"{cache_alias_prefix}_judge_neg" if cache_alias_prefix else None
        )

        negative_idx, neg_parse_success = select_negative(
            anchor_idx=anchor_idx,
            positive_idx=positive_idx,
            candidate_indices=negative_candidate_indices,
            documents=documents,
            annotations=annotations,
            criterion=criterion or "similarity",
            criterion_description=criterion_description
            or criterion
            or "general similarity",
            triplet_example_hint=triplet_example_hint,
            lm_judge_preset=lm_judge_preset_negative,
            cache_alias=neg_cache_alias,
        )

        if not neg_parse_success:
            parse_failures += 1
            continue  # Skip this triplet

        # Create triplet (IDs only, not text)
        triplet = (anchor_idx, positive_idx, negative_idx)
        triplets.append(triplet)

    logger.info(f"Created {len(triplets)} triplets")
    if parse_failures > 0:
        total_attempted = len(triplets) + parse_failures
        logger.error("=" * 80)
        logger.error(
            f"⚠️  CRITICAL: Skipped {parse_failures}/{total_attempted} triplets due to LM parse failures "
            f"({100*parse_failures/total_attempted:.1f}%)"
        )
        logger.error("Check logs above for details on what went wrong.")
        logger.error("=" * 80)
    return triplets
