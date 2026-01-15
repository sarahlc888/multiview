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
) -> list[tuple[int, int, int]]:
    """Create triplets based on pre-existing criterion labels.

    For datasets with known gold labels, creates triplets where:
    - Positives: Documents sharing at least one criterion value with anchor
    - Negatives: Documents sharing NO criterion values with anchor

    Default uses "hard_negatives" strategy (BM25-based selection)

    Args:
        documents: List of document texts or dicts (will extract text if needed)
        annotations: List of annotation dicts with "criterion_value" field
        max_triplets: Max number of triplets (None = use all documents as anchors)
        selection_strategy: "random" or "hard_negatives" (BM25-based)
        allow_multi_class: Allow documents to have multiple criterion values
        seed: Random seed for deterministic sampling

    Returns:
        List of (anchor_id, positive_id, negative_id) triplets

    Raises:
        ValueError: If documents don't have sufficient criterion diversity
    """
    from collections import defaultdict

    if len(documents) < 3:
        logger.warning("Need at least 3 documents to create triplets")
        return []

    # Store original documents before text extraction (to check for is_sentence markers)
    original_documents = documents

    # Extract text from documents (handle both string and dict formats)
    document_texts = []
    for doc in documents:
        if isinstance(doc, dict):
            document_texts.append(doc.get("text", str(doc)))
        elif isinstance(doc, str):
            document_texts.append(doc)
        else:
            document_texts.append(str(doc))

    documents = document_texts

    # Build classmap: {criterion_value: [doc_indices]}
    classmap = defaultdict(list)
    doc_to_classes = {}  # doc_idx -> [criterion_values]

    for idx, ann in enumerate(annotations):
        criterion_value = ann.get("criterion_value")
        if criterion_value is None:
            logger.warning(f"Document {idx} has no criterion_value, skipping")
            continue

        # Support multi-class: criterion_value can be a list or single value
        if isinstance(criterion_value, list):
            values = criterion_value
        else:
            values = [criterion_value]

        doc_to_classes[idx] = values
        for value in values:
            classmap[value].append(idx)

    if len(classmap) < 2:
        logger.warning(
            f"Need at least 2 distinct criterion values to create triplets, found {len(classmap)}"
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
    # If documents have is_anchor markers, only use those as anchors
    marked_anchor_indices = [
        idx
        for idx in doc_to_classes.keys()
        if isinstance(original_documents[idx], dict)
        and original_documents[idx].get("is_anchor", False)
    ]

    if marked_anchor_indices:
        logger.info(
            f"Found {len(marked_anchor_indices)} marked anchors, using only those"
        )
        anchor_indices = marked_anchor_indices
    else:
        anchor_indices = list(doc_to_classes.keys())
    if max_triplets is not None and max_triplets < len(anchor_indices):
        # Deterministic sampling of anchors
        anchor_indices = deterministic_sample(
            anchor_indices, max_triplets, seed_base=f"criterion_anchors_{seed}"
        )

    triplets = []
    used_docs = set(anchor_indices)  # Track used docs to avoid repeats

    for anchor_idx in anchor_indices:
        # Find positive candidates: share at least one criterion value, not anchor itself
        pos_candidates = [
            idx
            for idx in doc_to_classes.keys()
            if idx != anchor_idx
            and idx not in used_docs
            and shares_criterion(anchor_idx, idx)
        ]

        # Find negative candidates: share NO criterion values
        neg_candidates = [
            idx
            for idx in doc_to_classes.keys()
            if idx != anchor_idx
            and idx not in used_docs
            and not shares_criterion(anchor_idx, idx)
        ]

        if len(pos_candidates) == 0:
            logger.warning(
                f"No positive candidates for anchor {anchor_idx} "
                f"(criterion: {doc_to_classes[anchor_idx]}), skipping"
            )
            continue

        if len(neg_candidates) == 0:
            logger.warning(
                f"No negative candidates for anchor {anchor_idx} "
                f"(criterion: {doc_to_classes[anchor_idx]}), skipping"
            )
            continue

        # Select positive and negative based on strategy
        if selection_strategy == "random":
            # Random selection
            pos_idx = deterministic_sample(
                pos_candidates, 1, seed_base=f"criterion_pos_{anchor_idx}_{seed}"
            )[0]
            neg_idx = deterministic_sample(
                neg_candidates, 1, seed_base=f"criterion_neg_{anchor_idx}_{seed}"
            )[0]

        elif selection_strategy == "hard_negatives":
            # BM25-based selection (implemented separately)
            pos_idx = _select_with_bm25(
                anchor_idx=anchor_idx,
                documents=documents,
                candidate_indices=pos_candidates,
                num_to_select=1,
                prefer_low_score=True,  # Hard positives: low BM25
                seed=seed,
            )[0]

            neg_idx = _select_with_bm25(
                anchor_idx=anchor_idx,
                documents=documents,
                candidate_indices=neg_candidates,
                num_to_select=1,
                prefer_low_score=False,  # Hard negatives: high BM25
                seed=seed,
            )[0]

        else:
            raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

        triplets.append((anchor_idx, pos_idx, neg_idx))
        used_docs.add(pos_idx)
        used_docs.add(neg_idx)

    logger.info(f"Created {len(triplets)} prelabeled triplets")
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

    anchor_tags = extract_active_tags(anchor_ann, "tags")
    anchor_spurious = extract_active_tags(anchor_ann, "spurious_tags")

    if bm25_scores is None:
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
) -> tuple[list[int | None], list[bool]]:
    """Use LM judge to select positives for multiple anchors in one batch."""
    if not anchor_indices:
        return [], []

    triplet_example_section = _format_triplet_example_section(
        triplet_example_hint, selection_type="positive"
    )

    anchor_docs = []
    anchor_annotations = []
    candidates_texts = []

    for anchor_idx, candidate_indices in zip(
        anchor_indices, candidate_indices_by_anchor, strict=False
    ):
        anchor_docs.append(documents[anchor_idx])
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
        "criterion": [criterion] * len(anchor_indices),
        "criterion_description": [criterion_description or criterion]
        * len(anchor_indices),
        "triplet_example_section": [triplet_example_section] * len(anchor_indices),
        "anchor_doc": anchor_docs,
        "anchor_annotation": anchor_annotations,
        "candidates": candidates_texts,
    }

    logger.debug(
        f"Running LM judge for positive selection (batch_size={len(anchor_indices)}, "
        f"cache_alias={cache_alias})"
    )
    results = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )

    positive_indices = []
    parse_successes = []

    for _, candidate_indices, selected_num in zip(
        anchor_indices, candidate_indices_by_anchor, results, strict=False
    ):
        selected_num = _coerce_selected_num(selected_num)
        if isinstance(selected_num, int) and 1 <= selected_num <= len(
            candidate_indices
        ):
            positive_indices.append(candidate_indices[selected_num - 1])
            parse_successes.append(True)
        else:
            logger.error(
                "CRITICAL: Failed to parse positive selection from LM response. "
                f"Got: {selected_num}. Skipping triplet."
            )
            positive_indices.append(None)
            parse_successes.append(False)

    return positive_indices, parse_successes


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
) -> tuple[list[int | None], list[bool]]:
    """Use LM judge to select negatives for multiple anchors in one batch."""
    if not anchor_indices:
        return [], []

    triplet_example_section = _format_triplet_example_section(
        triplet_example_hint, selection_type="negative"
    )

    anchor_docs = []
    anchor_annotations = []
    positive_docs = []
    positive_annotations = []
    candidates_texts = []

    for anchor_idx, positive_idx, candidate_indices in zip(
        anchor_indices, positive_indices, candidate_indices_by_anchor, strict=False
    ):
        anchor_docs.append(documents[anchor_idx])
        anchor_ann = annotations[anchor_idx]
        anchor_annotations.append(
            format_annotation_for_display(anchor_ann, include_spurious=True)
        )

        positive_docs.append(documents[positive_idx])
        positive_ann = annotations[positive_idx]
        positive_annotations.append(
            format_annotation_for_display(positive_ann, include_spurious=True)
        )

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
        "criterion": [criterion] * len(anchor_indices),
        "criterion_description": [criterion_description or criterion]
        * len(anchor_indices),
        "triplet_example_section": [triplet_example_section] * len(anchor_indices),
        "anchor_doc": anchor_docs,
        "anchor_annotation": anchor_annotations,
        "positive_doc": positive_docs,
        "positive_annotation": positive_annotations,
        "candidates": candidates_texts,
    }

    logger.debug(
        f"Running LM judge for negative selection (batch_size={len(anchor_indices)}, "
        f"cache_alias={cache_alias})"
    )
    results = run_inference(
        inputs=inputs,
        config=lm_judge_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=False,
    )

    negative_indices = []
    parse_successes = []

    for positive_idx, candidate_indices, selected_num in zip(
        positive_indices, candidate_indices_by_anchor, results, strict=False
    ):
        selected_num = _coerce_selected_num(selected_num)
        if isinstance(selected_num, int) and 1 <= selected_num <= len(
            candidate_indices
        ):
            negative_idx = candidate_indices[selected_num - 1]
            if negative_idx == positive_idx and len(candidate_indices) > 1:
                negative_idx = (
                    candidate_indices[1]
                    if candidate_indices[0] == positive_idx
                    else candidate_indices[0]
                )
            negative_indices.append(negative_idx)
            parse_successes.append(True)
        else:
            logger.error(
                "CRITICAL: Failed to parse negative selection from LM response. "
                f"Got: {selected_num}. Skipping triplet."
            )
            negative_indices.append(None)
            parse_successes.append(False)

    return negative_indices, parse_successes


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
        embedding_preset_overrides: Optional overrides for embedding preset (e.g., custom embed_query_instr_template)
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
        # Build inputs - include criterion if provided (needed for instruction-tuned embeddings)
        inputs = {"document": summary_texts}
        if criterion is not None:
            inputs["criterion"] = criterion

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
            top_k_indices = np.argsort(scores)[::-1][: max_num_candidates * 2]
            candidate_indices = [int(idx) for idx in top_k_indices]

        elif candidate_strategy == "embedding":
            similarities = embedding_summary @ embedding_summary[anchor_idx]
            similarities[anchor_idx] = -np.inf
            top_k_indices = np.argsort(similarities)[::-1][: max_num_candidates * 2]
            candidate_indices = [int(idx) for idx in top_k_indices]

        elif candidate_strategy == "jaccard":
            candidates = select_candidates_jaccard(
                annotations, anchor_idx, k=max_num_candidates * 2, use_spurious=False
            )
            candidate_indices = [idx for idx, score in candidates]

        elif candidate_strategy == "multi":
            bm25_scores = bm25_summary_matrix[anchor_idx].copy()
            bm25_scores[anchor_idx] = -np.inf
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:max_num_candidates]
            bm25_candidates = [
                (int(idx), float(bm25_scores[idx])) for idx in bm25_top_indices
            ]

            emb_scores = embedding_summary @ embedding_summary[anchor_idx]
            emb_scores[anchor_idx] = -np.inf
            emb_top_indices = np.argsort(emb_scores)[::-1][:max_num_candidates]
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

    positive_indices, pos_parse_successes = select_positive_batch(
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
    for anchor_idx, positive_idx, parse_success in zip(
        anchors_for_pos, positive_indices, pos_parse_successes, strict=False
    ):
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
            top_k_indices = np.argsort(scores)[::-1][: max_num_candidates * 2]
            negative_candidate_indices = [
                int(idx) for idx in top_k_indices if idx != positive_idx
            ]

        elif candidate_strategy == "embedding":
            # Use BM25 on raw docs for negatives (avoid evaluation bias from embeddings)
            scores = bm25_raw_matrix[anchor_idx].copy()
            scores[anchor_idx] = -np.inf
            top_k_indices = np.argsort(scores)[::-1][: max_num_candidates * 2]
            negative_candidate_indices = [
                int(idx) for idx in top_k_indices if idx != positive_idx
            ]

        elif candidate_strategy == "jaccard":
            all_jaccard = select_candidates_jaccard(
                annotations, anchor_idx, k=len(documents) - 1, use_spurious=False
            )
            low_jaccard = list(reversed(all_jaccard))[: max_num_candidates * 2]
            negative_candidate_indices = [
                idx for idx, score in low_jaccard if idx != positive_idx
            ]

        elif candidate_strategy == "multi":
            # For negatives, use BM25, Jaccard, and spurious (NO embeddings to avoid bias)
            bm25_scores = bm25_raw_matrix[anchor_idx].copy()
            bm25_scores[anchor_idx] = -np.inf
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:max_num_candidates]
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

    negative_indices, neg_parse_successes = select_negative_batch(
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

    for anchor_idx, negative_idx, parse_success in zip(
        anchors_for_neg, negative_indices, neg_parse_successes, strict=False
    ):
        if not parse_success or negative_idx is None:
            parse_failures += 1
            continue
        triplets.append((anchor_idx, positive_by_anchor[anchor_idx], negative_idx))

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
