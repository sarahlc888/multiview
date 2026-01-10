"""Utilities for synthesizing documents using LM-based rewriting with remix strategy."""

import logging
import random
from typing import Any

from multiview.inference.inference import run_inference
from multiview.inference.presets import InferenceConfig

logger = logging.getLogger(__name__)


def _extract_problem_text(completion: str) -> str:
    """Extract clean problem text from LLM completion.

    Generic cleaning that removes reasoning and LLM artifacts. Looks for the
    "---FINAL OUTPUT---" delimiter to extract the final answer, or falls back
    to heuristic parsing. Does not assume any specific document format.

    Args:
        completion: Raw LLM completion text

    Returns:
        Cleaned problem text with reasoning and artifacts removed
    """
    text = completion.strip()

    # Primary extraction: Look for the FINAL OUTPUT delimiter
    if "---FINAL OUTPUT---" in text:
        text = text.split("---FINAL OUTPUT---", 1)[1].strip()
        return text

    # Fallback: Use heuristic parsing if delimiter not found
    # Remove markdown section headers if present
    if "### Rewritten Problem" in text:
        text = text.split("### Rewritten Problem", 1)[1].strip()

    # Remove leading parenthetical instructions if present
    if text.startswith("(Output ONLY") or text.startswith("(output only"):
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1].strip()

    # Remove leading numbered analysis lists (e.g., "1. Response 1 arithmetic...")
    # Stop when we hit a blank line followed by actual content
    lines = text.split("\n")
    skip_until_blank = False
    if lines and lines[0].strip() and lines[0].strip()[0].isdigit():
        skip_until_blank = True

    if skip_until_blank:
        # Find the first blank line, then start from the next non-blank line
        for i, line in enumerate(lines):
            if not line.strip():  # Found blank line
                # Get content after blank line
                remaining = "\n".join(lines[i + 1 :]).strip()
                if remaining:
                    text = remaining
                    break

    return text


def synthesize_documents(
    documents: list[Any],
    document_set: Any,
    criterion_name: str,
    num_synthetic_per_doc: int = 2,
) -> tuple[list[Any], dict]:
    """Generate synthetic documents using LM-based remix strategy.

    For each (X, Y) pair of documents:
    - Hard positive: Preserves X's criterion + borrows Y's themes
    - Hard negative: Changes to Y's criterion + borrows X's themes

    Args:
        documents: List of original documents
        document_set: DocumentSet instance (for getting text and configs)
        criterion_name: Criterion being used
        num_synthetic_per_doc: How many synthetic docs per original (split 50/50)

    Returns:
        Tuple of (synthetic_documents, synthesis_metadata)
        - synthetic_documents: List of synthetic documents
        - synthesis_metadata: Dict with metadata about synthesis process including:
            - anchor_indices: List of X document indices (backward compatibility)
            - num_original_docs: Number of original documents
            - num_pairs: Number of (X,Y) pairs generated
            - num_filtered: Number of documents filtered out
            - pairs: List of {pair_id, doc_x_idx, doc_y_idx} dicts
            - synthetic_docs: List of metadata dicts for each synthetic document

    Raises:
        ValueError: If no synthesis config found for this criterion
    """
    if len(documents) == 0:
        logger.warning("No documents to synthesize")
        return [], {}

    # Check for criterion-specific config
    synthesis_configs = getattr(document_set, "SYNTHESIS_CONFIGS", {})
    if criterion_name not in synthesis_configs:
        raise ValueError(
            f"No synthesis config found for criterion '{criterion_name}' "
            f"in {document_set.__class__.__name__}. "
            f"Available criteria: {list(synthesis_configs.keys())}"
        )

    criterion_config = synthesis_configs[criterion_name]
    hard_positive_prompt = criterion_config["hard_positive_prompt"]
    hard_negative_prompt = criterion_config["hard_negative_prompt"]

    # Calculate number of pairs needed
    # Each pair generates 1 hard_positive + 1 hard_negative
    num_pairs = (len(documents) * num_synthetic_per_doc) // 2

    if num_pairs == 0:
        logger.warning("num_synthetic_per_doc too small, no pairs will be generated")
        return [], {}

    logger.info(f"Generating {num_pairs} (X, Y) pairs for synthesis...")

    # Sample (X, Y) pairs using indices
    random.seed(42)  # For reproducibility
    pairs = []  # List of (idx_x, idx_y) tuples
    anchor_indices = []  # Track X indices for triplet anchors
    pairs_metadata = []  # Track pair metadata for validation

    for pair_id in range(num_pairs):
        idx_x = random.randint(0, len(documents) - 1)
        idx_y = random.randint(0, len(documents) - 1)
        # Ensure X != Y if possible
        while idx_y == idx_x and len(documents) > 1:
            idx_y = random.randint(0, len(documents) - 1)
        pairs.append((idx_x, idx_y))
        anchor_indices.append(idx_x)
        pairs_metadata.append({
            "pair_id": pair_id,
            "doc_x_idx": idx_x,
            "doc_y_idx": idx_y,
        })

    # Build inputs for both synthesis types
    texts_pos = [
        document_set.get_document_text(documents[idx_x]) for idx_x, idx_y in pairs
    ]
    refs_pos = [
        document_set.get_document_text(documents[idx_y]) for idx_x, idx_y in pairs
    ]

    texts_neg = [
        document_set.get_document_text(documents[idx_x]) for idx_x, idx_y in pairs
    ]
    refs_neg = [
        document_set.get_document_text(documents[idx_y]) for idx_x, idx_y in pairs
    ]

    criterias = [criterion_name] * len(pairs)

    # Create InferenceConfigs dynamically with criterion-specific prompts
    config_pos = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template=hard_positive_prompt,
        parser="text",
        temperature=0.7,
        max_tokens=2048,
    )

    config_neg = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template=hard_negative_prompt,
        parser="text",
        temperature=0.7,
        max_tokens=2048,
    )

    # Run inference for hard positives
    logger.info(
        f"Generating {len(pairs)} hard positive synthetic documents (preserve X's criterion, borrow Y's themes)..."
    )
    try:
        raw_positives = run_inference(
            inputs={"text": texts_pos, "ref": refs_pos, "criteria": criterias},
            config=config_pos,
            cache_alias=f"synth_hard_positive_{criterion_name}",
            force_refresh=False,
            verbose=True,
        )
    except Exception as e:
        logger.error(f"Error generating hard positive documents: {e}")
        raw_positives = []

    # Run inference for hard negatives
    logger.info(
        f"Generating {len(pairs)} hard negative synthetic documents (change to Y's criterion, borrow X's themes)..."
    )
    try:
        raw_negatives = run_inference(
            inputs={"text": texts_neg, "ref": refs_neg, "criteria": criterias},
            config=config_neg,
            cache_alias=f"synth_hard_negative_{criterion_name}",
            force_refresh=False,
            verbose=True,
        )
    except Exception as e:
        logger.error(f"Error generating hard negative documents: {e}")
        raw_negatives = []

    # Calculate maximum allowed length (125% of longest original doc)
    max_original_length = max(
        len(document_set.get_document_text(doc)) for doc in documents
    )
    max_allowed_length = int(max_original_length * 1.25)

    # Extract and combine results with filtering
    synthetic_docs = []
    synthetic_docs_metadata = []
    filtered_count = 0
    synthetic_idx = 0  # Track index in synthetic_docs list

    # Extract hard positives
    for pair_id, completion in enumerate(raw_positives):
        if completion and completion.strip():
            cleaned = _extract_problem_text(completion)
            if cleaned:
                # Filter out docs that are too long
                if len(cleaned) > max_allowed_length:
                    logger.warning(
                        f"Filtered out synthetic doc (too long: {len(cleaned)} > {max_allowed_length} chars)"
                    )
                    filtered_count += 1
                    continue
                synthetic_docs.append(cleaned)
                synthetic_docs_metadata.append({
                    "synthetic_idx": synthetic_idx,
                    "type": "hard_positive",
                    "anchor_doc_idx": pairs[pair_id][0],
                    "reference_doc_idx": pairs[pair_id][1],
                    "pair_id": pair_id,
                })
                synthetic_idx += 1

    # Extract hard negatives
    for pair_id, completion in enumerate(raw_negatives):
        if completion and completion.strip():
            cleaned = _extract_problem_text(completion)
            if cleaned:
                # Filter out docs that are too long
                if len(cleaned) > max_allowed_length:
                    logger.warning(
                        f"Filtered out synthetic doc (too long: {len(cleaned)} > {max_allowed_length} chars)"
                    )
                    filtered_count += 1
                    continue
                synthetic_docs.append(cleaned)
                synthetic_docs_metadata.append({
                    "synthetic_idx": synthetic_idx,
                    "type": "hard_negative",
                    "anchor_doc_idx": pairs[pair_id][0],
                    "reference_doc_idx": pairs[pair_id][1],
                    "pair_id": pair_id,
                })
                synthetic_idx += 1

    logger.info(
        f"Successfully generated {len(synthetic_docs)} synthetic documents "
        f"({len([x for x in raw_positives if x])} hard positive + "
        f"{len([x for x in raw_negatives if x])} hard negative). "
        f"Filtered out {filtered_count} docs that were too long."
    )

    # Build synthesis metadata dict
    synthesis_metadata = {
        "anchor_indices": anchor_indices,  # Backward compatibility
        "num_original_docs": len(documents),
        "num_pairs": num_pairs,
        "num_filtered": filtered_count,
        "pairs": pairs_metadata,
        "synthetic_docs": synthetic_docs_metadata,
    }

    return synthetic_docs, synthesis_metadata
