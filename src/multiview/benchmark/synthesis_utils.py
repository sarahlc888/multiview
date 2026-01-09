"""Utilities for synthesizing documents using LM-based rewriting with remix strategy."""

import logging
import random
from typing import Any

from multiview.inference.inference import run_inference
from multiview.inference.presets import InferenceConfig

logger = logging.getLogger(__name__)


def synthesize_documents(
    documents: list[Any],
    document_set: Any,
    criterion_name: str,
    num_synthetic_per_doc: int = 2,
) -> list[Any]:
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
        List of synthetic documents

    Raises:
        ValueError: If no synthesis config found for this criterion
    """
    if len(documents) == 0:
        logger.warning("No documents to synthesize")
        return []

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
        return []

    logger.info(f"Generating {num_pairs} (X, Y) pairs for synthesis...")

    # Sample (X, Y) pairs
    random.seed(42)  # For reproducibility
    pairs = []
    for _ in range(num_pairs):
        doc_x = random.choice(documents)
        doc_y = random.choice(documents)
        # Ensure X != Y if possible
        while doc_y == doc_x and len(documents) > 1:
            doc_y = random.choice(documents)
        pairs.append((doc_x, doc_y))

    # Build inputs for both synthesis types
    texts_pos = [document_set.get_document_text(x) for x, y in pairs]
    refs_pos = [document_set.get_document_text(y) for x, y in pairs]

    texts_neg = [document_set.get_document_text(x) for x, y in pairs]
    refs_neg = [document_set.get_document_text(y) for x, y in pairs]

    criterias = [criterion_name] * len(pairs)

    # Create InferenceConfigs dynamically with criterion-specific prompts
    config_pos = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template=hard_positive_prompt,
        parser="text",
        temperature=0.7,
        max_tokens=2048,
    )

    config_neg = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
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

    # Extract and combine results
    synthetic_docs = []

    # Extract hard positives
    for completion in raw_positives:
        if completion and completion.strip():
            synthetic_docs.append(completion.strip())

    # Extract hard negatives
    for completion in raw_negatives:
        if completion and completion.strip():
            synthetic_docs.append(completion.strip())

    logger.info(
        f"Successfully generated {len(synthetic_docs)} synthetic documents "
        f"({len([x for x in raw_positives if x])} hard positive + "
        f"{len([x for x in raw_negatives if x])} hard negative)"
    )

    return synthetic_docs
