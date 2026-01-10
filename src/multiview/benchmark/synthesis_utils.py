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
    num_synthetic_docs: int = 0,
) -> tuple[list[Any], list[int]]:
    """Generate synthetic documents using LM-based remix strategy."""
    if len(documents) == 0:
        logger.warning("No documents to synthesize")
        return [], []

    if num_synthetic_docs <= 0:
        logger.info("num_synthetic_docs <= 0; skipping synthesis")
        return [], []

    num_remix_anchors = num_synthetic_docs // 2
    if num_synthetic_docs % 2 != 0:
        num_remix_anchors += 1

    synthesis_configs = getattr(document_set, "SYNTHESIS_CONFIGS", {})
    if criterion_name not in synthesis_configs:
        raise ValueError(
            f"No synthesis config found for criterion '{criterion_name}' "
            f"in {document_set.__class__.__name__}. "
            f"Available criteria: {list(synthesis_configs.keys())}"
        )

    criterion_config = synthesis_configs[criterion_name]
    remix_prompt = criterion_config.get("remix_prompt")
    if remix_prompt is None:
        raise ValueError(
            f"Synthesis config for criterion '{criterion_name}' must define 'remix_prompt'."
        )

    rng = random.Random(42)
    all_indices = list(range(len(documents)))

    if num_remix_anchors <= len(all_indices):
        remix_anchor_indices = rng.sample(all_indices, k=num_remix_anchors)
    else:
        remix_anchor_indices = [
            rng.choice(all_indices) for _ in range(num_remix_anchors)
        ]
        logger.info(
            "num_remix_anchors=%d > len(documents)=%d; sampling anchors with replacement.",
            num_remix_anchors,
            len(all_indices),
        )

    logger.info(
        "Selected %d remix anchors (will generate %d synthetic docs total).",
        len(remix_anchor_indices),
        2 * len(remix_anchor_indices),
    )

    angle_a_doc1: list[str] = []
    angle_a_doc2: list[str] = []
    angle_b_doc1: list[str] = []
    angle_b_doc2: list[str] = []

    for anchor_idx in remix_anchor_indices:
        if len(all_indices) > 1:
            decoy1 = rng.choice(all_indices)
            while decoy1 == anchor_idx:
                decoy1 = rng.choice(all_indices)
            decoy2 = rng.choice(all_indices)
            while decoy2 == anchor_idx:
                decoy2 = rng.choice(all_indices)
        else:
            decoy1 = anchor_idx
            decoy2 = anchor_idx

        angle_a_doc1.append(document_set.get_document_text(documents[anchor_idx]))
        angle_a_doc2.append(document_set.get_document_text(documents[decoy1]))
        angle_b_doc1.append(document_set.get_document_text(documents[decoy2]))
        angle_b_doc2.append(document_set.get_document_text(documents[anchor_idx]))

    remix_config = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template=remix_prompt,
        parser="delimiter",
        parser_kwargs={"delimiter": "---FINAL OUTPUT---"},
        temperature=0.7,
        max_tokens=2048,
    )

    max_original_length = max(
        len(document_set.get_document_text(doc)) for doc in documents
    )
    max_allowed_length = int(max_original_length * 1.25)

    synthetic_docs: list[Any] = []
    filtered_count = 0
    anchor_indices: list[int] = []

    def _append_filtered(raw_outputs: list, anchors: list[int]) -> None:
        nonlocal filtered_count
        for cleaned, anchor_idx in zip(raw_outputs, anchors, strict=False):
            if not isinstance(cleaned, str):
                continue
            cleaned = cleaned.strip()
            if not cleaned:
                continue
            if len(cleaned) > max_allowed_length:
                logger.warning(
                    f"Filtered out synthetic doc (too long: {len(cleaned)} > {max_allowed_length} chars)"
                )
                filtered_count += 1
                continue
            synthetic_docs.append(cleaned)
            anchor_indices.append(anchor_idx)

    for mode, d1, d2 in (
        ("angle_a", angle_a_doc1, angle_a_doc2),
        ("angle_b", angle_b_doc1, angle_b_doc2),
    ):
        logger.info(
            f"Generating {len(remix_anchor_indices)} {mode} synthetic documents..."
        )
        try:
            raw_outputs = run_inference(
                inputs={"document1": d1, "document2": d2},
                config=remix_config,
                cache_alias=f"synth_remix_{mode}_{criterion_name}",
                force_refresh=False,
                verbose=True,
            )
        except Exception as e:
            logger.error(f"Error generating {mode} documents: {e}")
            raw_outputs = []
        _append_filtered(raw_outputs, remix_anchor_indices)

    logger.info(
        f"Successfully generated {len(synthetic_docs)} synthetic documents "
        f"Filtered out {filtered_count} docs that were too long."
    )

    return synthetic_docs, anchor_indices
