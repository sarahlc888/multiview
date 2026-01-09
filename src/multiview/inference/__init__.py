"""Inference module for LM and embedding model completions.

Main interface is the run_inference() function.

Usage:
    from multiview.inference import run_inference

    # Using a preset
    results = run_inference(
        inputs={"documents": ["text1", "text2"]},
        config="openai_embedding_large",
        cache_alias="my_task",
    )

    # Using a custom config
    from multiview.inference import InferenceConfig
    config = InferenceConfig(
        provider="openai",
        model_name="gpt-4o",
        prompt_template="Analyze document: {document}\nCriterion: {criterion}",
        parser="json",
    )
    results = run_inference(
        inputs={"documents": ["text"], "criterion": "word_count"},
        config=config
    )
"""

from multiview.inference.inference import run_inference
from multiview.inference.presets import (
    # Specialized task presets
    EMBED_PLAINTEXT_HFAPI,
    LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI,
    LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI,
    LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI,
    REWRITE_PLAINTEXT_FREEFORM_GEMINI,
    InferenceConfig,
    get_preset,
    list_presets,
)

__all__ = [
    "run_inference",
    "InferenceConfig",
    "get_preset",
    "list_presets",
    # Specialized task presets
    "EMBED_PLAINTEXT_HFAPI",
    "REWRITE_PLAINTEXT_FREEFORM_GEMINI",
    "LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI",
    "LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI",
    "LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI",
]
