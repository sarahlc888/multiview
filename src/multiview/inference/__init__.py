"""Inference module for LM and embedding model completions.

Main interface is the run_inference() function.

Usage:
    from multiview.inference import run_inference, get_preset

    # Using a preset by name
    results = run_inference(
        inputs={"documents": ["text1", "text2"]},
        config="openai_embedding_large",
        cache_alias="my_task",
    )

    # Or get the preset config explicitly
    config = get_preset("openai_embedding_large")
    results = run_inference(
        inputs={"documents": ["text1", "text2"]},
        config=config,
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
    InferenceConfig,
    get_preset,
    list_presets,
)

__all__ = [
    "run_inference",
    "InferenceConfig",
    "get_preset",
    "list_presets",
]
