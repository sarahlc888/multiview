"""Specialized inference method presets (in-one-word, pseudologit)."""

from __future__ import annotations

from ._base import InferenceConfig

SPECIALIZED_PRESETS = {
    # ========================================================================
    # IN-ONE-WORD HIDDEN STATE EMBEDDINGS
    # ========================================================================
    "inoneword_hf_qwen3_8b": InferenceConfig(
        provider="hf_local_hidden_state",
        model_name="Qwen/Qwen3-8B",
        prompt_template="prompts/eval/inoneword.txt",
        parser="vector",
        force_prefill_template="In one word:",
        extra_kwargs={
            "is_chatml_prompt": True,
            "hidden_layer_idx": -1,
            "batch_size": 8,
            "max_length": 2048,
        },
    ),
    # ========================================================================
    # PSEUDOLOGIT EMBEDDINGS
    # ========================================================================
    # Gemini 2.5 Flash Lite - different sample counts
    "pseudologit_gemini_n3": InferenceConfig(
        provider="pseudologit",
        model_name="gemini-2.5-flash",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={
            "prompt_template": "prompts/custom/pseudologit_classify.txt",
            "n_samples": 3,
            "temperature": 0.7,
            "provider": "gemini",
            "max_tokens": 8192,
        },
    ),
    "pseudologit_gemini_n10": InferenceConfig(
        provider="pseudologit",
        model_name="gemini-2.5-flash",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={
            "classes_file": "prompts/custom/gsm8k_classes.json",
            "prompt_template": "prompts/custom/pseudologit_classify.txt",
            "n_samples": 10,
            "temperature": 0.7,
            "provider": "gemini",
            "max_tokens": 8192,
        },
    ),
    "pseudologit_gemini_n50": InferenceConfig(
        provider="pseudologit",
        model_name="gemini-2.5-flash",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={
            "classes_file": "prompts/custom/gsm8k_classes.json",
            "prompt_template": "prompts/custom/pseudologit_classify.txt",
            "n_samples": 50,
            "temperature": 0.7,
            "provider": "gemini",
            "max_tokens": 8192,
        },
    ),
}
