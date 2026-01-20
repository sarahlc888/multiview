"""Annotation presets for schema generation and application."""

from __future__ import annotations

from ._base import InferenceConfig

ANNOTATION_PRESETS = {
    # ========================================================================
    # ANNOTATION - SCHEMA GENERATION
    # ========================================================================
    "category_schema_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/category_schema_generation.txt",
        parser="json",
        temperature=1.0,
        max_tokens=8192,
    ),
    "tag_schema_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/tag_schema_generation.txt",
        parser="json",
        temperature=0.7,
        max_tokens=8192,
    ),
    "spurious_tag_schema_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/spurious_tag_schema_generation.txt",
        parser="json",
        temperature=0.7,
        max_tokens=8192,
    ),
    "summary_guidance_generation_gemini": InferenceConfig(  # no longer used
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/summary_guidance_generation.txt",
        parser="json",
        temperature=0.7,
        max_tokens=8192,
    ),
    "summary_guidance_generation_sentence_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/summary_guidance_generation_sentence.txt",
        parser="json",
        temperature=0.7,
        max_tokens=8192,
    ),
    "summary_guidance_generation_dict_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/summary_guidance_generation_dict.txt",
        parser="json",
        temperature=0.7,
        max_tokens=8192,
    ),
    # ========================================================================
    # ANNOTATION - SCHEMA APPLICATION
    # ========================================================================
    "category_classify_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/category_classify.txt",
        parser="json",
        temperature=1.0,
        max_tokens=4096,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "tag_apply_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/tag_apply.txt",
        parser="json",
        temperature=1.0,
        max_tokens=4096,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "tag_apply_gemini_flash_lite": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/tag_apply.txt",
        parser="json",
        temperature=1.0,
        max_tokens=4096,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "tag_apply_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/tag_apply.txt",
        parser="json",
        temperature=1.0,
        max_tokens=4096,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "summary_generate_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/summary_generate.txt",
        parser="json",
        temperature=1.0,
        max_tokens=8192,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "summary_generate_sentence_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/summary_generate_sentence.txt",
        parser="json",
        temperature=1.0,
        max_tokens=8192,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "summary_generate_dict_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/summary_generate_dict.txt",
        parser="json",
        temperature=1.0,
        max_tokens=8192,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "summary_generate_dict_gemini_flash_lite": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/summary_generate_dict.txt",
        parser="json",
        temperature=1.0,
        max_tokens=8192,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "summary_generate_dict_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/annotation/summary_generate_dict.txt",
        parser="json",
        temperature=1.0,
        max_tokens=8192,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "summary_generate_dict_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/annotation/summary_generate_dict.txt",
        parser="json",
        temperature=1.0,
        max_tokens=8192,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
}
