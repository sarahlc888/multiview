"""Inference configuration and presets.

This module defines InferenceConfig and provides preset configurations
for common models/providers via the PRESET_REGISTRY.

All preset configurations are defined inline in the registry, accessed
via string keys using get_preset().

Usage:
    from multiview.inference.presets import get_preset

    # Get a preset by name
    config = get_preset("openai_embedding_large")

    # List all available presets
    from multiview.inference.presets import list_presets
    presets = list_presets()
"""

from ._base import InferenceConfig

# ============================================================================
# PRESET REGISTRY
# ============================================================================

PRESET_REGISTRY = {
    # ========================================================================
    # EMBEDDING MODELS
    # ========================================================================
    # OpenAI Embeddings
    "openai_embedding_large": InferenceConfig(
        provider="openai_embedding",
        model_name="text-embedding-3-large",
        prompt_template="{document}",
        parser="vector",
    ),
    "openai_embedding_small": InferenceConfig(
        provider="openai_embedding",
        model_name="text-embedding-3-small",
        prompt_template="{document}",
        parser="vector",
    ),
    # HuggingFace API Embeddings (without instruction prefix)
    "hf_qwen3_embedding_8b": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-8B",
        prompt_template="{document}",
        parser="vector",
    ),
    "hf_qwen3_embedding_4b": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-4B",
        prompt_template="{document}",
        parser="vector",
    ),
    # HuggingFace API Embeddings (with criterion-aware instruction)
    "instr_hf_qwen3_embedding_8b": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-8B",
        prompt_template="{document}",
        embed_query_instr_template="Given a query, retrieve documents based on {criterion}",
        parser="vector",
    ),
    "instr_hf_qwen3_embedding_4b": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-4B",
        prompt_template="{document}",
        embed_query_instr_template="Given a query, retrieve documents based on {criterion}",
        parser="vector",
    ),
    # Specialized embedding preset for plain text retrieval via HF API
    "embed_plaintext_hfapi": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-8B",
        prompt_template="{document}",
        parser="vector",
    ),
    # ========================================================================
    # BASIC LM MODELS
    # ========================================================================
    # Anthropic Models
    "claude_sonnet": InferenceConfig(
        provider="anthropic",
        model_name="claude-3-5-sonnet-20241022",
        prompt_template="{document}",
        parser="json",
        temperature=0.0,
        max_tokens=4096,
    ),
    "claude_haiku": InferenceConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        prompt_template="{document}",
        parser="json",
        temperature=0.0,
        max_tokens=4096,
    ),
    # OpenAI Models
    "gpt41": InferenceConfig(
        provider="openai",
        model_name="gpt-4.1",
        prompt_template="{document}",
        parser="json",
        temperature=0.0,
        max_tokens=4096,
    ),
    "gpt41_mini": InferenceConfig(
        provider="openai",
        model_name="gpt-4.1-mini",
        prompt_template="{document}",
        parser="json",
        temperature=0.0,
        max_tokens=4096,
    ),
    # Gemini Models
    "gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="{document}",
        parser="json",
        temperature=0.0,
        max_tokens=4096,
    ),
    "gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="{document}",
        parser="json",
        temperature=0.0,
        max_tokens=4096,
    ),
    # ========================================================================
    # DOCUMENT REWRITING / SUMMARIZATION
    # ========================================================================
    "rewrite_plaintext_freeform_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/rewrite/rewrite_plaintext_freeform.txt",
        parser="json",
        parser_kwargs={"annotation_key": "summary"},
        temperature=0.0,
        max_tokens=2048,
    ),
    # ========================================================================
    # LM JUDGE - PAIRWISE COMPARISON
    # ========================================================================
    # Likert scale judge (1-5)
    "lmjudge_pair_plaintext_likerthard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_plaintext_likerthard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    # Binary judge (0=same, 1=different)
    "lmjudge_pair_norewrite_binaryhard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_norewrite_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            }
        },
        temperature=0.0,
        max_tokens=2048,
    ),
    # Likert scale judge with annotations (1-5)
    "lmjudge_pair_with_annotation_likerthard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_with_annotation_likerthard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    # Binary judge with annotations (0=same, 1=different)
    "lmjudge_pair_with_annotation_binaryhard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/pair_with_annotation_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
                r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            }
        },
        temperature=0.0,
        max_tokens=2048,
    ),
    # ========================================================================
    # LM JUDGE - TRIPLET COMPARISON
    # ========================================================================
    # Gemini Flash Lite (default)
    "lmjudge_triplet_plaintext_binaryhard_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    "lmjudge_triplet_with_annotation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    # Gemini Flash (full, not lite)
    "lmjudge_triplet_plaintext_binaryhard_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    "lmjudge_triplet_with_annotation_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    # Gemini Pro
    "lmjudge_triplet_plaintext_binaryhard_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/lm_judge/triplet_plaintext_binaryhard.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    "lmjudge_triplet_with_annotation_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    "lmjudge_triplet_plaintext_binaryhard_with_annotation_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/lm_judge/triplet_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Aa]nswer:\s*\(b\)": 1,
                r"Final\s+[Aa]nswer:\s*\(c\)": -1,
                r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    # ========================================================================
    # LM JUDGE - QUALITY RATING
    # ========================================================================
    "lmjudge_quality_rating_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/quality/quality_rating.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Rr]ating:\s*1": 1,
                r"Final\s+[Rr]ating:\s*2": 2,
                r"Final\s+[Rr]ating:\s*3": 3,
                r"Final\s+[Rr]ating:\s*4": 4,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    "lmjudge_quality_rating_with_annotation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/quality/quality_rating_with_annotation.txt",
        parser="regex",
        parser_kwargs={
            "outputs_to_match": {
                r"Final\s+[Rr]ating:\s*1": 1,
                r"Final\s+[Rr]ating:\s*2": 2,
                r"Final\s+[Rr]ating:\s*3": 3,
                r"Final\s+[Rr]ating:\s*4": 4,
            }
        },
        temperature=0.0,
        max_tokens=4096,
    ),
    # ========================================================================
    # ANNOTATION - SCHEMA GENERATION
    # ========================================================================
    "category_schema_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/category_schema_generation.txt",
        parser="json",
        temperature=0.0,
        max_tokens=8192,
    ),
    "tag_schema_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/tag_schema_generation.txt",
        parser="json",
        temperature=0.0,
        max_tokens=8192,
    ),
    "spurious_tag_schema_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/spurious_tag_schema_generation.txt",
        parser="json",
        temperature=0.0,
        max_tokens=8192,
    ),
    "summary_guidance_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/summary_guidance_generation.txt",
        parser="json",
        temperature=0.0,
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
        max_tokens=2048,
        extra_kwargs={"top_p": 0.95, "top_k": 64},
    ),
    "tag_apply_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/annotation/tag_apply.txt",
        parser="json",
        temperature=1.0,
        max_tokens=2048,
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
    # ========================================================================
    # PAIRWISE SIMILARITY HINT GENERATION
    # ========================================================================
    "pairwise_sim_hint_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/pairwise_sim_hint_generation.txt",
        parser="delimiter",
        parser_kwargs={"delimiter": "---FINAL OUTPUT---"},
        temperature=0.0,
        max_tokens=8192,
    ),
    # ========================================================================
    # TRIPLET SELECTION
    # ========================================================================
    "triplet_select_positive_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/triplet/triplet_select_positive.txt",
        parser="json",
        parser_kwargs={"annotation_key": "chosen_positive"},
        temperature=0.5,
        max_tokens=8192,
        extra_kwargs={"response_mime_type": "application/json"},
    ),
    "triplet_select_negative_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/triplet/triplet_select_negative.txt",
        parser="json",
        parser_kwargs={"annotation_key": "chosen_negative"},
        temperature=0.5,
        max_tokens=8192,
        extra_kwargs={"response_mime_type": "application/json"},
    ),
    # ========================================================================
    # QUERY EXPANSION (EVALUATION)
    # ========================================================================
    "query_expansion_summary_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/eval/query_expansion_summary.txt",
        parser="json",
        temperature=0.0,
        max_tokens=2048,
    ),
    "query_expansion_summary_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/eval/query_expansion_summary.txt",
        parser="json",
        temperature=0.0,
        max_tokens=2048,
    ),
    "query_expansion_summary_gemini_pro": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/eval/query_expansion_summary.txt",
        parser="json",
        temperature=0.0,
        max_tokens=2048,
    ),
    "query_expansion_summary_claude_haiku": InferenceConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
        prompt_template="prompts/eval/query_expansion_summary.txt",
        parser="json",
        temperature=0.0,
        max_tokens=2048,
    ),
    "query_expansion_summary_claude_sonnet": InferenceConfig(
        provider="anthropic",
        model_name="claude-3-7-sonnet-20250219",
        prompt_template="prompts/eval/query_expansion_summary.txt",
        parser="json",
        temperature=0.0,
        max_tokens=2048,
    ),
    # ========================================================================
    # RERANKER MODELS
    # ========================================================================
    "qwen3_reranker_8b": InferenceConfig(
        provider="hf_local_reranker",
        model_name="Qwen/Qwen3-Reranker-8B",
        prompt_template="<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
        parser="score",
        extra_kwargs={
            "device": "cuda",
            "batch_size": 32,
            "max_length": 8192,
            "use_fp16": True,
        },
    ),
    "qwen3_reranker_8b_cpu": InferenceConfig(
        provider="hf_local_reranker",
        model_name="Qwen/Qwen3-Reranker-8B",
        prompt_template="<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}",
        parser="score",
        extra_kwargs={
            "device": "cpu",
            "batch_size": 16,
            "max_length": 8192,
            "use_fp16": False,
        },
    ),
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
}


def get_preset(name: str) -> InferenceConfig:
    """Get a preset configuration by name.

    Args:
        name: Preset name (e.g., "openai_embedding_large", "lmjudge_pair_plaintext_likerthard_gemini")

    Returns:
        InferenceConfig for the preset

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESET_REGISTRY:
        raise ValueError(
            f"Unknown preset: {name}. "
            f"Available presets: {sorted(PRESET_REGISTRY.keys())}"
        )
    return PRESET_REGISTRY[name]


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        Sorted list of preset names
    """
    return sorted(PRESET_REGISTRY.keys())


# Export all public symbols
__all__ = [
    "InferenceConfig",
    "get_preset",
    "list_presets",
    "PRESET_REGISTRY",
]
