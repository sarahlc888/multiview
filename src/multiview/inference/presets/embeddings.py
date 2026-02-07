"""Embedding model presets."""

from __future__ import annotations

from ._base import InferenceConfig

# Standard embedding instruction for criterion-aware retrieval
# Includes both criterion name and description for better context
CRITERION_AWARE_EMBED_INSTR = (
    "Given a query, retrieve documents based on the criterion '{criterion}': "
    "{criterion_description}"
)

EMBEDDING_PRESETS = {
    # ========================================================================
    # Gemini Embeddings (text only)
    # ========================================================================
    "gemini_embedding_001": InferenceConfig(
        provider="gemini_embedding",
        model_name="gemini-embedding-001",
        prompt_template="{document}",
        parser="vector",
    ),
    # ========================================================================
    # OpenAI Embeddings
    # ========================================================================
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
    # ========================================================================
    # Voyage AI Embeddings
    # ========================================================================
    "voyage_4_lite": InferenceConfig(
        provider="voyage_embedding",
        model_name="voyage-4-lite",
        prompt_template="{document}",
        parser="vector",
    ),
    "voyage_4_lite_1024": InferenceConfig(
        provider="voyage_embedding",
        model_name="voyage-4-lite",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"output_dimension": 1024},
    ),
    "voyage_4_lite_512": InferenceConfig(
        provider="voyage_embedding",
        model_name="voyage-4-lite",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"output_dimension": 512},
    ),
    "voyage_4_lite_256": InferenceConfig(
        provider="voyage_embedding",
        model_name="voyage-4-lite",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"output_dimension": 256},
    ),
    # ========================================================================
    # Voyage AI Embeddings (with criterion-aware instruction)
    # ========================================================================
    "instr_voyage_4_lite": InferenceConfig(
        provider="voyage_embedding",
        model_name="voyage-4-lite",
        prompt_template="{document}",
        instruction=CRITERION_AWARE_EMBED_INSTR,
        parser="vector",
    ),
    "instr_voyage_4_lite_1024": InferenceConfig(
        provider="voyage_embedding",
        model_name="voyage-4-lite",
        prompt_template="{document}",
        instruction=CRITERION_AWARE_EMBED_INSTR,
        parser="vector",
        extra_kwargs={"output_dimension": 1024},
    ),
    # ========================================================================
    # Voyage AI Multimodal Embeddings (text + images via multimodal_embed)
    # ========================================================================
    "voyage_multimodal_3_5": InferenceConfig(
        provider="voyage_multimodal_embedding",
        model_name="voyage-multimodal-3.5",
        prompt_template="{document}",
        parser="vector",
    ),
    "voyage_multimodal_3_5_1024": InferenceConfig(
        provider="voyage_multimodal_embedding",
        model_name="voyage-multimodal-3.5",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"output_dimension": 1024},
    ),
    # ========================================================================
    # HuggingFace API Embeddings (without instruction prefix)
    # ========================================================================
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
    # ========================================================================
    # HuggingFace API Embeddings (with criterion-aware instruction)
    # ========================================================================
    "instr_hf_qwen3_embedding_8b": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-8B",
        prompt_template="{document}",
        instruction=CRITERION_AWARE_EMBED_INSTR,
        parser="vector",
    ),
    "instr_hf_qwen3_embedding_4b": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-4B",
        prompt_template="{document}",
        instruction=CRITERION_AWARE_EMBED_INSTR,
        parser="vector",
    ),
    # ========================================================================
    # Specialized embedding preset for plain text retrieval via HF API
    # ========================================================================
    "embed_plaintext_hfapi": InferenceConfig(
        provider="hf_embedding",
        model_name="Qwen/Qwen3-Embedding-8B",
        prompt_template="{document}",
        parser="vector",
    ),
    # ========================================================================
    # ColBERT Multi-Vector Embeddings (local GPU)
    # ========================================================================
    "colbert_reason_modern": InferenceConfig(
        provider="hf_local_colbert",
        model_name="lightonai/Reason-ModernColBERT",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"device": "cuda", "batch_size": 8, "max_length": 512},
    ),
    "colbert_jina_v2": InferenceConfig(
        provider="hf_local_colbert",
        model_name="jinaai/jina-colbert-v2",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"device": "cuda", "batch_size": 8, "max_length": 512},
    ),
    "colbert_gte_modern_v1": InferenceConfig(
        provider="hf_local_colbert",
        model_name="lightonai/GTE-ModernColBERT-v1",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"device": "cuda", "batch_size": 8, "max_length": 512},
    ),
    # ========================================================================
    # ColBERT with instruction support
    # ========================================================================
    "instr_colbert_reason_modern": InferenceConfig(
        provider="hf_local_colbert",
        model_name="lightonai/Reason-ModernColBERT",
        prompt_template="{document}",
        parser="vector",
        extra_kwargs={"device": "cuda", "batch_size": 8, "max_length": 512},
    ),
}
