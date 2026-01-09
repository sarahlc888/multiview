"""Embedding model presets.

This module defines preset configurations for embedding models from various providers.
"""

from ._base import InferenceConfig

# ============================================================================
# OPENAI EMBEDDINGS
# ============================================================================

OPENAI_EMBEDDING_LARGE = InferenceConfig(
    provider="openai",
    model_name="text-embedding-3-large",
    prompt_template="{document}",
    is_embedding=True,
    parser="vector",
)

OPENAI_EMBEDDING_SMALL = InferenceConfig(
    provider="openai",
    model_name="text-embedding-3-small",
    prompt_template="{document}",
    is_embedding=True,
    parser="vector",
)

# ============================================================================
# HUGGINGFACE API EMBEDDINGS
# ============================================================================

HF_QWEN3_EMBEDDING_8B = InferenceConfig(
    provider="hf_api",
    model_name="Qwen/Qwen3-Embedding-8B",
    prompt_template="{document}",
    embed_query_instr_template="Represent this query for retrieval: ",
    is_embedding=True,
    parser="vector",
)

HF_QWEN3_EMBEDDING_4B = InferenceConfig(
    provider="hf_api",
    model_name="Qwen/Qwen3-Embedding-4B",
    prompt_template="{document}",
    embed_query_instr_template="Represent this query for retrieval: ",
    is_embedding=True,
    parser="vector",
)

# Specialized embedding preset for plain text retrieval via HF API
# Note: Qwen3-Embedding-4B is not available via HF Inference API, using 8B instead
EMBED_PLAINTEXT_HFAPI = InferenceConfig(
    provider="hf_api",
    model_name="Qwen/Qwen3-Embedding-8B",
    prompt_template="{document}",
    embed_query_instr_template="Represent this query for retrieval: ",
    is_embedding=True,
    parser="vector",
)
