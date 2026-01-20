"""Reranker model presets."""

from __future__ import annotations

from ._base import InferenceConfig

RERANKER_PRESETS = {
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
    "contextual_reranker_6b": InferenceConfig(
        provider="hf_local_contextual_reranker",
        model_name="ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b",
        prompt_template="Check whether a given document contains information helpful to answer the query.\n<Document> {document}\n<Query> {query}{instruction} ??",
        parser="score",
        extra_kwargs={
            "device": "cuda",
            "batch_size": 32,
            "max_length": 8192,
        },
    ),
    "contextual_reranker_6b_cpu": InferenceConfig(
        provider="hf_local_contextual_reranker",
        model_name="ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b",
        prompt_template="Check whether a given document contains information helpful to answer the query.\n<Document> {document}\n<Query> {query}{instruction} ??",
        parser="score",
        extra_kwargs={
            "device": "cpu",
            "batch_size": 16,
            "max_length": 8192,
        },
    ),
    "voyage_rerank_2_5_lite": InferenceConfig(
        provider="voyage_reranker",
        model_name="rerank-2.5-lite",
        prompt_template="{document}",  # Voyage expects raw documents, not formatted prompts
        parser="score",
        extra_kwargs={
            "truncation": True,
        },
    ),
    "voyage_rerank_2_5": InferenceConfig(
        provider="voyage_reranker",
        model_name="rerank-2.5",
        prompt_template="{document}",  # Voyage expects raw documents, not formatted prompts
        parser="score",
        extra_kwargs={
            "truncation": True,
        },
    ),
}
