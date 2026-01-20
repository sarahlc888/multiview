"""Voyage AI provider for reranking and embeddings.

Provides access to Voyage AI's reranking and embedding models via their Python SDK.
"""

from __future__ import annotations

import logging
from typing import Any

from multiview.constants import VOYAGE_API_KEY
from multiview.inference.cost_tracker import record_usage

logger = logging.getLogger(__name__)


def _get_voyage_client():
    """Get Voyage AI client with API key from environment."""
    try:
        import voyageai
    except ImportError:
        raise ImportError(
            "voyageai package required. Install with: pip install voyageai"
        ) from None

    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY environment variable not set")

    return voyageai.Client(api_key=VOYAGE_API_KEY)


def voyage_reranker_completions(
    prompts: list[str],
    model_name: str,
    queries: list[str],
    top_k: int | None = None,
    truncation: bool = True,
    instructions: list[str] | None = None,
    **kwargs,
) -> dict[str, list[dict[str, Any]]]:
    # breakpoint()
    """Get reranker scores from Voyage AI API.

    Args:
        prompts: List of raw documents (without instructions)
        model_name: Voyage model name (e.g., "rerank-2.5-lite", "rerank-2.5")
        queries: List of queries, one per document (for triplet evaluation)
                 If you have a single query for all docs, pass it as a list repeated
        top_k: Optional number of top results to return
        truncation: Whether to truncate oversized inputs (default: True)
        instructions: List of instructions to prepend to queries (one per query)
        **kwargs: Additional parameters

    Returns:
        Dict with "completions" key containing list of completion dicts.
        Each completion dict has "score" key with the relevance score.

    Note:
        Unlike other reranker providers, Voyage expects raw documents and queries
        as separate parameters. Instructions are prepended to queries using:
        f"{instruction}\nQuery: {query}"

        Queries are automatically grouped by unique values for efficient API calls,
        so passing the same query repeated multiple times is fine.
    """
    client = _get_voyage_client()

    # Use prompts directly as documents (they are already non-packed/raw)
    documents = prompts

    if len(queries) != len(documents):
        raise ValueError(
            f"Length mismatch: {len(queries)} queries but {len(documents)} documents"
        )

    # Prepend instructions to queries if provided
    formatted_queries = queries
    if instructions:
        if len(instructions) != len(queries):
            raise ValueError(
                f"Length mismatch: {len(instructions)} instructions "
                f"but {len(queries)} queries"
            )
        formatted_queries = [
            f"{instr}\nQuery: {q}" if instr else q
            for instr, q in zip(instructions, queries, strict=True)
        ]
        logger.debug(f"Formatted query with instruction: {formatted_queries[0][:200]}")

    # Group documents by query for efficient API calls
    # Voyage API takes one query and multiple documents
    from collections import defaultdict

    query_to_doc_indices = defaultdict(list)
    for i, q in enumerate(formatted_queries):
        query_to_doc_indices[q].append(i)

    logger.info(
        f"Voyage reranker: processing {len(documents)} documents "
        f"across {len(query_to_doc_indices)} unique queries"
    )
    logger.info(f"Model: {model_name}")

    # Initialize scores array
    scores = [0.0] * len(documents)

    # Process each unique query
    for query_text, doc_indices in query_to_doc_indices.items():
        docs_for_query = [documents[i] for i in doc_indices]

        # Call Voyage rerank API
        try:
            reranking = client.rerank(
                query=query_text,
                documents=docs_for_query,
                model=model_name,
                top_k=top_k,
                truncation=truncation,
            )
        except Exception as e:
            logger.error(f"Voyage API error: {e}")
            raise

        # Map scores back to original indices
        logger.info(f"{reranking.results=}")
        for result in reranking.results:
            original_idx = doc_indices[result.index]
            scores[original_idx] = result.relevance_score

    logger.info(f"Computed {len(scores)} reranker scores")

    # Format as completions
    completions = [{"score": float(score)} for score in scores]

    return {"completions": completions}


def voyage_embedding_completions(
    prompts: list[str],
    model_name: str,
    **kwargs,
) -> dict[str, list[dict[str, Any]]]:
    """Get embeddings from Voyage AI API.

    Args:
        prompts: List of texts to embed (without instructions)
        model_name: Voyage model name (e.g., "voyage-4-lite", "voyage-3-lite")
        **kwargs: Additional parameters including:
            - instructions: Optional list of instructions to prepend to prompts
            - input_type: Optional type hint ("document" or "query")
            - output_dimension: Optional output dimension (e.g., 256, 512, 1024, 2048)
            - output_dtype: Optional output data type ("float", "int8", "uint8", "binary", "ubinary")
            - truncation: Whether to truncate oversized inputs (default: True)

    Returns:
        Dict with "completions" key containing list of completion dicts.
        Each completion dict has "vector" key with the embedding.
    """
    client = _get_voyage_client()

    # Handle embedding instructions by prepending them to prompts
    instructions = kwargs.pop("instructions", None)

    # Strip internal/provider-level arguments that must not be forwarded to the Voyage SDK
    kwargs.pop("max_retries", None)
    kwargs.pop("initial_retry_delay", None)
    kwargs.pop("retry_backoff_factor", None)
    kwargs.pop("max_workers", None)

    final_prompts = []
    for i, prompt in enumerate(prompts):
        final_prompt = prompt
        if instructions and i < len(instructions):
            final_prompt = instructions[i] + final_prompt
        final_prompts.append(final_prompt)

    # Filter out parameters not relevant to embeddings API
    embedding_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]
    }

    try:
        # Call Voyage embeddings API
        response = client.embed(
            texts=final_prompts,
            model=model_name,
            **embedding_kwargs,
        )

        # Record usage if available
        if hasattr(response, "total_tokens") and response.total_tokens:
            record_usage(
                model_name=model_name,
                input_tokens=response.total_tokens,
                output_tokens=0,  # Embeddings don't have output tokens
            )

        # Extract embeddings
        completions = [{"vector": emb} for emb in response.embeddings]

        logger.info(f"Generated {len(completions)} embeddings using {model_name}")

        return {"completions": completions}

    except Exception as e:
        logger.error(f"Error getting Voyage embeddings: {e}")
        # Return empty vectors on error
        return {"completions": [{"vector": []} for _ in final_prompts]}
