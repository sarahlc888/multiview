"""HuggingFace API provider for embeddings.

Uses the HuggingFace Inference API for embedding models.
"""

import logging
import time

from multiview.constants import HF_API_KEY

logger = logging.getLogger(__name__)


def hf_embedding_completions(
    prompts: list[str],
    model_name: str,
    **kwargs,
) -> dict:
    """Get embeddings from HuggingFace Inference API.

    Args:
        prompts: List of texts to embed (without instructions)
        model_name: Model name on HuggingFace Hub (e.g., "Qwen/Qwen2.5-Embedding-7B")
        **kwargs: Additional parameters including:
            - embed_query_instrs: Optional list of query instructions to prepend
            - embed_doc_instrs: Optional list of document instructions to prepend

    Returns:
        Dict with "completions" key containing list of completion dicts
        Each completion dict has "vector" key with the embedding
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        raise ImportError(
            "huggingface_hub package required. Install with: pip install huggingface_hub"
        ) from None

    # Initialize client with provider="auto" to avoid StopIteration bug
    api_key = HF_API_KEY
    if not api_key:
        logger.warning(
            "HF_TOKEN environment variable not set. Using public API (rate limited)."
        )

    client = (
        InferenceClient(provider="auto", api_key=api_key)
        if api_key
        else InferenceClient(provider="auto")
    )

    # Handle embedding instructions by prepending them to prompts
    embed_query_instrs = kwargs.pop("embed_query_instrs", None)
    embed_doc_instrs = kwargs.pop("embed_doc_instrs", None)

    logger.debug(f"embed_query_instrs: {embed_query_instrs}")
    logger.debug(f"embed_doc_instrs: {embed_doc_instrs}")
    logger.debug(f"prompts (before prepending): {prompts}")

    final_prompts = []
    for i, prompt in enumerate(prompts):
        final_prompt = prompt
        if embed_query_instrs and i < len(embed_query_instrs):
            final_prompt = embed_query_instrs[i] + final_prompt
        if embed_doc_instrs and i < len(embed_doc_instrs):
            final_prompt = embed_doc_instrs[i] + final_prompt
        final_prompts.append(final_prompt)

    logger.debug(f"final_prompts (after prepending): {final_prompts}")

    # Validate instructions were applied correctly
    if embed_query_instrs or embed_doc_instrs:
        for i, (original, final) in enumerate(
            zip(prompts, final_prompts, strict=False)
        ):
            if (
                embed_query_instrs
                and i < len(embed_query_instrs)
                and embed_query_instrs[i]
            ):
                assert final != original, f"Query instruction not applied to prompt {i}"
                assert (
                    embed_query_instrs[i] in final
                ), f"Query instruction not in final prompt {i}"
            if embed_doc_instrs and i < len(embed_doc_instrs) and embed_doc_instrs[i]:
                assert final != original, f"Doc instruction not applied to prompt {i}"
                assert (
                    embed_doc_instrs[i] in final
                ), f"Doc instruction not in final prompt {i}"

    # Batch embeddings for efficiency (process all prompts at once)
    try:
        # Call feature extraction endpoint with batch
        result = client.feature_extraction(
            final_prompts,  # Pass list directly, not text= parameter
            model=model_name,
        )

        # Handle different response formats
        import numpy as np

        if isinstance(result, np.ndarray):
            embeddings = result
        elif isinstance(result, list):
            embeddings = np.array(result)
        else:
            raise ValueError(f"Unexpected response type: {type(result)}")

        # Handle nested arrays (some models return [batch, 1, dim])
        if len(embeddings.shape) == 3:
            embeddings = embeddings.squeeze(1)

        # Convert to list of dicts with Python floats (not numpy types)
        completions = [{"vector": [float(x) for x in emb]} for emb in embeddings]

    except StopIteration:
        logger.error(
            f"StopIteration error from HuggingFace API. This is likely caused by an unsupported model: {model_name}. "
            f"Check that the model is available via HF Inference API at https://huggingface.co/{model_name}"
        )
        # Return empty vectors
        completions = [{"vector": []} for _ in final_prompts]
    except Exception as e:
        logger.warning(f"Error getting embeddings: {e}. Retrying after 5s...")
        time.sleep(5)
        try:
            result = client.feature_extraction(
                final_prompts,
                model=model_name,
            )
            import numpy as np

            if isinstance(result, np.ndarray):
                embeddings = result
            elif isinstance(result, list):
                embeddings = np.array(result)
            else:
                embeddings = np.array([[]] * len(final_prompts))

            if len(embeddings.shape) == 3:
                embeddings = embeddings.squeeze(1)

            completions = [{"vector": [float(x) for x in emb]} for emb in embeddings]
        except Exception as e2:
            logger.error(f"Error getting embeddings after retry: {e2}")
            # Return empty vectors on error
            completions = [{"vector": []} for _ in final_prompts]

    return {"completions": completions}
