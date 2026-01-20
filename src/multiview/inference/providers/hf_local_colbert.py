"""HuggingFace local ColBERT provider for multi-vector embeddings.

This provider loads ColBERT models locally using the pylate library.
Supports MaxSim-based retrieval with multi-vector embeddings.
"""

import gc
import logging
from collections.abc import Sequence

from multiview.constants import HF_CACHE_DIR

logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_MODEL_CACHE = {}
_MODEL_NAME_CACHE = None


def _make_batches_by_instruction(
    prompts: list[str],
    instructions: list[str | None],
    batch_size: int,
) -> tuple[list[list[str]], list[list[str | None]]]:
    """Batch prompts by instruction type to maximize efficiency.

    Groups prompts with the same instruction together, then creates batches.

    Args:
        prompts: List of prompts to batch
        instructions: List of instructions (one per prompt, can be None)
        batch_size: Maximum batch size

    Returns:
        Tuple of (batched_prompts, batched_instructions)
    """
    # Sort by instruction type, keeping track of original order
    original_order, packed = zip(
        *sorted(
            enumerate(zip(prompts, instructions, strict=False)),
            key=lambda x: (x[1][1] or "", len(x[1][0])),
        ),
        strict=False,
    )
    sorted_prompts, sorted_instructions = zip(*packed, strict=False)
    sorted_prompts = list(sorted_prompts)
    sorted_instructions = list(sorted_instructions)

    # Create batches
    batched_prompts = []
    batched_instructions = []

    for i in range(0, len(sorted_prompts), batch_size):
        batch_prompts = sorted_prompts[i : i + batch_size]
        batch_instructions = sorted_instructions[i : i + batch_size]

        batched_prompts.append(batch_prompts)
        batched_instructions.append(batch_instructions)

    return batched_prompts, batched_instructions, original_order


def hf_local_colbert_completions(
    prompts: list[str],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 512,
    instructions: Sequence[str | None] | None = None,
    **kwargs,
) -> dict:
    """Get multi-vector ColBERT embeddings from local model.

    Args:
        prompts: List of text prompts to encode
        model_name: Model name on HuggingFace Hub (supported: lightonai/Reason-ModernColBERT,
                    jinaai/jina-colbert-v2, lightonai/GTE-ModernColBERT-v1)
        device: Device to run model on ("cuda" or "cpu")
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        instructions: Optional instructions (one per prompt or None)
        **kwargs: Additional parameters

    Returns:
        Dict with "completions" key containing list of completion dicts.
        Each completion dict has "vector" key with multi-vector embedding (shape: [num_tokens, dim])
    """
    try:
        import torch
        from pylate.models import ColBERT
    except ImportError as e:
        raise ImportError(
            "pylate, transformers, and torch packages required. "
            "Install with: pip install pylate transformers torch"
        ) from e

    # Validate model name
    supported_models = [
        "lightonai/Reason-ModernColBERT",
        "jinaai/jina-colbert-v2",
        "lightonai/GTE-ModernColBERT-v1",
    ]
    if model_name not in supported_models:
        logger.warning(
            f"Model {model_name} not in tested models {supported_models}. "
            f"Proceeding anyway, but behavior may be unexpected."
        )

    global _MODEL_NAME_CACHE

    # Process instructions: use document mode when instructions provided
    if instructions is not None:
        instructions = list(instructions)
        is_query = False  # Document mode for symmetric retrieval
    else:
        # Default: all None instructions, query mode
        instructions = [None] * len(prompts)
        is_query = True

    # Ensure instruction list matches prompts length
    if len(instructions) != len(prompts):
        raise ValueError(
            f"Instruction list length ({len(instructions)}) must match "
            f"prompts length ({len(prompts)})"
        )

    # Batch by instruction type for efficiency
    batched_prompts, batched_instructions, original_order = (
        _make_batches_by_instruction(prompts, instructions, batch_size)
    )

    # Load model (with caching)
    cache_key = model_name

    if cache_key not in _MODEL_CACHE or _MODEL_NAME_CACHE != model_name:
        # Clear cache if switching models
        if _MODEL_NAME_CACHE is not None and _MODEL_NAME_CACHE != model_name:
            logger.info(
                f"Switching from {_MODEL_NAME_CACHE} to {model_name}, clearing cache"
            )
            _MODEL_CACHE.clear()
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        logger.info("=" * 70)
        logger.info(f"Loading ColBERT model: {model_name}")
        logger.info(f"Cache directory: {HF_CACHE_DIR}")
        logger.info("=" * 70)

        # Model-specific configuration
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_folder": str(HF_CACHE_DIR),
        }

        # Special handling for jina-colbert-v2
        if model_name == "jinaai/jina-colbert-v2":
            model_kwargs["query_prefix"] = "[QueryMarker]"
            model_kwargs["document_prefix"] = "[DocumentMarker]"
            model_kwargs["attend_to_expansion_tokens"] = True
            model_kwargs["trust_remote_code"] = True

        # Separate model_kwargs from ColBERT kwargs
        hf_model_kwargs = {}
        colbert_kwargs = {}

        for key in ["torch_dtype", "attn_implementation"]:
            if key in model_kwargs:
                hf_model_kwargs[key] = model_kwargs.pop(key)

        colbert_kwargs = model_kwargs

        # Load model
        logger.info(
            f"Loading model with HF kwargs: {hf_model_kwargs}, ColBERT kwargs: {colbert_kwargs}"
        )
        model = ColBERT(
            model_name_or_path=model_name,
            model_kwargs=hf_model_kwargs,
            **colbert_kwargs,
        )

        # Compile model for better performance
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

        logger.info("✓ Model loaded and compiled")

        # Cache model
        _MODEL_CACHE[cache_key] = model
        _MODEL_NAME_CACHE = model_name
    else:
        logger.info(f"Using cached ColBERT model: {model_name}")
        model = _MODEL_CACHE[cache_key]

    # Process batches
    all_embeddings = []

    logger.info(
        f"Processing {len(batched_prompts)} batches with batch_size={batch_size}"
    )

    with torch.inference_mode():
        for batch_idx, (batch_prompts, batch_instructions) in enumerate(
            zip(batched_prompts, batched_instructions, strict=False)
        ):
            # All instructions in a batch should be the same (due to batching strategy)
            unique_instructions = set(batch_instructions)
            if len(unique_instructions) > 1:
                logger.warning(
                    f"Batch {batch_idx} has multiple instruction types: {unique_instructions}. "
                    f"This may reduce efficiency."
                )

            # Get the instruction for this batch (should be consistent)
            batch_instruction = batch_instructions[0]

            # Encode batch
            # Shape: (batch_size, max_tokens, embedding_dim)
            embeddings = model.encode(
                sentences=batch_prompts,
                prompt=batch_instruction,  # Can be None
                batch_size=len(batch_prompts),  # Process entire batch at once
                normalize_embeddings=True,  # Normalize for dot product = cosine similarity
                is_query=is_query,
                show_progress_bar=False,
            )

            # Convert to list of individual embeddings
            for j in range(len(embeddings)):
                all_embeddings.append(embeddings[j])

    # Reorder embeddings to match original prompt order
    reordered_embeddings = [None] * len(prompts)
    for emb, orig_idx in zip(all_embeddings, original_order, strict=False):
        reordered_embeddings[orig_idx] = emb

    logger.info(
        f"Generated {len(reordered_embeddings)} ColBERT embeddings "
        f"(shape: {reordered_embeddings[0].shape if reordered_embeddings else 'N/A'})"
    )

    # Format as completions (return numpy arrays directly - they'll be handled by similarity.py)
    completions = [{"vector": emb} for emb in reordered_embeddings]

    return {"completions": completions}
