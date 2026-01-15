"""Main inference engine.

This module provides the main annotate() function, which is the primary
interface for running inference with LMs or embedding models.

Ported from old query_annotator() with simplifications.
"""

from __future__ import annotations

import logging

from multiview.constants import INFERENCE_CACHE_DIR, USE_CACHE
from multiview.inference.caching import (
    cached_fn_completions,
    get_cache_hash,
    load_cached_completions,
)
from multiview.inference.parsers import get_parser
from multiview.inference.presets import InferenceConfig, get_preset
from multiview.inference.prompt_logger import log_batch_prompts_responses
from multiview.inference.prompts import format_prompts
from multiview.inference.providers import get_completion_fn

logger = logging.getLogger(__name__)


def generate_cache_path_if_needed(
    *,
    config,
    cache_path: str | None,
    cache_alias: str | None,
    run_name: str | None = None,
) -> str | None:
    """Generate a cache_path based on config + cache_alias, if needed.

    This is a thin wrapper around `get_cache_hash()` that keeps cache naming logic
    out of the main inference loop.

    Args:
        config: InferenceConfig object
        cache_path: Explicit cache path (if provided, returned as-is)
        cache_alias: Human-readable cache alias for naming
        run_name: Optional experiment/run name for subdirectory organization

    Returns:
        Cache path string, or None if caching is disabled
    """
    if cache_path is not None or cache_alias is None:
        return cache_path

    config_dict = {
        "provider": config.provider,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "prompt_template": config.prompt_template,
        "embed_query_instr_template": config.embed_query_instr_template,
        "embed_doc_instr_template": config.embed_doc_instr_template,
        "force_prefill_template": config.force_prefill_template,
        "parser": config.parser,
        "parser_kwargs": config.parser_kwargs,
    }
    cache_hash = get_cache_hash(config_dict, cache_alias)

    # Use run_name as subdirectory if provided (for experiment isolation)
    # Otherwise use flat directory structure (backward compatible)
    if run_name:
        cache_dir = INFERENCE_CACHE_DIR / run_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / f"{cache_hash}.json")
    else:
        return str(INFERENCE_CACHE_DIR / f"{cache_hash}.json")


def run_inference(
    inputs: dict[str, list],
    config: InferenceConfig | str,
    cache_path: str | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
    force_refresh: bool = False,
    chunk_size: int | None = 2048,
    verbose: bool = False,
    return_raw: bool = False,
    **config_overrides,
) -> list | tuple[list, list]:
    """Run inference on inputs using a language model or embedding model.

    This is the main inference function, similar to query_annotator() from the old repo.
    It handles:
    - Prompt formatting from inputs
    - Deduplication for efficiency
    - Caching to disk
    - Calling the appropriate provider (OpenAI, Anthropic, HF API)
    - Parsing outputs

    Args:
        inputs: Dictionary of input lists. Keys should match template variables.
            Example: {"documents": ["text1", "text2"], "criterion": "word_count"}
        config: InferenceConfig or preset name (e.g., "openai_embedding_large")
        cache_path: Path to cache file. If None, generates from config hash.
        cache_alias: Human-readable alias for cache file naming.
        run_name: Optional experiment/run name for cache subdirectory organization.
        force_refresh: If True, ignore cache and recompute all.
        chunk_size: Optional max number of prompts per provider call.
        verbose: Whether to log verbose output.
        return_raw: If True, return (parsed, raw) tuple instead of just parsed.
        **config_overrides: Override config fields (e.g., temperature=0.5)

    Returns:
        List of parsed outputs (length matches input lists), or
        Tuple of (parsed_outputs, raw_completions) if return_raw=True

    Example:
        >>> # Using a preset
        >>> results = run_inference(
        ...     inputs={"documents": ["Hello world", "Goodbye world"]},
        ...     config="openai_embedding_large",
        ... )
        >>> # Using a custom config
        >>> from multiview.inference.presets import InferenceConfig
        >>> config = InferenceConfig(
        ...     provider="openai",
        ...     model_name="gpt-4.1-mini",
        ...     prompt_template="Analyze: {document}\nCriterion: {criterion}",
        ...     parser="json",
        ... )
        >>> results = run_inference(
        ...     inputs={"documents": ["Good text"], "criterion": "sentiment"},
        ...     config=config,
        ... )
    """
    # Load config if string preset name
    if isinstance(config, str):
        config = get_preset(config)

    # Apply overrides
    if config_overrides:
        config = config.with_overrides(**config_overrides)

    # Generate cache path if not provided
    cache_path = generate_cache_path_if_needed(
        config=config,
        cache_path=cache_path,
        cache_alias=cache_alias,
        run_name=run_name,
    )

    # Respect global USE_CACHE flag
    # If USE_CACHE is False, disable caching by setting cache_path to None
    if not USE_CACHE:
        if verbose:
            logger.info("Caching disabled globally (USE_CACHE=False)")
        cache_path = None

    if verbose and cache_path:
        logger.info(f"Using cache path: {cache_path}")

    # Format prompts from inputs
    prompt_collection = format_prompts(inputs, config, verbose=verbose)

    # Deduplicate prompts for efficiency
    remap_idxs, deduped_prompt_collection = prompt_collection.dedup()
    if len(deduped_prompt_collection.packed_prompts) != len(
        prompt_collection.packed_prompts
    ):
        logger.info(
            f"Deduped from {len(prompt_collection.packed_prompts)} to "
            f"{len(deduped_prompt_collection.packed_prompts)} prompts"
        )

    # Get completion function for this provider
    fn_completions = get_completion_fn(config.provider)

    # Load cache
    completion_cache = load_cached_completions(cache_path)

    # Run completions with caching
    prompt_dict = deduped_prompt_collection.to_dict()
    raw_completions = cached_fn_completions(
        packed_prompts=prompt_dict.pop("packed_prompts"),
        non_packed_prompts=prompt_dict.pop("non_packed_prompts"),
        fn_completions=fn_completions,
        completion_cache=completion_cache,
        completion_cache_path=cache_path,
        force_refresh=force_refresh,
        verbose=verbose,
        chunk_size=chunk_size,
        # Pass remaining prompt collection fields and model params to fn_completions
        **prompt_dict,
        **config.to_completion_kwargs(),
    )

    # Log prompts and responses for auditing
    # Extract prompts for logging
    prompt_texts = []
    if deduped_prompt_collection.packed_prompts:
        prompt_texts.extend(deduped_prompt_collection.packed_prompts)
    if deduped_prompt_collection.prompts:
        prompt_texts.extend(deduped_prompt_collection.prompts)

    log_batch_prompts_responses(
        prompts=prompt_texts,
        responses=raw_completions,
        metadata={
            "provider": config.provider,
            "model_name": config.model_name,
            "config": config.__class__.__name__,
            "cache_alias": cache_alias,
        },
    )

    # Parse completions
    parser_fn = get_parser(config.parser)
    parser_kwargs = config.parser_kwargs or {}

    parsed_completions = []
    for idx, raw_completion in enumerate(raw_completions):
        try:
            parsed = parser_fn(raw_completion, **parser_kwargs)
            parsed_completions.append(parsed)
        except Exception as e:
            # Error details already logged by parser
            logger.warning(
                f"Parse error for completion {idx+1}/{len(raw_completions)}: "
                f"{type(e).__name__}: {e}"
            )
            # Return None or empty on parse error
            parsed_completions.append(None)

    # Re-duplicate to match original input order
    output = [parsed_completions[i] for i in remap_idxs]

    if return_raw:
        # Also return raw completions in original order
        raw_output = [raw_completions[i] for i in remap_idxs]
        return output, raw_output
    else:
        return output
