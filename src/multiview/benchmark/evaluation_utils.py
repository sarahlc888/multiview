"""Benchmark evaluation helpers.

This module keeps `Benchmark` lean by encapsulating method dispatch and shared
evaluation utilities (triplet dict construction, annotation gating, etc.).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any

from omegaconf import OmegaConf

from multiview.eval import (
    evaluate_with_bm25,
    evaluate_with_document_rewrite,
    evaluate_with_embeddings,
    evaluate_with_in_one_word,
    evaluate_with_lm_judge_pair,
    evaluate_with_lm_judge_triplet,
    evaluate_with_multisummary,
    evaluate_with_pseudologit,
    evaluate_with_query_relevance_vectors,
    evaluate_with_reranker,
)
from multiview.inference.presets import is_gpu_available, preset_requires_gpu

logger = logging.getLogger(__name__)


def _make_class_label(i: int) -> str:
    """Generate a class label for index i. A-Z for 0-25, then A1, B1, ... Z1, A2, etc."""
    if i < 26:
        return chr(65 + i)
    suffix = i // 26
    return f"{chr(65 + i % 26)}{suffix}"


def _schema_to_classes_data(items: list[dict], name_key: str = "name") -> dict:
    """Convert a list of schema items (categories or tags) into a classes_data dict.

    Args:
        items: List of dicts with 'name' and 'description' keys.
        name_key: Key to use for the item name (default: "name").

    Returns:
        Dict with classes, descriptions, taxonomy_context, and instruction.
    """
    classes = [_make_class_label(i) for i in range(len(items))]
    descriptions = {
        _make_class_label(i): item["description"] for i, item in enumerate(items)
    }
    taxonomy_lines = []
    for i, item in enumerate(items):
        label = _make_class_label(i)
        taxonomy_lines.append(f"{label}. {item[name_key]}\n{item['description']}\n")
    taxonomy_context = "\n".join(taxonomy_lines)
    return {
        "classes": classes,
        "descriptions": descriptions,
        "taxonomy_context": taxonomy_context,
        "instruction": (
            "Instruction: think out loud and explain your reasoning, then choose "
            "the SINGLE best label. For the final judgement, use this format exactly: "
            "'The final answer is $\\boxed{<label>}$'"
        ),
    }


def build_triplet_dicts(
    documents: list[str],
    triplet_ids: list[tuple[int, int, int]],
) -> list[dict]:
    """Convert ID triplets into text triplet dicts (doc IDs NEVER in prompts!).

    Validates that anchor, positive, and negative are all distinct indices.
    Filters out any invalid triplets and logs warnings.
    """
    valid_triplets = []
    invalid_count = 0

    for triplet_idx, (anchor_id, positive_id, negative_id) in enumerate(triplet_ids):
        # Validate that all three indices are distinct
        if anchor_id == positive_id:
            logger.warning(
                f"Skipping invalid triplet {triplet_idx}: anchor_id == positive_id ({anchor_id})"
            )
            invalid_count += 1
            continue

        if anchor_id == negative_id:
            logger.warning(
                f"Skipping invalid triplet {triplet_idx}: anchor_id == negative_id ({anchor_id})"
            )
            invalid_count += 1
            continue

        if positive_id == negative_id:
            logger.warning(
                f"Skipping invalid triplet {triplet_idx}: positive_id == negative_id ({positive_id})"
            )
            invalid_count += 1
            continue

        # Triplet is valid
        valid_triplets.append(
            {
                "anchor": documents[anchor_id],
                "positive": documents[positive_id],
                "negative": documents[negative_id],
                # Include IDs for annotation lookup / artifact writing
                "anchor_id": anchor_id,
                "positive_id": positive_id,
                "negative_id": negative_id,
            }
        )

    if invalid_count > 0:
        logger.error(
            f"⚠️  Filtered out {invalid_count} invalid triplet(s) where anchor/positive/negative were not distinct "
            f"({len(triplet_ids)} -> {len(valid_triplets)} triplets)"
        )

    return valid_triplets


def _resolved_document_type(task: Any) -> str | None:
    """Single source of truth for document type used in evaluation.

    Prefer Task's metadata-aware resolver when available.
    Falls back to docset's DOCUMENT_TYPE if config doesn't specify.
    """
    resolver = getattr(task, "_resolved_document_type", None)
    if callable(resolver):
        return resolver()

    # Check config first
    config_doc_type = getattr(task, "config", {}).get("document_type")
    if config_doc_type:
        return config_doc_type

    # Fall back to docset's DOCUMENT_TYPE attribute
    docset = getattr(task, "document_set", None)
    if docset:
        return getattr(docset, "DOCUMENT_TYPE", None)

    return None


def preset_requires_annotations(preset: str) -> bool:
    return "with_annotation" in preset.lower()


def has_rich_annotations(annotations: list[dict] | None) -> bool:
    """Check if annotations contain rich information (summaries, tags, etc.).

    Returns:
        True if annotations have rich fields beyond just prelabel
        False if annotations only have prelabel (simple precomputed annotations)
    """
    if not annotations or len(annotations) == 0:
        return False

    # Check first annotation for rich fields
    first_ann = annotations[0]

    # Rich annotations have at least one of: summary, tags, category
    has_summary = "summary" in first_ann and first_ann.get("summary")
    has_tags = "tags" in first_ann and first_ann.get("tags")
    has_category = "category" in first_ann and first_ann.get("category")

    return has_summary or has_tags or has_category


def has_category_schemas(annotations: list[dict] | None) -> bool:
    """Check if annotations contain category schema information.

    Returns:
        True if annotations have category_schema field
        False otherwise
    """
    if not annotations or len(annotations) == 0:
        return False

    # Check first annotation for category_schema
    first_ann = annotations[0]
    return "category_schema" in first_ann and first_ann.get("category_schema")


def get_annotations_if_required(*, preset: str, task: Any) -> dict | None:
    """Return task annotations only if the preset expects them (else None).

    Also checks if annotations are rich enough for the preset - simple precomputed
    annotations (only prelabel) are not sufficient for presets that need
    summaries/tags.
    """
    if not preset_requires_annotations(preset):
        return None

    if task.document_annotations is None:
        logger.warning(
            f"Preset '{preset}' requires annotations but task has none. "
            "Skipping this method."
        )
        return None

    # Check if annotations are rich enough
    if not has_rich_annotations(task.document_annotations):
        logger.warning(
            f"Preset '{preset}' requires rich annotations (summaries/tags), "
            f"but task only has simple criterion values. "
            f"Skipping this method. Use presets without 'with_annotation' for simple annotations."
        )
        return None

    logger.info("Using annotations for evaluation (preset requires them)")
    return task.document_annotations


def _to_plain_python(obj: Any) -> Any:
    """Recursively convert OmegaConf objects to plain Python types.

    Args:
        obj: Object to convert (can be OmegaConf object or regular Python object)

    Returns:
        Plain Python object (dict, list, or primitive type)
    """
    # Check if it's an OmegaConf object
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True, throw_on_missing=True)
    # Recursively handle dicts
    elif isinstance(obj, dict):
        return {k: _to_plain_python(v) for k, v in obj.items()}
    # Recursively handle lists
    elif isinstance(obj, list):
        return [_to_plain_python(item) for item in obj]
    # Return primitives as-is
    else:
        return obj


def get_config_hash_suffix(method_config: dict) -> str:
    """Generate short hash suffix from config fields that affect evaluation.

    This creates a unique identifier for method configs based on ALL fields
    that would affect the evaluation results (preset, overrides, custom params).
    This ensures that different configurations get different cache keys and
    result names.

    Args:
        method_config: Method configuration dict

    Returns:
        Empty string if only standard config (name + preset), otherwise "_<hash>" (6 chars)
    """
    # Hash everything except "name" (which is just for display)
    # Include: preset, preset_overrides, and any custom fields
    hashable_fields = {k: v for k, v in method_config.items() if k not in ["name"]}

    # Only add hash suffix if there's more than just the preset
    # (i.e., if there are overrides or custom fields)
    has_custom_config = False
    if "preset_overrides" in hashable_fields and hashable_fields["preset_overrides"]:
        has_custom_config = True

    # Check for custom fields beyond preset and preset_overrides
    custom_fields = {
        k: v
        for k, v in hashable_fields.items()
        if k not in ["preset", "preset_overrides"]
    }
    if custom_fields:
        has_custom_config = True

    if has_custom_config:
        # Convert any OmegaConf objects to regular Python objects for JSON serialization
        hashable_fields = _to_plain_python(hashable_fields)
        # Create short hash of full config (6 chars for readability)
        config_str = json.dumps(hashable_fields, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        return f"_{config_hash}"

    return ""


def make_cache_alias(*, task: Any, method_config: dict, default_name: str) -> str:
    """Generate cache alias with hash of custom config fields."""
    base_name = method_config.get("name", default_name)
    hash_suffix = get_config_hash_suffix(method_config)
    return f"{task.get_task_name()}_eval_{base_name}{hash_suffix}"


def _extract_document_texts(task: Any) -> list[str]:
    """Extract text from task documents.

    Args:
        task: Task object with documents and document_set

    Returns:
        List of document texts
    """
    return [task.document_set.get_document_text(doc) for doc in task.documents]


def _prepare_instruction_overrides(
    method_config: dict, criterion_metadata: dict
) -> dict:
    """Prepare preset overrides with instruction field.

    Handles instruction from criterion metadata or method config.
    Priority: method_config["embed_instr"] > criterion_metadata["embed_instr"]

    Args:
        method_config: Method configuration dict
        criterion_metadata: Criterion metadata dict

    Returns:
        Dict with instruction preset override (if applicable)
    """
    preset_overrides = method_config.get("preset_overrides", {})
    if not isinstance(preset_overrides, dict):
        preset_overrides = {}

    # Check if preset name indicates instruction usage (e.g., "instr_*" prefix)
    preset = method_config.get("preset", "")
    wants_instructions = "instr" in preset.lower()

    # Support embed_instr from criterion metadata, but only if:
    # 1. The preset indicates it wants instructions (name contains "instr"), OR
    # 2. The method config explicitly provides embed_instr (handled below)
    # This prevents non-instruction presets from automatically getting instructions
    if "embed_instr" in criterion_metadata and wants_instructions:
        preset_overrides["instruction"] = criterion_metadata["embed_instr"]

    # Support convenient embed_instr shorthand in method config (always applies if specified)
    # This allows methods to explicitly opt into instructions even if preset doesn't have "instr"
    if "embed_instr" in method_config:
        preset_overrides["instruction"] = method_config["embed_instr"]

    return preset_overrides if preset_overrides else None


def _get_default_cache_name(method_type: str, method_config: dict) -> str:
    """Get the default cache name for a method type.

    Most methods use their type as the cache name, but some have special cases.
    """
    if method_type == "lm_judge_triplet":
        return "lmjudge"
    elif method_type == "lm_judge_pair":
        return "lmjudge_pair"
    elif method_type == "document_rewrite":
        embedding_preset = method_config.get("embedding_preset", "bm25_lexical")
        return f"dr_{embedding_preset}"
    elif method_type == "query_expansion_bm25":
        return "qe_bm25"
    elif method_type == "query_relevance_vectors":
        return "qrv"
    elif method_type == "in_one_word":
        return "inoneword"
    else:
        return method_type


def evaluate_method(
    *, method_type: str, task: Any, method_config: dict
) -> dict[str, Any]:
    """Dispatch evaluation for a single method config on a single task.

    Args:
        method_type: Type of evaluation method (e.g., "embeddings", "pseudologit", "in_one_word")
        task: Task object with documents and triplets
        method_config: Method configuration dict

    Returns:
        Dict with evaluation results including:
        - method_name: Display name with config hash suffix
        - accuracy: Fraction of correct triplet judgments
        - n_correct: Number of correct triplets
        - n_incorrect: Number of incorrect triplets
        - n_ties: Number of tied triplets
        - Additional method-specific fields

        OR for multi-trial configs (when num_trials > 1):
        - _multi_trial: True
        - trials: List of result dicts, one per trial

    Example method configs:
        pseudologit:
            {
                "name": "pseudologit",
                "preset": "pseudologit_gemini_n50",
                "classes_file": "prompts/custom/gsm8k_classes.json"
            }

        in_one_word (with config categories):
            {
                "name": "inoneword_custom",
                "preset": "inoneword_hf_qwen3_8b",
                "categories": ["addition", "subtraction", "multiplication", "other"],
                "category_context": "Classify the math problem type"
            }

        in_one_word (with annotations):
            {
                "name": "inoneword_from_annotations",
                "preset": "inoneword_hf_qwen3_8b"
                # Will extract categories from task annotations
            }

        in_one_word (with multiple trials):
            {
                "name": "inoneword_proposed",
                "preset": "inoneword_hf_qwen3_8b",
                "generate_schema": true,
                "num_trials": 10  # Run 10 times with different schemas
            }

        embeddings:
            {
                "name": "qwen3_embeddings",
                "preset": "hf_qwen3_embedding_8b"
            }
    """
    if task.triplets is None:
        raise ValueError("Task has no triplets.")

    # Check for multi-trial configuration
    num_trials = method_config.get("num_trials", 1)
    if num_trials > 1:
        logger.info(
            f"Running {num_trials} trials for method: {method_config.get('name', method_type)}"
        )
        trials = []

        for trial_idx in range(num_trials):
            # Create trial-specific config
            # Convert OmegaConf to plain dict to avoid struct mode issues
            trial_config = _to_plain_python(method_config)

            # Remove num_trials to avoid recursion
            if "num_trials" in trial_config:
                del trial_config["num_trials"]

            # Add trial suffix to name
            base_name = method_config.get("name", method_type)
            trial_name = f"{base_name}_trial{trial_idx + 1}"
            trial_config["name"] = trial_name

            # CRITICAL: Add trial index to config hash so each trial gets unique cache key
            # This prevents caching from returning identical schemas across trials
            trial_config["_trial_idx"] = trial_idx

            # Run evaluation for this trial
            logger.info(f"  Trial {trial_idx + 1}/{num_trials}")
            try:
                trial_result = evaluate_method(
                    method_type=method_type, task=task, method_config=trial_config
                )
            except Exception as e:
                # If a trial fails, store error result but continue with other trials
                logger.error(
                    f"Error in trial {trial_idx + 1}/{num_trials}: {e}", exc_info=True
                )
                trial_result = {
                    "error": str(e),
                    "accuracy": 0.0,
                    "n_correct": 0,
                    "n_incorrect": 0,
                    "n_ties": 0,
                    "n_total": 0,
                    "method_name": trial_name,
                }
            trials.append(trial_result)

        # Return multi-trial result
        return {
            "_multi_trial": True,
            "trials": trials,
            "num_trials": num_trials,
            "base_method_name": method_config.get("name", method_type),
        }

    # Use user-provided name for display (hash still used in cache alias for safety)
    method_name = method_config.get("name", method_type)
    display_name = method_name

    # Check if method requires GPU when none is available
    preset = method_config.get("preset")
    if preset:
        try:
            if preset_requires_gpu(preset) and not is_gpu_available():
                logger.warning(
                    f"Skipping {display_name}: requires GPU but none available"
                )
                return {
                    "skipped": True,
                    "reason": "Requires GPU but none available",
                    "method_name": display_name,
                }
        except ValueError:
            # Preset not found in registry - let it fail later with better error message
            pass

    # Extract shared data once
    document_texts = _extract_document_texts(task)
    criterion_metadata = (
        task.document_set.get_criterion_metadata(task.criterion_name) or {}
    )

    # Compute cache alias once (used by most methods except bm25)
    cache_alias = make_cache_alias(
        task=task,
        method_config=method_config,
        default_name=_get_default_cache_name(method_type, method_config),
    )

    if method_type == "lm_judge_triplet":
        preset = method_config.get(
            "preset", "lmjudge_triplet_plaintext_binaryhard_gemini"
        )
        annotations = get_annotations_if_required(preset=preset, task=task)

        # Skip if preset requires annotations but they're not available/suitable
        if preset_requires_annotations(preset) and annotations is None:
            logger.warning("Skipping method due to missing/insufficient annotations")
            return {
                "skipped": True,
                "reason": "Missing or insufficient annotations",
                "method_name": display_name,
            }

        raw = evaluate_with_lm_judge_triplet(
            triplets=build_triplet_dicts(task.documents, task.triplets),
            criterion=task.criterion_name,
            criterion_description=task.criterion_description,
            document_type=_resolved_document_type(task),
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            annotations=annotations,
        )
        return finalize_method_results(raw, display_name)

    if method_type == "lm_judge_pair":
        preset = method_config.get("preset", "lmjudge_pair_plaintext_likerthard_gemini")
        annotations = get_annotations_if_required(preset=preset, task=task)

        # Skip if preset requires annotations but they're not available/suitable
        if preset_requires_annotations(preset) and annotations is None:
            logger.warning("Skipping method due to missing/insufficient annotations")
            return {
                "skipped": True,
                "reason": "Missing or insufficient annotations",
                "method_name": display_name,
            }

        raw = evaluate_with_lm_judge_pair(
            triplets=build_triplet_dicts(task.documents, task.triplets),
            criterion=task.criterion_name,
            criterion_description=task.criterion_description,
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            annotations=annotations,
        )
        return finalize_method_results(raw, display_name)

    if method_type == "embeddings":
        preset = method_config.get("preset", "openai_embedding_small")
        preset_overrides = _prepare_instruction_overrides(
            method_config, criterion_metadata
        )

        raw = evaluate_with_embeddings(
            documents=task.documents,
            triplet_ids=task.triplets,
            embedding_preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides,
            criterion=task.criterion_name,
            criterion_description=task.criterion_description,
        )
        return finalize_method_results(raw, display_name)

    if method_type == "reranker":
        preset = method_config.get("preset", "qwen3_reranker_8b")
        preset_overrides = _prepare_instruction_overrides(
            method_config, criterion_metadata
        )
        # Fall back to standard criterion-aware template if no instruction provided
        if not preset_overrides or "instruction" not in preset_overrides:
            criterion_desc = criterion_metadata.get("description", "")
            instruction = (
                f"Given a query, retrieve documents based on the criterion "
                f"'{task.criterion_name}': {criterion_desc}"
            )
            if not preset_overrides:
                preset_overrides = {}
            preset_overrides["instruction"] = instruction

        raw = evaluate_with_reranker(
            documents=task.documents,
            triplet_ids=task.triplets,
            reranker_preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides,
        )
        return finalize_method_results(raw, display_name)

    if method_type == "bm25":
        preset = method_config.get("preset", "bm25_lexical")

        if method_config.get("use_annotations"):
            logger.warning("bm25: use_annotations is not supported; ignoring.")

        raw = evaluate_with_bm25(
            documents=document_texts,
            triplet_ids=task.triplets,
            preset=preset,
        )
        result = finalize_method_results(raw)
        result["method_name"] = display_name
        return result

    if method_type == "document_rewrite":
        summary_preset = method_config.get("summary_preset", "document_summary_gemini")
        embedding_preset = method_config.get("embedding_preset", "bm25_lexical")
        preset_overrides = method_config.get("preset_overrides")

        raw = evaluate_with_document_rewrite(
            documents=task.documents,
            triplet_ids=task.triplets,
            criterion=task.criterion_name,
            criterion_description=task.criterion_description,
            summary_preset=summary_preset,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides,
        )
        result = finalize_method_results(raw)
        result["method_name"] = display_name
        return result

    if method_type == "multisummary":
        summary_preset = method_config.get(
            "summary_preset", "document_to_summaries_gemini"
        )
        embedding_preset = method_config.get(
            "embedding_preset", "openai_embedding_small"
        )
        num_summaries = method_config.get("num_summaries", 5)
        preset_overrides = method_config.get("preset_overrides")

        raw = evaluate_with_multisummary(
            documents=task.documents,
            triplet_ids=task.triplets,
            criterion=task.criterion_name,
            criterion_description=task.criterion_description,
            num_summaries=num_summaries,
            summary_preset=summary_preset,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides,
        )
        result = finalize_method_results(raw)
        result["method_name"] = display_name
        return result

    # Backwards compatibility for query_expansion_bm25
    if method_type == "query_expansion_bm25":
        summary_preset = method_config.get("summary_preset", "document_summary_gemini")

        raw = evaluate_with_document_rewrite(
            documents=document_texts,
            triplet_ids=task.triplets,
            criterion=task.criterion_name,
            criterion_description=task.criterion_description,
            summary_preset=summary_preset,
            embedding_preset="bm25_lexical",
            cache_alias=cache_alias,
            run_name=task.run_name,
        )
        result = finalize_method_results(raw)
        result["method_name"] = display_name
        return result

    if method_type == "in_one_word":
        preset = method_config.get("preset", "inoneword_hf_qwen3_8b")

        # Support config-based category_context (no annotations needed)
        # Can be a string (same for all tasks) or dict (per-task)
        category_context_config = method_config.get("category_context")
        category_context = None

        if category_context_config is not None:
            if isinstance(category_context_config, dict):
                # Dict format: {task_name: context} or {criterion: context}
                task_key = f"{task.document_set.DATASET_NAME}_{task.criterion_name}"
                category_context = category_context_config.get(
                    task_key, category_context_config.get(task.criterion_name)
                )
                if category_context is None:
                    logger.warning(
                        f"No category_context found for task '{task_key}' "
                        f"or criterion '{task.criterion_name}' in dict config. "
                        f"Available keys: {list(category_context_config.keys())}"
                    )
            else:
                # String format: use for all tasks
                category_context = category_context_config

        annotations = None

        # Check if we should generate a fresh schema (for "proposed schema" experiments)
        generate_schema = method_config.get("generate_schema", False)

        if generate_schema:
            # Generate a fresh schema dynamically during evaluation
            logger.info("Generating fresh schema for in_one_word evaluation")
            from multiview.benchmark.annotations import generate_category_schema

            schema = generate_category_schema(
                documents=document_texts,
                criterion=task.criterion_name,
                criterion_description=task.criterion_description,
                document_type=_resolved_document_type(task),
                n_samples=method_config.get("n_schema_samples", 10),
                cache_alias=f"{cache_alias}_generated_schema",
                run_name=task.run_name,
            )

            # Format category context from generated schema
            categories = schema.get("categories", [])
            category_names = [cat["name"] for cat in categories]
            category_str = ", ".join(category_names)
            category_context = (
                f"Categories: {category_str}\n"
                f"Question: Categorize this text in one word."
            )
            logger.info(f"Generated schema with categories: {category_names}")

        # Fallback to annotations if config doesn't provide category_context
        if not category_context and not generate_schema:
            if not task.document_annotations:
                logger.error(
                    "in_one_word requires either 'category_context' in config, "
                    "'generate_schema: true', or annotations with schemas"
                )
                return {
                    "skipped": True,
                    "reason": "Missing category_context and annotations",
                    "method_name": display_name,
                }

            # Check for category schema (lm_all) or tag schema (lm_tags)
            first_ann = task.document_annotations[0]
            has_schema = (
                "category_schema" in first_ann and first_ann.get("category_schema")
            ) or ("tag_schema" in first_ann and first_ann.get("tag_schema"))

            if not has_schema:
                logger.error(
                    "in_one_word requires either 'category_context' in config, "
                    "'generate_schema: true', or annotations with category_schema/tag_schema. "
                    "Make sure triplet_style is 'lm_all' or 'lm_tags'."
                )
                return {
                    "skipped": True,
                    "reason": "Missing schemas in annotations",
                    "method_name": display_name,
                }

            annotations = task.document_annotations

        raw = evaluate_with_in_one_word(
            documents=document_texts,
            triplet_ids=task.triplets,
            category_context=category_context,
            annotations=annotations,
            preset=preset,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=method_config.get("preset_overrides"),
        )
        result = finalize_method_results(raw)
        result["method_name"] = display_name
        return result

    if method_type == "pseudologit":
        preset = method_config.get("preset", "pseudologit_gemini_n100")
        classes_file_config = method_config.get("classes_file")
        classes_file = None

        # Handle classes_file: can be string (same for all) or dict (per-task)
        if classes_file_config is not None:
            # Convert OmegaConf to plain Python types
            classes_file_config = _to_plain_python(classes_file_config)

            if isinstance(classes_file_config, dict):
                # Dict format: {task_name: file} or {criterion: file}
                task_key = f"{task.document_set.DATASET_NAME}_{task.criterion_name}"
                classes_file = classes_file_config.get(
                    task_key, classes_file_config.get(task.criterion_name)
                )
                if classes_file is None:
                    logger.warning(
                        f"No classes_file found for task '{task_key}' "
                        f"or criterion '{task.criterion_name}' in dict config. "
                        f"Available keys: {list(classes_file_config.keys())}"
                    )
            else:
                # String format: use for all tasks
                classes_file = classes_file_config

        # Check if we should use oracle schema from annotations
        use_oracle_schema = method_config.get("use_oracle_schema", False)

        # Check if we should generate a fresh schema (for "proposed schema" experiments)
        generate_schema = method_config.get("generate_schema", False)

        if use_oracle_schema:
            # Extract classes from oracle schema in annotations
            logger.info("Extracting classes from oracle schema in annotations")

            if not task.document_annotations:
                logger.error("use_oracle_schema=true but no annotations available")
                return {
                    "skipped": True,
                    "reason": "Missing annotations",
                    "method_name": display_name,
                }

            # Try category schema first (from lm_all)
            schema = task.document_annotations[0].get("category_schema")
            if schema:
                categories = schema.get("categories", [])
                classes_data = _schema_to_classes_data(categories)

                logger.info(
                    f"Using oracle category schema with {len(categories)} classes: "
                    f"{classes_data['classes']} -> {[c['name'] for c in categories]}"
                )
            else:
                # Fall back to tag schema (from lm_tags)
                tag_schema = task.document_annotations[0].get("tag_schema")
                if tag_schema:
                    tags = tag_schema.get("tags", [])
                    classes_data = _schema_to_classes_data(tags)

                    logger.info(
                        f"Using oracle tag schema with {len(tags)} tags: "
                        f"{classes_data['classes']} -> {[t['name'] for t in tags]}"
                    )
                else:
                    logger.error(
                        "use_oracle_schema=true but no category_schema or tag_schema in annotations. "
                        "Make sure triplet_style is 'lm_all' or 'lm_tags'."
                    )
                    return {
                        "skipped": True,
                        "reason": "Missing category_schema or tag_schema in annotations",
                        "method_name": display_name,
                    }

            # Write classes_data to a temporary file
            import tempfile

            temp_classes_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(classes_data, temp_classes_file, indent=2)
            temp_classes_file.close()
            classes_file = temp_classes_file.name

        elif generate_schema:
            # Generate a fresh schema dynamically during evaluation
            logger.info("Generating fresh schema for pseudologit evaluation")
            from multiview.benchmark.annotations import generate_category_schema

            schema = generate_category_schema(
                documents=document_texts,
                criterion=task.criterion_name,
                criterion_description=task.criterion_description,
                document_type=_resolved_document_type(task),
                n_samples=method_config.get("n_schema_samples", 10),
                cache_alias=f"{cache_alias}_generated_schema",
                run_name=task.run_name,
            )

            # Extract categories and create classes file
            categories = schema.get("categories", [])
            classes_data = _schema_to_classes_data(categories)

            import tempfile

            temp_classes_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(classes_data, temp_classes_file, indent=2)
            temp_classes_file.close()
            classes_file = temp_classes_file.name

            logger.info(
                f"Generated schema with classes: {classes_data['classes']} -> "
                f"{[c['name'] for c in categories]}"
            )

        elif not classes_file:
            raise ValueError(
                "Pseudologit requires a classes_file. Specify 'classes_file' in the method config, "
                "or use 'use_oracle_schema: true' or 'generate_schema: true'."
            )

        raw = evaluate_with_pseudologit(
            documents=document_texts,
            triplet_ids=task.triplets,
            preset=preset,
            classes_file=classes_file,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=method_config.get("preset_overrides"),
        )
        result = finalize_method_results(raw)
        result["method_name"] = display_name
        return result

    if method_type == "query_relevance_vectors":
        expansion_preset = method_config.get(
            "expansion_preset", "query_relevance_scores_gemini"
        )
        embedding_preset = method_config.get(
            "embedding_preset", "openai_embedding_small"
        )
        num_expansions = method_config.get("num_expansions", 10)
        dev_set_size = method_config.get("dev_set_size", 25)
        preset_overrides = method_config.get("preset_overrides")

        raw = evaluate_with_query_relevance_vectors(
            documents=document_texts,
            triplet_ids=task.triplets,
            criterion=task.criterion_name,
            criterion_description=task.criterion_description,
            expansion_preset=expansion_preset,
            embedding_preset=embedding_preset,
            num_expansions=num_expansions,
            dev_set_size=dev_set_size,
            cache_alias=cache_alias,
            run_name=task.run_name,
            preset_overrides=preset_overrides,
        )
        result = finalize_method_results(raw)
        result["method_name"] = display_name
        return result

    raise ValueError(f"Unknown method type: {method_type}")


#
# Metrics + normalization helpers (merged here to keep `Benchmark` lean).
#


def metrics_from_correctness(
    correct: Sequence[bool],
    *,
    is_tie: Sequence[bool] | None = None,
    exclude_ties: bool = True,
) -> dict[str, Any]:
    """Aggregate correctness booleans into the canonical metrics dict."""
    n_total = len(correct)

    if is_tie is not None and len(is_tie) != n_total:
        raise ValueError(
            "correct and is_tie must have the same length: "
            f"{n_total} != {len(is_tie)}"
        )

    if is_tie is None:
        n_correct = sum(1 for c in correct if c)
        n_incorrect = n_total - n_correct
        n_ties = 0
        accuracy = (n_correct / n_total) if n_total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_ties": n_ties,
            "n_total": n_total,
        }

    n_ties = sum(1 for t in is_tie if t)

    if exclude_ties:
        n_correct = sum(1 for c, t in zip(correct, is_tie, strict=False) if c and not t)
        n_incorrect = sum(
            1 for c, t in zip(correct, is_tie, strict=False) if (not c) and (not t)
        )
        n_judged = n_correct + n_incorrect
        accuracy = (n_correct / n_judged) if n_judged > 0 else 0.0
    else:
        # exclude_ties=False: penalize ties as incorrect
        n_correct = sum(1 for c in correct if c)
        n_incorrect = n_total - n_correct  # includes ties
        accuracy = (n_correct / n_total) if n_total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_ties": n_ties,
        "n_total": n_total,
    }


def outcomes_from_pair_scores(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    *,
    tie_tol: float = 0.0,
) -> list[int]:
    """Convert pairwise scores into standardized triplet outcomes."""
    if len(positive_scores) != len(negative_scores):
        raise ValueError(
            "positive_scores and negative_scores must have the same length: "
            f"{len(positive_scores)} != {len(negative_scores)}"
        )
    if tie_tol < 0:
        raise ValueError(f"tie_tol must be >= 0, got {tie_tol}")

    outcomes: list[int] = []
    for pos, neg in zip(positive_scores, negative_scores, strict=False):
        if pos > neg + tie_tol:
            outcomes.append(1)
        elif neg > pos + tie_tol:
            outcomes.append(-1)
        else:
            outcomes.append(0)
    return outcomes


def metrics_from_outcomes(outcomes: Sequence[int]) -> dict[str, Any]:
    """Aggregate triplet outcomes into the canonical metrics dict."""
    n_total = len(outcomes)
    correct = [o == 1 for o in outcomes]
    is_tie = [o == 0 for o in outcomes]
    metrics = metrics_from_correctness(correct, is_tie=is_tie, exclude_ties=False)
    metrics["n_total"] = n_total
    return metrics


def finalize_method_results(
    method_results: dict[str, Any], method_name: str | None = None
) -> dict[str, Any]:
    """Normalize method results to include standard metrics + correctness vectors.

    Args:
        method_results: Raw results from evaluation method
        method_name: Optional method name to include in results

    Returns:
        Finalized results dict with standard metrics
    """
    result = method_results.copy()

    if "accuracy" not in result or "n_total" not in result:
        if "correct" in result:
            correct = result["correct"]
            is_tie = result.get("is_tie")
            metrics = metrics_from_correctness(
                correct, is_tie=is_tie, exclude_ties=False
            )
            result = {**metrics, **result}

        elif "outcomes" in result:
            outcomes = result["outcomes"]
            correct = [o == 1 for o in outcomes]
            is_tie = [o == 0 for o in outcomes]
            metrics = metrics_from_correctness(
                correct, is_tie=is_tie, exclude_ties=False
            )
            result = {**metrics, "correct": correct, "is_tie": is_tie, **result}

        elif "positive_scores" in result and "negative_scores" in result:
            outcomes = outcomes_from_pair_scores(
                result["positive_scores"],
                result["negative_scores"],
            )
            correct = [o == 1 for o in outcomes]
            is_tie = [o == 0 for o in outcomes]
            metrics = metrics_from_correctness(
                correct, is_tie=is_tie, exclude_ties=False
            )
            result = {**metrics, "correct": correct, "is_tie": is_tie, **result}

    if method_name is not None:
        result["method_name"] = method_name

    return result


def _method_uses_instruction(
    method_type: str, method_config: dict, criterion_metadata: dict
) -> bool:
    """Check if a method configuration uses instructions.

    Args:
        method_type: Type of method (embeddings, reranker, etc.)
        method_config: Method configuration dict
        criterion_metadata: Criterion metadata dict

    Returns:
        True if method uses instruction, False otherwise
    """
    # Only these method types support instructions
    if method_type not in ["embeddings", "reranker"]:
        return False

    # Check if instruction is specified in config
    if "embed_instr" in method_config:
        return True

    # Check if preset_overrides contains instruction
    preset_overrides = method_config.get("preset_overrides", {})
    if isinstance(preset_overrides, dict) and "instruction" in preset_overrides:
        return True

    # Check if preset name indicates instruction usage (e.g., "instr_*" prefix)
    # This allows distinguishing between instruction-based and non-instruction presets
    preset = method_config.get("preset", "")
    if "instr" in preset.lower():
        return True

    # Rerankers always use instructions (they have a fallback instruction template)
    if method_type == "reranker":
        return True

    # Don't automatically treat criterion's embed_instr as making ALL methods use instructions
    # This allows proper baseline detection for instruction sensitivity calculation
    return False


def _get_method_baseline_key(method_type: str, preset: str) -> str:
    """Generate a baseline key for matching methods to their baselines.

    Normalizes preset names by removing instruction-related prefixes/markers
    so that instruction and non-instruction variants of the same preset
    can be matched together.

    Args:
        method_type: Type of method (embeddings, reranker, etc.)
        preset: Preset name

    Returns:
        Baseline key string for matching

    Example:
        - "instr_hf_qwen3_embedding_8b" -> "embeddings:hf_qwen3_embedding_8b"
        - "hf_qwen3_embedding_8b" -> "embeddings:hf_qwen3_embedding_8b"
        Both map to the same baseline key
    """
    # Normalize preset name by removing instruction prefix
    normalized_preset = preset
    if normalized_preset.startswith("instr_"):
        normalized_preset = normalized_preset[6:]  # Remove "instr_" prefix

    return f"{method_type}:{normalized_preset}"


def compute_trial_statistics(
    results: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """Compute mean and std across trial results.

    Args:
        results: Nested dict of {task_name: {method_name: metrics_dict}}

    Returns:
        Dict of {task_name: {base_method_name: {mean_accuracy, std_accuracy, ...}}}
        Only includes methods with multiple trials (e.g., "method_trial1", "method_trial2")
    """
    import numpy as np

    stats = {}

    for task_name, task_results in results.items():
        stats[task_name] = {}

        # Group methods by base name (before "_trial")
        trial_groups = {}
        for method_name, metrics in task_results.items():
            if "_trial" in method_name:
                base_name = method_name.rsplit("_trial", 1)[0]
                if base_name not in trial_groups:
                    trial_groups[base_name] = []
                trial_groups[base_name].append(metrics)

        # Compute statistics for each trial group
        for base_name, trial_results_list in trial_groups.items():
            if len(trial_results_list) > 1:
                accuracies = [r.get("accuracy", 0.0) for r in trial_results_list]
                stats[task_name][f"{base_name}_avg"] = {
                    "mean_accuracy": float(np.mean(accuracies)),
                    "std_accuracy": float(np.std(accuracies)),
                    "min_accuracy": float(np.min(accuracies)),
                    "max_accuracy": float(np.max(accuracies)),
                    "num_trials": len(accuracies),
                }

    return stats


def compute_instruction_sensitivity(
    results: dict[str, dict[str, dict]],
    method_configs: dict[str, list[dict]],
    tasks: list,
) -> dict[str, dict[str, float | None]]:
    """Compute instruction sensitivity for all methods.

    For each method that uses instructions, finds the corresponding baseline
    (same method type and preset, but without instructions) and computes the
    accuracy delta.

    Args:
        results: Nested dict of {task_name: {method_name: metrics_dict}}
        method_configs: Dict mapping method types to lists of method configs
        tasks: List of Task objects

    Returns:
        Dict of {task_name: {method_name: sensitivity_score}}
        where sensitivity_score is:
        - accuracy_delta if baseline found
        - None if no baseline or not applicable
    """
    sensitivity = {}

    for task in tasks:
        task_name = task.get_task_name()
        sensitivity[task_name] = {}

        if task_name not in results:
            continue

        criterion_metadata = (
            task.document_set.get_criterion_metadata(task.criterion_name) or {}
        )

        # Build mapping of baseline keys to method names and their accuracies
        baselines = {}  # baseline_key -> (method_name, accuracy)

        # First pass: identify baselines (methods without instructions)
        for method_type, method_list in method_configs.items():
            for method_config in method_list:
                method_name = method_config.get("name", method_type)

                # Skip if method was skipped or errored
                if method_name not in results[task_name]:
                    continue

                method_results = results[task_name][method_name]
                if method_results.get("skipped") or "error" in method_results:
                    continue

                # Check if this is a baseline (no instruction)
                uses_instruction = _method_uses_instruction(
                    method_type, method_config, criterion_metadata
                )

                if not uses_instruction:
                    preset = method_config.get("preset", "")
                    baseline_key = _get_method_baseline_key(method_type, preset)
                    accuracy = method_results.get("accuracy", 0.0)
                    baselines[baseline_key] = (method_name, accuracy)

        # Second pass: compute sensitivity for methods with instructions
        for method_type, method_list in method_configs.items():
            for method_config in method_list:
                method_name = method_config.get("name", method_type)

                # Skip if method was skipped or errored
                if method_name not in results[task_name]:
                    sensitivity[task_name][method_name] = None
                    continue

                method_results = results[task_name][method_name]
                if method_results.get("skipped") or "error" in method_results:
                    sensitivity[task_name][method_name] = None
                    continue

                # Check if this method uses instructions
                uses_instruction = _method_uses_instruction(
                    method_type, method_config, criterion_metadata
                )

                if uses_instruction:
                    # Look for corresponding baseline
                    preset = method_config.get("preset", "")
                    baseline_key = _get_method_baseline_key(method_type, preset)

                    if baseline_key in baselines:
                        baseline_name, baseline_accuracy = baselines[baseline_key]
                        current_accuracy = method_results.get("accuracy", 0.0)
                        delta = current_accuracy - baseline_accuracy
                        sensitivity[task_name][method_name] = delta

                        logger.debug(
                            f"Instruction sensitivity for {task_name}/{method_name}: "
                            f"{delta:.4f} (vs baseline {baseline_name})"
                        )
                    else:
                        # No baseline found
                        sensitivity[task_name][method_name] = None
                        logger.debug(f"No baseline found for {task_name}/{method_name}")
                else:
                    # Method doesn't use instructions, set to None
                    sensitivity[task_name][method_name] = None

    return sensitivity


__all__ = [
    "build_triplet_dicts",
    "compute_instruction_sensitivity",
    "compute_trial_statistics",
    "evaluate_method",
    "finalize_method_results",
    "get_annotations_if_required",
    "make_cache_alias",
    "metrics_from_correctness",
    "metrics_from_outcomes",
    "outcomes_from_pair_scores",
    "preset_requires_annotations",
]
