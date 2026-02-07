"""Shared benchmark-loading and embedding helpers for visualization workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from multiview.benchmark.evaluation_utils import get_config_hash_suffix
from multiview.docsets import DOCSETS
from multiview.eval.generation_utils import validate_criterion_description
from multiview.inference import run_inference
from multiview.utils.prompt_utils import read_or_return

logger = logging.getLogger(__name__)


def _infer_dataset_and_criterion(task_name: str) -> tuple[str, str]:
    """Infer dataset name and criterion from a task artifact name."""
    task_parts = task_name.split("__")
    dataset_name = task_parts[0]

    if len(task_parts) < 2:
        return dataset_name, "default"

    style_indicators = {
        "random",
        "prelabeled",
        "hard_negative",
        "lm",
        "lm_all",
        "lm_category",
        "lm_tags",
        "lm_summary_dict",
        "lm_summary_sentence",
        "rnd",
        "pre",
        "hn",
        "cat",
        "tag",
        "sdict",
        "ssent",
    }

    if (
        len(task_parts) >= 4
        and task_parts[-2] in style_indicators
        and task_parts[-1].isdigit()
    ):
        criterion_parts = task_parts[1:-2]
    else:
        criterion_parts = task_parts[1:]

    criterion = "__".join(criterion_parts) if criterion_parts else "default"
    return dataset_name, criterion


def _load_method_metadata_from_logs(
    output_base: Path, task_name: str, method_name: str
) -> dict[str, Any] | None:
    """Load method metadata (method_type, presets) from first method-log record."""
    method_log_file = output_base / "method_logs" / task_name / f"{method_name}.jsonl"
    if not method_log_file.exists():
        return None

    with open(method_log_file) as f:
        first_line = f.readline().strip()
    if not first_line:
        return None

    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse method log metadata: {method_log_file}")
        return None

    return {
        "method_type": record.get("method_type"),
        "cache_alias": record.get("cache_alias"),
        "criterion_description": record.get("criterion_description"),
        "embedding_preset": record.get("embedding_preset"),
        "summary_preset": record.get("summary_preset"),
        "num_summaries": record.get("num_summaries"),
        "expansion_preset": record.get("expansion_preset"),
        "num_expansions": record.get("num_expansions"),
        "dev_set_size": record.get("dev_set_size"),
        "dev_set_seed": record.get("dev_set_seed"),
        "preset": record.get("preset"),
    }


def _corpus_document_texts_path(
    output_base: Path, task_name: str, method_name: str
) -> Path:
    """Path for persisted full-corpus display text used by visualization."""
    return output_base / "corpus" / "documents" / task_name / f"{method_name}.txt"


def _method_eval_cache_alias_with_metadata(
    task_name: str,
    method_name: str,
    method_metadata: dict[str, Any] | None,
) -> str:
    """Build eval cache alias using the same hash logic as run_eval."""
    if method_metadata and isinstance(method_metadata.get("cache_alias"), str):
        cache_alias = method_metadata.get("cache_alias", "").strip()
        if cache_alias:
            return cache_alias

    method_config: dict[str, Any] = {"name": method_name}
    for key in [
        "preset",
        "preset_overrides",
        "embedding_preset",
        "summary_preset",
        "num_summaries",
        "expansion_preset",
        "num_expansions",
        "dev_set_size",
        "dev_set_seed",
    ]:
        if (
            method_metadata
            and key in method_metadata
            and method_metadata[key] is not None
        ):
            method_config[key] = method_metadata[key]

    hash_suffix = get_config_hash_suffix(method_config)
    return f"{task_name}_eval_{method_name}{hash_suffix}"


def _save_document_texts(texts: list[str], output_path: Path) -> None:
    """Persist one display text per line (newline-escaped)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(str(text).replace("\n", "\\n") + "\n")


def _load_document_texts(output_path: Path) -> list[str] | None:
    """Load persisted document display text if present."""
    if not output_path.exists():
        return None

    with open(output_path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    return [line.replace("\\n", "\n") for line in lines]


def _load_criterion_description_from_triplet_config(
    output_base: Path, task_name: str
) -> str | None:
    """Load criterion_description from triplet config artifact if present."""
    config_path = output_base / "triplets" / task_name / "triplet_config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Failed to read triplet config: {config_path}")
        return None

    description = config.get("criterion_description")
    return description if isinstance(description, str) else None


def _load_criterion_description_from_docset_metadata(
    *, dataset_name: str, criterion: str
) -> str | None:
    """Load criterion description from docset metadata without reading run artifacts."""
    docset_cls = DOCSETS.get(dataset_name)
    if docset_cls is None:
        return None

    try:
        docset = docset_cls(config={})
        get_meta = getattr(docset, "get_criterion_metadata", None)
        if not callable(get_meta):
            return None
        metadata = get_meta(criterion) or {}
        description = metadata.get("description")
        if isinstance(description, str) and description.strip():
            return description
        return None
    except Exception as e:
        logger.debug(
            f"Could not load criterion metadata for {dataset_name}/{criterion}: {e}"
        )
        return None


def _resolve_required_criterion_description(
    *,
    dataset_name: str,
    criterion: str,
    method_metadata: dict[str, Any] | None,
    output_base: Path,
    task_name: str,
    context: str,
) -> str:
    """Resolve a non-empty criterion description or fail with a clear error."""
    docset_description = _load_criterion_description_from_docset_metadata(
        dataset_name=dataset_name,
        criterion=criterion,
    )
    candidates = []
    if method_metadata:
        candidates.append(method_metadata.get("criterion_description"))
    candidates.append(
        _load_criterion_description_from_triplet_config(output_base, task_name)
    )
    candidates.append(docset_description)

    raw_description = next(
        (candidate for candidate in candidates if isinstance(candidate, str)), None
    )
    return validate_criterion_description(
        criterion=criterion,
        criterion_description=raw_description,
        context=context,
    )


def generate_embeddings(
    doc_texts: list[str],
    embedding_preset: str,
    cache_alias: str | None = None,
    force_refresh: bool = False,
    criterion: str | None = None,
    criterion_description: str | None = None,
    in_one_word_context: str | None = None,
    pseudologit_classes: str | None = None,
    images: list[str | None] | None = None,
) -> np.ndarray:
    """Generate embeddings for documents."""
    logger.info(f"Generating embeddings with {embedding_preset}")

    inputs = {"document": doc_texts}
    if images and any(img is not None for img in images):
        inputs["images"] = images

    if criterion:
        criterion_description = validate_criterion_description(
            criterion=criterion,
            criterion_description=criterion_description,
            context=f"embedding preset '{embedding_preset}'",
        )
        inputs["criterion"] = criterion
        inputs["criterion_description"] = criterion_description

    if in_one_word_context:
        context_text = read_or_return(in_one_word_context)
        context_preview = context_text[:100] + (
            "..." if len(context_text) > 100 else ""
        )
        logger.info(f"Using context: {context_preview}")
        inputs["context"] = context_text

    config_overrides = {}

    if pseudologit_classes:
        from multiview.inference.presets import get_preset

        logger.info(f"Using pseudologit classes from: {pseudologit_classes}")
        preset_config = get_preset(embedding_preset)
        merged_extra_kwargs = preset_config.extra_kwargs.copy()
        merged_extra_kwargs["classes_file"] = pseudologit_classes
        config_overrides["extra_kwargs"] = merged_extra_kwargs

    results = run_inference(
        inputs=inputs,
        config=embedding_preset,
        cache_alias=cache_alias,
        force_refresh=force_refresh,
        **config_overrides,
    )

    embeddings = []
    for result in results:
        if isinstance(result, dict) and "vector" in result:
            embeddings.append(result["vector"])
        else:
            embeddings.append(result)

    embeddings = np.array(embeddings)
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings


def _regenerate_full_document_rewrite_embeddings(
    run_name: str,
    task_name: str,
    method_name: str,
    documents: list[Any],
    criterion: str,
    criterion_description: str,
    method_metadata: dict[str, Any],
    output_base: Path,
    cache_alias: str,
) -> np.ndarray | None:
    """Regenerate full-corpus embeddings for document_rewrite methods."""
    summary_preset = method_metadata.get("summary_preset")
    embedding_preset = method_metadata.get("embedding_preset")
    if not summary_preset or not embedding_preset:
        logger.warning(
            f"Cannot regenerate full embeddings for {task_name}/{method_name}: "
            "missing summary_preset or embedding_preset metadata"
        )
        return None

    from multiview.eval.document_summary import (
        _compute_embeddings_for_documents,
        _generate_summaries,
    )

    logger.info(
        "Regenerating full-corpus document-rewrite embeddings "
        f"for {task_name}/{method_name} ({len(documents)} docs)"
    )

    summaries = _generate_summaries(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        summary_preset=summary_preset,
        cache_alias=cache_alias,
        run_name=run_name,
    )

    display_texts_path = _corpus_document_texts_path(
        output_base, task_name, method_name
    )
    _save_document_texts(summaries, display_texts_path)

    doc_ids = list(range(len(documents)))
    doc_id_to_embedding = _compute_embeddings_for_documents(
        summaries=summaries,
        doc_ids=doc_ids,
        embedding_preset=embedding_preset,
        cache_alias=cache_alias,
        run_name=run_name,
        preset_overrides=None,
        criterion=criterion,
    )

    full_embeddings = np.array(
        [doc_id_to_embedding[i] for i in range(len(documents))], dtype=np.float32
    )

    if np.isnan(full_embeddings).any() or np.isinf(full_embeddings).any():
        logger.warning(
            "Regenerated full embeddings still contain NaN/Inf; falling back to filtered rows"
        )
        return None

    return full_embeddings


def _regenerate_full_method_embeddings(
    run_name: str,
    output_base: Path,
    task_name: str,
    method_name: str,
    documents: list[Any],
    doc_texts: list[str],
    criterion: str,
    criterion_description: str | None,
    method_metadata: dict[str, Any] | None,
    cache_alias: str,
) -> np.ndarray | None:
    """Regenerate full-corpus embeddings for supported method types."""
    if not method_metadata:
        return None

    method_type = method_metadata.get("method_type")

    if method_type == "embeddings":
        embedding_preset = method_metadata.get(
            "embedding_preset"
        ) or method_metadata.get("preset")
        if not embedding_preset:
            return None
        # Extract images from documents for multimodal embedding providers
        images = None
        if documents:
            images = [
                doc.get("image_path") if isinstance(doc, dict) else None
                for doc in documents
            ]
        return generate_embeddings(
            doc_texts=doc_texts,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            criterion=criterion,
            criterion_description=criterion_description,
            images=images,
        )

    if method_type == "document_rewrite":
        return _regenerate_full_document_rewrite_embeddings(
            run_name=run_name,
            task_name=task_name,
            method_name=method_name,
            documents=documents,
            criterion=criterion,
            criterion_description=validate_criterion_description(
                criterion=criterion,
                criterion_description=criterion_description,
                context=f"benchmark regeneration for {task_name}/{method_name}",
            ),
            method_metadata=method_metadata,
            output_base=output_base,
            cache_alias=cache_alias,
        )

    if method_type == "multisummary":
        from multiview.eval.multisummary import (
            _embed_summaries,
            _generate_multisummaries,
        )

        summary_preset = method_metadata.get("summary_preset")
        embedding_preset = method_metadata.get("embedding_preset")
        num_summaries = method_metadata.get("num_summaries")
        if (
            not summary_preset
            or not embedding_preset
            or not isinstance(num_summaries, int)
        ):
            return None

        summaries_flat = _generate_multisummaries(
            documents=documents,
            criterion=criterion,
            criterion_description=validate_criterion_description(
                criterion=criterion,
                criterion_description=criterion_description,
                context=f"benchmark regeneration for {task_name}/{method_name}",
            ),
            num_summaries=num_summaries,
            summary_preset=summary_preset,
            cache_alias=cache_alias,
            run_name=run_name,
        )
        all_embeddings_flat = _embed_summaries(
            summaries=summaries_flat,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            run_name=run_name,
            preset_overrides=None,
            criterion=criterion,
        )

        emb_arr = np.array(all_embeddings_flat, dtype=np.float32).reshape(
            len(documents), num_summaries, -1
        )
        return np.mean(emb_arr, axis=1, dtype=np.float32)

    if method_type == "query_relevance_vectors":
        from multiview.eval.generation_utils import (
            generate_text_variations_from_documents,
        )
        from multiview.eval.query_relevance_vectors import (
            _create_score_vectors,
            _sample_dev_set,
        )

        expansion_preset = method_metadata.get("expansion_preset")
        embedding_preset = method_metadata.get("embedding_preset")
        num_expansions = method_metadata.get("num_expansions")
        dev_set_size = method_metadata.get("dev_set_size")
        dev_set_seed = method_metadata.get("dev_set_seed")
        if not expansion_preset or not embedding_preset:
            return None
        if not isinstance(num_expansions, int) or not isinstance(dev_set_size, int):
            return None

        dev_set_docs, _ = _sample_dev_set(
            documents=doc_texts,
            dev_set_size=dev_set_size,
            criterion=criterion,
            random_seed=dev_set_seed,
        )
        expanded_queries = generate_text_variations_from_documents(
            documents=dev_set_docs,
            criterion=criterion,
            criterion_description=validate_criterion_description(
                criterion=criterion,
                criterion_description=criterion_description,
                context=f"benchmark regeneration for {task_name}/{method_name}",
            ),
            num_variations=num_expansions,
            generation_preset=expansion_preset,
            cache_alias=cache_alias,
            run_name=run_name,
            cache_suffix="expansion",
        )
        _score_vectors, document_embeddings = _create_score_vectors(
            documents=doc_texts,
            expanded_queries=expanded_queries,
            embedding_preset=embedding_preset,
            cache_alias=cache_alias,
            run_name=run_name,
            preset_overrides=None,
            criterion=criterion,
        )
        return np.array(document_embeddings, dtype=np.float32)

    return None


def _method_supports_full_regeneration(method_metadata: dict[str, Any] | None) -> bool:
    if not method_metadata:
        return False
    return method_metadata.get("method_type") in {
        "embeddings",
        "document_rewrite",
        "multisummary",
        "query_relevance_vectors",
    }


def load_from_benchmark(
    run_name: str,
    task_name: str,
    method_name: str,
    preferred_view: str | None = None,
) -> tuple[list[Any], list[str], str, np.ndarray | None, str, str | None]:
    """Load documents and embeddings from benchmark run."""
    from multiview.benchmark.artifacts import (
        load_documents_from_jsonl as load_docs_artifact,
    )

    output_base = Path("outputs") / run_name

    documents_dir = output_base / "documents"
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents not found: {documents_dir}\n"
            f"Run 'python scripts/create_eval.py --config-name {run_name}' first"
        )

    documents = load_docs_artifact(output_dir=str(documents_dir), task_name=task_name)

    def _to_text(doc: Any) -> str:
        if isinstance(doc, str):
            return doc
        if isinstance(doc, dict):
            text = doc.get("text", "")
            if text:
                return str(text)
            if doc.get("image_path"):
                return "<image>"
            return json.dumps(doc, ensure_ascii=False)
        return str(doc)

    doc_texts = [_to_text(doc) for doc in documents]

    dataset_name, criterion = _infer_dataset_and_criterion(task_name)
    method_metadata = _load_method_metadata_from_logs(
        output_base=output_base, task_name=task_name, method_name=method_name
    )
    criterion_description = _load_criterion_description_from_triplet_config(
        output_base=output_base,
        task_name=task_name,
    )
    if not criterion_description and method_metadata:
        method_desc = method_metadata.get("criterion_description")
        if isinstance(method_desc, str):
            criterion_description = method_desc
    is_document_rewrite_method = (
        method_metadata is not None
        and method_metadata.get("method_type") == "document_rewrite"
    )
    eval_cache_alias = _method_eval_cache_alias_with_metadata(
        task_name=task_name,
        method_name=method_name,
        method_metadata=method_metadata,
    )
    doc_texts_artifact = _corpus_document_texts_path(
        output_base, task_name, method_name
    )
    stored_doc_texts = _load_document_texts(doc_texts_artifact)
    if stored_doc_texts:
        if len(stored_doc_texts) == len(documents):
            doc_texts = stored_doc_texts
        else:
            logger.warning(
                "Found stale document-text artifact with mismatched length: "
                f"{doc_texts_artifact} ({len(stored_doc_texts)} != {len(documents)})"
            )

    triples_dir = output_base / "triples" / "embeddings" / task_name
    triples_file = triples_dir / f"{method_name}.npy"

    viz_embeddings_dir = output_base / "corpus" / "embeddings" / task_name
    viz_full_embeddings_file = viz_embeddings_dir / f"{method_name}.npy"

    if preferred_view == "triples":
        embedding_candidates = [triples_file, viz_full_embeddings_file]
    elif preferred_view == "corpus":
        embedding_candidates = [viz_full_embeddings_file, triples_file]
    else:
        # Backward-compatible default: prefer corpus-level artifacts if present.
        embedding_candidates = [viz_full_embeddings_file, triples_file]

    method_type = method_metadata.get("method_type") if method_metadata else None
    should_regenerate = _method_supports_full_regeneration(method_metadata)
    if should_regenerate and method_type == "document_rewrite":
        if preferred_view == "triples" and triples_file.exists():
            # In triples mode, keep the eval-scope embeddings when available.
            should_regenerate = False
        else:
            has_cached_embeddings = (
                viz_full_embeddings_file.exists() or triples_file.exists()
            )
            has_cached_rewrite_texts = stored_doc_texts is not None and len(
                stored_doc_texts
            ) == len(documents)
            # Document-rewrite regeneration is expensive and should only run when
            # visualization artifacts are incomplete or missing.
            should_regenerate = not (has_cached_embeddings and has_cached_rewrite_texts)

    embeddings = None
    if should_regenerate:
        requires_criterion_description = (
            method_metadata is not None
            and method_metadata.get("method_type")
            in {
                "embeddings",
                "document_rewrite",
                "multisummary",
                "query_relevance_vectors",
            }
        )
        resolved_criterion_description = criterion_description
        if requires_criterion_description:
            resolved_criterion_description = _resolve_required_criterion_description(
                dataset_name=dataset_name,
                criterion=criterion,
                method_metadata=method_metadata,
                output_base=output_base,
                task_name=task_name,
                context=f"benchmark regeneration for {task_name}/{method_name}",
            )

        regenerated = _regenerate_full_method_embeddings(
            run_name=run_name,
            output_base=output_base,
            task_name=task_name,
            method_name=method_name,
            documents=documents,
            doc_texts=doc_texts,
            criterion=criterion,
            criterion_description=resolved_criterion_description,
            method_metadata=method_metadata,
            cache_alias=eval_cache_alias,
        )
        if regenerated is not None:
            embeddings = regenerated
            viz_embeddings_dir.mkdir(parents=True, exist_ok=True)
            np.save(viz_full_embeddings_file, embeddings)
            stored_doc_texts = _load_document_texts(doc_texts_artifact)
            if stored_doc_texts and len(stored_doc_texts) == len(documents):
                doc_texts = stored_doc_texts
            elif is_document_rewrite_method:
                raise RuntimeError(
                    "Document-rewrite visualization requires persisted rewrite text, "
                    f"but regeneration did not produce a valid artifact: {doc_texts_artifact}"
                )
        elif is_document_rewrite_method:
            raise RuntimeError(
                "Missing document-rewrite text artifact and failed to regenerate embeddings/text. "
                f"Required artifact: {doc_texts_artifact}"
            )

    if embeddings is None:
        for candidate in embedding_candidates:
            if candidate.exists():
                embeddings = np.load(candidate)
                break

    if embeddings is None:
        raise FileNotFoundError(
            f"No embeddings found for {task_name} / {method_name}\n"
            f"Expected one of: {', '.join(str(p) for p in embedding_candidates)}\n"
            f"Run evaluation first: python scripts/run_eval.py --config-name {run_name}"
        )

    if embeddings is not None:
        is_similarity_matrix = embeddings.ndim == 2 and embeddings.shape[
            0
        ] == embeddings.shape[1] == len(documents)

        if is_similarity_matrix:
            logger.info(
                f"Loaded NxN similarity matrix ({embeddings.shape[0]}x{embeddings.shape[1]}) for {method_name}. "
                "This method supports heatmap visualization only."
            )
        else:
            nan_count = np.isnan(embeddings).sum()
            if nan_count > 0:
                valid_mask = ~np.isnan(embeddings).any(axis=1)
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) == 0:
                    raise ValueError(
                        f"All embeddings contain NaN values for {task_name} / {method_name}\n"
                        "This method may not be suitable for visualization."
                    )

                nan_vector_count = (~valid_mask).sum()
                logger.warning(f"Found {nan_vector_count} NaN vectors in embeddings")
                embeddings = embeddings[valid_mask]
                documents = [documents[i] for i in valid_indices]
                doc_texts = [doc_texts[i] for i in valid_indices]

    return (
        documents,
        doc_texts,
        dataset_name,
        embeddings,
        criterion,
        criterion_description,
    )
