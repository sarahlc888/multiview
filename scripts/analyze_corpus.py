#!/usr/bin/env python3
"""Hydra entrypoint for config-driven benchmark corpus visualization.

Current usage:
- Invoked via Hydra config (same task/method config shape as `run_eval.py`)
- Treats tasks as corpus-visualization units (not triplet-eval units)
- Generates visualization artifacts under `outputs/viz/<run_name>/corpus`
- Applies task/method filters from `tasks.task_list` and `methods_to_evaluate`

Notes:
- Benchmark docs/embeddings are loaded via `multiview.utils.benchmark_loading`,
  including full-corpus regeneration with eval-consistent cache semantics.
- The executable entrypoint is Hydra-only (`hydra_main`).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# NOTE: This script is runnable directly; we modify sys.path above so these
# imports work without installing the package. Ruff flags E402 for this pattern.
from multiview.benchmark.artifacts import (  # noqa: E402
    load_documents_from_jsonl,
    save_task_documents,
)
from multiview.benchmark.evaluation_utils import (  # noqa: E402
    _extract_document_texts,
    _get_default_cache_name,
    make_cache_alias,
)
from multiview.benchmark.task import Task  # noqa: E402
from multiview.utils.benchmark_loading import (  # noqa: E402
    _method_supports_full_regeneration,
    _regenerate_full_method_embeddings,
    _resolve_required_criterion_description,
)
from multiview.utils.script_utils import setup_benchmark_config  # noqa: E402

logger = logging.getLogger(__name__)


def _build_task_name_from_config(
    task_config: dict[str, Any], defaults: dict[str, Any]
) -> str | None:
    """Build task artifact name from task/default config values."""
    document_set = task_config.get("document_set")
    criterion = task_config.get("criterion")
    if not document_set or not criterion:
        return None

    merged = {**defaults, **task_config}
    base_name = f"{document_set}__{criterion}"
    if not merged.get("use_config_suffix", True):
        return base_name

    style_abbrev = {
        "random": "rnd",
        "prelabeled": "pre",
        "lm": "hn",
        "lm_all": "hn",
        "lm_category": "cat",
        "lm_tags": "tag",
        "lm_summary_dict": "sdict",
        "lm_summary_sentence": "ssent",
    }
    triplet_style = merged.get("triplet_style", "lm")
    suffix_style = style_abbrev.get(triplet_style, str(triplet_style)[:4])
    max_triplets = merged.get("max_triplets", 0)
    return f"{base_name}__{suffix_style}__{max_triplets}"


def _extract_method_filter_from_config(cfg: Any) -> list[str]:
    """Extract enabled method names from benchmark config."""
    method_names: list[str] = []
    methods_to_evaluate = cfg.get("methods_to_evaluate")
    if not methods_to_evaluate:
        return method_names

    for _method_type, method_list in methods_to_evaluate.items():
        if not method_list:
            continue
        for method_config in method_list:
            method_name = method_config.get("name")
            if not method_name:
                continue
            num_trials = method_config.get("num_trials", 1)
            if num_trials and num_trials > 1:
                for trial_num in range(1, num_trials + 1):
                    method_names.append(f"{method_name}_trial{trial_num}")
            else:
                method_names.append(method_name)
    return method_names


def _extract_task_filter_from_config(cfg: Any) -> list[str]:
    """Extract task artifact names from benchmark config."""
    task_names: list[str] = []
    tasks_cfg = cfg.get("tasks")
    if not tasks_cfg:
        return task_names

    defaults = OmegaConf.to_container(tasks_cfg.get("defaults", {}), resolve=True) or {}
    task_list = tasks_cfg.get("task_list", [])
    for task in task_list:
        task_dict = OmegaConf.to_container(task, resolve=True) or {}
        task_name = _build_task_name_from_config(task_dict, defaults)
        if task_name:
            task_names.append(task_name)
    return task_names


def _ensure_artifacts_for_visualization(cfg: DictConfig) -> None:
    """Prepare documents and embeddings so visualization can run standalone.

    This mirrors the artifact-creation logic in create_eval.py and the
    embedding-generation logic in run_eval.py, using identical cache aliases
    so that inference results are shared across all three scripts.

    Phase 1 — Documents: for each configured task, load documents from the
    docset (or from disk if create_eval already ran) and persist them.

    Phase 2 — Embeddings: for each task + method combination that supports
    full-corpus embedding regeneration, compute embeddings (hitting the
    inference cache when possible) and save them to
    ``outputs/{run}/corpus/embeddings/{task}/{method}.npy``.
    """
    run_name = cfg.run_name
    output_base = Path("outputs") / run_name
    documents_dir = output_base / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)

    defaults = OmegaConf.to_container(cfg.tasks.get("defaults", {}), resolve=True) or {}

    # ------------------------------------------------------------------
    # Phase 1: ensure documents exist on disk for every configured task
    # ------------------------------------------------------------------
    tasks: list[Task] = []
    for task_spec in cfg.tasks.task_list:
        task_dict = OmegaConf.to_container(task_spec, resolve=True) or {}
        task = Task(
            config={
                **defaults,
                **task_dict,
                "run_name": run_name,
            }
        )
        task_name = task.get_task_name()

        docs_file = documents_dir / task_name / "documents.jsonl"
        if docs_file.exists():
            logger.info(f"Loading existing documents for {task_name}")
            task.documents = load_documents_from_jsonl(
                output_dir=str(documents_dir), task_name=task_name
            )
        else:
            logger.info(f"Loading documents from docset for {task_name}")
            task.load_documents()
            save_task_documents(task, documents_dir)

        tasks.append(task)

    # ------------------------------------------------------------------
    # Phase 2: ensure embeddings exist for each task × method
    # ------------------------------------------------------------------
    methods_to_evaluate = cfg.get("methods_to_evaluate")
    if not methods_to_evaluate:
        logger.info("No methods_to_evaluate configured; skipping embedding phase")
        return

    corpus_emb_base = output_base / "corpus" / "embeddings"

    for task in tasks:
        task_name = task.get_task_name()
        doc_texts = _extract_document_texts(task)

        for method_type, method_list in methods_to_evaluate.items():
            if not method_list:
                continue
            method_type_str = str(method_type)

            for method_cfg_raw in method_list:
                method_dict = OmegaConf.to_container(method_cfg_raw, resolve=True) or {}
                method_name = method_dict.get("name", method_type_str)

                # Build method_metadata in the same shape that
                # _method_supports_full_regeneration expects.
                method_metadata: dict[str, Any] = {
                    "method_type": method_type_str,
                }
                for key in [
                    "embedding_preset",
                    "summary_preset",
                    "num_summaries",
                    "expansion_preset",
                    "num_expansions",
                    "dev_set_size",
                    "dev_set_seed",
                    "preset",
                ]:
                    if key in method_dict:
                        method_metadata[key] = method_dict[key]

                if not _method_supports_full_regeneration(method_metadata):
                    logger.debug(
                        f"Skipping {task_name}/{method_name}: "
                        f"method type '{method_type_str}' does not support "
                        "full regeneration"
                    )
                    continue

                corpus_emb_file = corpus_emb_base / task_name / f"{method_name}.npy"

                # Build cache alias identically to the eval pipeline
                cache_alias = make_cache_alias(
                    task=task,
                    method_config=method_dict,
                    default_name=_get_default_cache_name(method_type_str, method_dict),
                )

                # Resolve criterion description
                dataset_name = task_name.split("__")[0]
                criterion_description = _resolve_required_criterion_description(
                    dataset_name=dataset_name,
                    criterion=task.criterion_name,
                    method_metadata=method_metadata,
                    output_base=output_base,
                    task_name=task_name,
                    context=(
                        f"analyze_corpus artifact prep for "
                        f"{task_name}/{method_name}"
                    ),
                )

                logger.info(
                    f"Generating embeddings for {task_name}/{method_name} "
                    f"({len(task.documents)} docs)"
                )
                embeddings = _regenerate_full_method_embeddings(
                    run_name=run_name,
                    output_base=output_base,
                    task_name=task_name,
                    method_name=method_name,
                    documents=task.documents,
                    doc_texts=doc_texts,
                    criterion=task.criterion_name,
                    criterion_description=criterion_description,
                    method_metadata=method_metadata,
                    cache_alias=cache_alias,
                )

                if embeddings is not None:
                    corpus_emb_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(corpus_emb_file, embeddings)
                    logger.info(
                        f"Saved embeddings {embeddings.shape} to " f"{corpus_emb_file}"
                    )
                else:
                    logger.warning(
                        f"Failed to generate embeddings for "
                        f"{task_name}/{method_name}"
                    )


@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    """Hydra-native entrypoint for benchmark visualization."""
    from multiview.utils.visualization_utils import (
        generate_visualizations_for_benchmark,
    )

    # Guard against Hydra changing working directory in some environments.
    os.chdir(get_original_cwd())

    # Mirror run_eval setup (logging, seed, step-through handling)
    _seed, _output_base = setup_benchmark_config(cfg)

    run_name = cfg.get("run_name")
    if not run_name:
        raise ValueError("Hydra config must define run_name")

    viz_cfg = cfg.get("visualization", {})
    reducers = list(viz_cfg.get("reducers", ["tsne"]))
    output_base = viz_cfg.get("output_dir", "outputs/viz")
    use_thumbnails = viz_cfg.get("thumbnails", True)

    # Ensure documents and embeddings exist before visualization.
    _ensure_artifacts_for_visualization(cfg)

    task_filter = _extract_task_filter_from_config(cfg)
    method_filter = _extract_method_filter_from_config(cfg)
    output_benchmark_run = f"{run_name}/corpus"

    logger.info("=" * 60)
    logger.info("HYDRA CONFIG-DRIVEN BENCHMARK VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"Benchmark run: {run_name}")
    logger.info(f"Output benchmark key: {output_benchmark_run}")
    logger.info(f"Reducers: {', '.join(reducers)}")
    logger.info(f"Output: {output_base}")
    logger.info(f"Task filter: {len(task_filter)} configured tasks")
    logger.info(f"Method filter: {len(method_filter)} configured methods")
    logger.info("")

    success_count, fail_count = generate_visualizations_for_benchmark(
        benchmark_run=run_name,
        reducers=reducers,
        output_base=output_base,
        output_benchmark_run=output_benchmark_run,
        use_thumbnails=use_thumbnails,
        task_filter=task_filter if task_filter else None,
        method_filter=method_filter if method_filter else None,
        quiet=False,
    )

    if fail_count > 0:
        logger.warning(
            f"Some visualizations failed ({fail_count}/{success_count + fail_count})"
        )


if __name__ == "__main__":
    main()
