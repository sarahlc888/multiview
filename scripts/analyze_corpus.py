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
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# NOTE: This script is runnable directly; we modify sys.path above so these
# imports work without installing the package. Ruff flags E402 for this pattern.
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
