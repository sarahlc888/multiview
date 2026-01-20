"""Run evaluation on pre-generated triplets."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from multiview.benchmark.artifacts import load_documents_from_jsonl
from multiview.benchmark.benchmark import Benchmark
from multiview.benchmark.task import Task
from multiview.inference.cost_tracker import print_summary as print_cost_summary
from multiview.utils.script_utils import (
    save_benchmark_results,
    setup_benchmark_config,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    seed, output_base = setup_benchmark_config(cfg)
    logger.info(f"Running evaluation: {cfg.run_name} with {seed=}")

    # Setup output directories
    results_dir = output_base / "results"
    method_logs_dir = output_base / "method_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    method_logs_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached triplets first
    logger.info(f"Looking for cached triplets in {output_base}")

    triplets_dir = output_base / "triplets"
    documents_dir = output_base / "documents"
    if not triplets_dir.exists() or not documents_dir.exists():
        logger.error(f"✗ Evaluation artifacts not found in {output_base}")
        logger.error("")
        logger.error("To generate evaluation artifacts, run:")
        logger.error(f"  python scripts/create_eval.py run_name={cfg.run_name}")
        logger.error("")
        raise FileNotFoundError(
            "Missing evaluation artifacts. Run create_eval.py first."
        )

    # Load tasks from saved artifacts
    # create_eval.py saves artifacts to subdirectories (triplets/, documents/, etc.)
    triplets_cache_dir = str(output_base / "triplets")
    documents_cache_dir = str(output_base / "documents")

    tasks = []
    for task_spec in cfg.tasks.task_list:
        # Set triplet_cache_dir to point to the triplets subdirectory
        cur_task = Task(
            config={
                **cfg.tasks.defaults,
                **task_spec,
                "run_name": cfg.run_name,
                "triplet_cache_dir": triplets_cache_dir,
                "reuse_cached_triplets": cfg.get("reuse_cached_triplets", True),
            }
        )

        task_name = cur_task.get_task_name()

        # Check if cached triplets are available
        if not cur_task.can_use_cached_triplets():
            logger.error(f"✗ No cached triplets found for {task_name}")
            logger.error("")
            logger.error("To generate evaluation artifacts, run:")
            logger.error(f"  python scripts/create_eval.py run_name={cfg.run_name}")
            logger.error("")
            raise FileNotFoundError(
                f"Missing cached triplets for {task_name}. Run create_eval.py first."
            )

        logger.info(f"✓ Loading cached artifacts for {task_name}")
        # Load triplets directly without re-checking cache
        if not cur_task.try_load_cached_triplets(triplets_cache_dir):
            raise RuntimeError(f"Failed to load cached triplets for {task_name}")
        # Load cached documents (includes synthetic docs from previous run)
        # The cached triplet indices reference these exact documents
        cur_task.documents = load_documents_from_jsonl(
            output_dir=documents_cache_dir,
            task_name=task_name,
        )
        logger.info(
            f"  Loaded {len(cur_task.documents)} cached documents (including synthetic)"
        )

        tasks.append(cur_task)

    # create benchmark object
    benchmark = Benchmark(tasks, method_log_output_dir=str(method_logs_dir))

    # evaluate multiview representation methods
    results, instruction_sensitivity = benchmark.evaluate(cfg.methods_to_evaluate)

    # Save results in multiple formats and log summary
    save_benchmark_results(results, results_dir, cfg.run_name, instruction_sensitivity)

    # Print API cost summary
    print_cost_summary()


if __name__ == "__main__":
    main()
