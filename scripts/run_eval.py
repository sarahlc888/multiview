"""Main experiment pipeline."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multiview.benchmark.benchmark import Benchmark
from multiview.benchmark.task import Task
from multiview.utils.logging_utils import setup_logging_from_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    setup_logging_from_config(cfg)

    logger.info(f"Running benchmark: {cfg.run_name}")
    # TODO: set_seed(cfg.seed)

    # Setup output directories
    output_base = Path("outputs") / cfg.run_name
    triplets_dir = output_base / "triplets"
    results_dir = output_base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # create tasks
    tasks = []
    for task_spec in cfg.tasks.task_list:
        cur_task = Task(config={**cfg.tasks.defaults, **task_spec})
        cur_task.load_documents()
        if cur_task.triplet_style != "random":
            cur_task.annotate_documents()
        cur_task.create_triplets()
        cur_task.save_triplets(triplets_dir)
        tasks.append(cur_task)

    # create benchmark object
    benchmark = Benchmark(tasks)

    # evaluate multiview representation methods
    results = benchmark.evaluate(cfg.methods_to_evaluate)

    # Save results to JSON
    import json

    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_file}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS SUMMARY")
    logger.info("=" * 80)
    for task_name, methods in results.items():
        logger.info(f"\nTask: {task_name}")
        for method_name, metrics in methods.items():
            acc = metrics.get("accuracy", 0)
            correct = metrics.get("n_correct", 0)
            total = metrics.get("n_total", 0)
            logger.info(f"  {method_name:35s}: {acc:6.2%} ({correct}/{total} correct)")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
