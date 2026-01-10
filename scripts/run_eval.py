"""Main experiment pipeline."""

import logging
import random
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from multiview.benchmark.benchmark import Benchmark
from multiview.benchmark.task import Task
from multiview.inference.cost_tracker import print_summary as print_cost_summary
from multiview.utils.logging_utils import setup_logging_from_config

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")


@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    setup_logging_from_config(cfg)

    logger.info(f"Running benchmark: {cfg.run_name}")
    set_seed(cfg.seed)

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
        cur_task.augment_with_synthetic_documents()
        if cur_task.triplet_style != "random":
            cur_task.annotate_documents()
        cur_task.create_triplets()
        cur_task.save_triplets(triplets_dir)
        tasks.append(cur_task)

    # create benchmark object
    benchmark = Benchmark(tasks)

    # evaluate multiview representation methods
    results = benchmark.evaluate(cfg.methods_to_evaluate)

    # Save results in multiple formats
    import csv
    import json

    # 1. JSON format (machine-readable, full detail)
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved JSON results to {results_file}")

    # 2. CSV format (spreadsheet-friendly)
    csv_file = results_dir / "results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task",
                "method",
                "accuracy",
                "n_correct",
                "n_incorrect",
                "n_ties",
                "n_total",
            ]
        )
        for task_name, methods in results.items():
            for method_name, metrics in methods.items():
                writer.writerow(
                    [
                        task_name,
                        method_name,
                        f"{metrics.get('accuracy', 0):.4f}",
                        metrics.get("n_correct", 0),
                        metrics.get("n_incorrect", 0),
                        metrics.get("n_ties", 0),
                        metrics.get("n_total", 0),
                    ]
                )
    logger.info(f"Saved CSV results to {csv_file}")

    # 3. Markdown format (human-readable documentation)
    md_file = results_dir / "results.md"
    with open(md_file, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"**Run**: {cfg.run_name}\n\n")
        f.write("---\n\n")
        for task_name, methods in results.items():
            f.write(f"## {task_name}\n\n")
            f.write("| Method | Accuracy | Correct | Total |\n")
            f.write("|--------|----------|---------|-------|\n")
            for method_name, metrics in methods.items():
                acc = metrics.get("accuracy", 0)
                correct = metrics.get("n_correct", 0)
                total = metrics.get("n_total", 0)
                f.write(f"| {method_name} | {acc:.2%} | {correct} | {total} |\n")
            f.write("\n")
    logger.info(f"Saved markdown summary to {md_file}")

    # 4. Console summary table
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

    # Print API cost summary
    print_cost_summary()


if __name__ == "__main__":
    main()
