"""Main experiment pipeline."""

import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from multiview.benchmark.artifacts import (
    save_task_annotations,
    save_task_documents,
    save_task_triplets,
)
from multiview.benchmark.benchmark import Benchmark
from multiview.benchmark.task import Task
from multiview.benchmark.triplets.quality_assurance import QUALITY_SCALE
from multiview.inference.cost_tracker import print_summary as print_cost_summary
from multiview.utils.logging_utils import setup_logging_from_config

logger = logging.getLogger(__name__)


def log_triplet_quality_distribution(
    *, stats: dict, task_name: str, min_quality: int | None = None
) -> None:
    """Log triplet quality distribution in a readable format."""
    logger.info("=" * 60)
    logger.info(f"QUALITY DISTRIBUTION - {task_name}")
    logger.info("=" * 60)
    logger.info(f"Total triplets rated: {stats['n_total']}")
    logger.info("")
    logger.info("Rating | Label      | Count | Percentage")
    logger.info("-------|------------|-------|------------")

    for level in [4, 3, 2, 1]:  # best → worst
        count = stats["counts"][level]
        pct = stats["percentages"][level]
        label = QUALITY_SCALE[level]["class"]
        logger.info(f"   {level}   | {label:10s} | {count:5d} | {pct:5.1f}%")

    if min_quality is not None and "n_filtered" in stats:
        n_total = stats["n_total"]
        n_kept = stats["n_kept"]
        n_filtered = stats["n_filtered"]
        kept_pct = (n_kept / n_total * 100) if n_total else 0.0
        filtered_pct = (n_filtered / n_total * 100) if n_total else 0.0
        logger.info("")
        logger.info(f"Filtering (min_quality >= {min_quality}):")
        logger.info(f"  Kept:     {n_kept:5d} ({kept_pct:5.1f}%)")
        logger.info(f"  Removed:  {n_filtered:5d} ({filtered_pct:5.1f}%)")

    logger.info("=" * 60)


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
    documents_dir = output_base / "documents"
    annotations_dir = output_base / "annotations"
    results_dir = output_base / "results"
    method_logs_dir = output_base / "method_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    method_logs_dir.mkdir(parents=True, exist_ok=True)

    # create tasks
    tasks = []
    for task_spec in cfg.tasks.task_list:
        cur_task = Task(config={**cfg.tasks.defaults, **task_spec})
        cur_task.load_documents()
        cur_task.augment_with_synthetic_documents()
        if cur_task.triplet_style != "random":
            cur_task.annotate_documents()
        cur_task.create_triplets()

        # Rate and filter triplet quality if enabled
        if cfg.tasks.defaults.get("rate_triplet_quality", False):
            min_quality = cfg.tasks.defaults.get("min_triplet_quality")
            quality_stats = cur_task.rate_triplet_quality(min_quality=min_quality)
            log_triplet_quality_distribution(
                stats=quality_stats,
                task_name=cur_task.get_task_name(),
                min_quality=min_quality,
            )

        # Save documents and annotations
        save_task_documents(cur_task, documents_dir)
        if cur_task.document_annotations is not None:
            save_task_annotations(cur_task, annotations_dir)

        save_task_triplets(cur_task, triplets_dir)
        tasks.append(cur_task)

    # create benchmark object
    benchmark = Benchmark(tasks, method_log_output_dir=str(method_logs_dir))

    # evaluate multiview representation methods
    results = benchmark.evaluate(cfg.methods_to_evaluate)

    # Save results in multiple formats
    import csv

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
