"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

import multiview.constants
from multiview.benchmark.triplets.quality_assurance import QUALITY_SCALE
from multiview.utils.logging_utils import setup_logging_from_config

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value to set
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Set random seed to {seed}")


def setup_benchmark_config(cfg: DictConfig) -> tuple[int, Path]:
    """Common setup for benchmark scripts.

    Performs logging setup, applies step_through config, sets random seed,
    and creates output base directory.

    Args:
        cfg: Hydra configuration object

    Returns:
        tuple of (seed, output_base): Configured seed and output base directory path
    """
    setup_logging_from_config(cfg)

    # Apply step_through config to constants module
    if hasattr(cfg, "step_through"):
        multiview.constants.STEP_THROUGH = cfg.step_through
        logger.debug(f"Set STEP_THROUGH to {cfg.step_through}")

    seed = getattr(cfg, "seed", 42)  # Default to 42 if not specified
    set_seed(seed)

    output_base = Path("outputs") / cfg.run_name

    return seed, output_base


def log_triplet_quality_distribution(
    *, stats: dict, task_name: str, min_quality: int | None = None
) -> None:
    """Log triplet quality distribution in a readable format.

    Args:
        stats: Quality statistics dictionary with 'n_total', 'counts', 'percentages'
        task_name: Name of the task being logged
        min_quality: Optional minimum quality threshold for filtering stats
    """
    logger.info("=" * 60)
    logger.info(f"QUALITY DISTRIBUTION - {task_name}")
    logger.info("=" * 60)
    logger.info(f"Total triplets rated: {stats['n_total']}")
    logger.info("")
    logger.info("Rating | Label      | Count | Percentage")
    logger.info("-------|------------|-------|------------")

    for level in [5, 4, 3, 2, 1]:  # best → worst
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


def save_benchmark_results(
    results: dict,
    results_dir: Path,
    run_name: str,
    instruction_sensitivity: dict[str, dict[str, float | None]] | None = None,
) -> None:
    """Save benchmark results in multiple formats and log summary.

    Saves results as JSON, CSV, and Markdown files, then logs a console summary.

    Args:
        results: Nested dict of {task_name: {method_name: metrics_dict}}
        results_dir: Directory path where results should be saved
        run_name: Name of the benchmark run for documentation
        instruction_sensitivity: Optional dict of {task_name: {method_name: sensitivity_score}}
    """
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
                "instruction_sensitivity",
            ]
        )
        for task_name, methods in results.items():
            for method_name, metrics in methods.items():
                # Get instruction sensitivity if available
                sensitivity = None
                if instruction_sensitivity and task_name in instruction_sensitivity:
                    sensitivity = instruction_sensitivity[task_name].get(method_name)

                writer.writerow(
                    [
                        task_name,
                        method_name,
                        f"{metrics.get('accuracy', 0):.4f}",
                        metrics.get("n_correct", 0),
                        metrics.get("n_incorrect", 0),
                        metrics.get("n_ties", 0),
                        metrics.get("n_total", 0),
                        f"{sensitivity:.4f}" if sensitivity is not None else "",
                    ]
                )
    logger.info(f"Saved CSV results to {csv_file}")

    # 3. Markdown format (human-readable documentation)
    md_file = results_dir / "results.md"
    with open(md_file, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(f"**Run**: {run_name}\n\n")
        f.write("---\n\n")
        for task_name, methods in results.items():
            f.write(f"## {task_name}\n\n")
            f.write("| Method | Accuracy | Correct | Total | Instr. Sensitivity |\n")
            f.write("|--------|----------|---------|-------|--------------------|\n")
            for method_name, metrics in methods.items():
                acc = metrics.get("accuracy", 0)
                correct = metrics.get("n_correct", 0)
                total = metrics.get("n_total", 0)

                # Get instruction sensitivity if available
                sensitivity = None
                if instruction_sensitivity and task_name in instruction_sensitivity:
                    sensitivity = instruction_sensitivity[task_name].get(method_name)

                sensitivity_str = (
                    f"{sensitivity:+.4f}" if sensitivity is not None else "—"
                )
                f.write(
                    f"| {method_name} | {acc:.2%} | {correct} | {total} | {sensitivity_str} |\n"
                )
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

            # Skip methods that were skipped (0/0)
            if total == 0:
                continue

            # Get instruction sensitivity if available
            sensitivity = None
            if instruction_sensitivity and task_name in instruction_sensitivity:
                sensitivity = instruction_sensitivity[task_name].get(method_name)

            sensitivity_str = (
                f" [Δ{sensitivity:+.4f}]" if sensitivity is not None else ""
            )

            # For LM judge triplet methods, display out of 8 (half of 16 due to bidirectional eval)
            if "triplet" in method_name:
                display_correct = correct / 2
                if display_correct % 1 == 0:
                    display_correct = int(display_correct)
                else:
                    display_correct = f"{display_correct:.1f}"
                display_total = total // 2
                logger.info(
                    f"  {method_name:35s}: {acc:6.2%} ({display_correct}/{display_total} correct){sensitivity_str}"
                )
            else:
                logger.info(
                    f"  {method_name:35s}: {acc:6.2%} ({correct}/{total} correct){sensitivity_str}"
                )
    logger.info("=" * 80 + "\n")
