"""Benchmark orchestration and evaluation.

This module coordinates running benchmarks across multiple tasks and methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from multiview.benchmark.artifacts import save_method_triplet_logs_jsonl
from multiview.benchmark.evaluation_utils import evaluate_method

logger = logging.getLogger(__name__)


class Benchmark:
    """Benchmark orchestrator.

    Manages a set of tasks and evaluates them using various methods.
    """

    def __init__(self, tasks: list, *, method_log_output_dir: str | None = None):
        """Initialize benchmark with tasks.

        Args:
            tasks: List of Task objects to evaluate
            method_log_output_dir: Directory to write per-triplet per-method logs (optional)
        """
        self.tasks = tasks
        self.method_log_output_dir = (
            Path(method_log_output_dir) if method_log_output_dir else None
        )
        if self.method_log_output_dir:
            self.method_log_output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self, method_configs: dict[str, list[dict]]) -> dict[str, Any]:
        """Evaluate all methods on all tasks.

        Args:
            method_configs: Dict mapping method types to lists of method configs.
                Example:
                {
                    "lm_judge_triplet": [
                        {"preset": "lmjudge_triplet_plaintext_binaryhard_gemini", "name": "gemini_flash"},
                    ],
                    "lm_judge_pair": [
                        {"preset": "lmjudge_pair_plaintext_likerthard_gemini", "name": "gemini_pair"},
                    ],
                    "embeddings": [
                        {"preset": "openai_embedding_small", "name": "openai_small"},
                        {"preset": "openai_embedding_large", "name": "openai_large"},
                    ],
                    "bm25": [
                        {"name": "bm25"},
                    ],
                }

        Returns:
            Dict with structure:
            {
                "task_name": {
                    "method_name": {
                        "accuracy": float,
                        "n_correct": int,
                        "n_incorrect": int,
                        "n_ties": int,
                        "n_total": int,
                    },
                    ...
                },
                ...
            }
        """
        results = {}

        logger.info(f"Evaluating {len(self.tasks)} tasks")

        for task in self.tasks:
            task_name = task.get_task_name()
            logger.info(f"Evaluating task: {task_name}")

            if task.triplets is None:
                logger.warning(f"Task {task_name} has no triplets, skipping evaluation")
                continue

            task_results = {}

            # Evaluate each method type
            for method_type, method_list in method_configs.items():
                if not method_list:
                    continue

                logger.info(f"  Method type: {method_type}")

                # Evaluate each method configuration
                for method_config in method_list:
                    method_name = method_config.get("name", f"unnamed_{method_type}")
                    logger.info(f"    Method: {method_name}")

                    try:
                        method_results = evaluate_method(
                            method_type=method_type,
                            task=task,
                            method_config=method_config,
                        )
                        triplet_logs = method_results.get("triplet_logs")

                        if self.method_log_output_dir and triplet_logs:
                            save_method_triplet_logs_jsonl(
                                triplet_logs=triplet_logs,
                                output_dir=self.method_log_output_dir,
                                task_name=task_name,
                                method_name=method_name,
                            )

                        # Keep returned results lean: logs are written to disk.
                        task_results[method_name] = {
                            k: v
                            for k, v in method_results.items()
                            if k != "triplet_logs"
                        }
                        logger.info(
                            f"      Result: {task_results[method_name]['accuracy']:.2%} accuracy "
                            f"({task_results[method_name]['n_correct']}/{task_results[method_name]['n_total']} correct)"
                        )

                    except Exception as e:
                        logger.error(
                            f"Error evaluating {method_name}: {e}", exc_info=True
                        )
                        task_results[method_name] = {
                            "error": str(e),
                            "accuracy": 0.0,
                            "n_correct": 0,
                            "n_incorrect": 0,
                            "n_ties": 0,
                            "n_total": 0,
                        }

            results[task_name] = task_results

        logger.info("Benchmark evaluation complete")
        return results
