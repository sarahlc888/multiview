"""Benchmark orchestration and evaluation.

This module coordinates running benchmarks across multiple tasks and methods.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from multiview.benchmark.artifacts import save_method_triplet_logs_jsonl
from multiview.benchmark.evaluation_utils import (
    compute_instruction_sensitivity,
    evaluate_method,
)

logger = logging.getLogger(__name__)


def save_embeddings_to_npy(
    embeddings: list,
    output_dir: Path,
    task_name: str,
    method_name: str,
) -> None:
    """Save embeddings array to NPY file.

    Args:
        embeddings: List of embedding vectors (from run_inference)
        output_dir: Base embeddings output directory
        task_name: Name of the task
        method_name: Name of the method
    """
    # Create task subdirectory
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings array
    embeddings_path = task_dir / f"{method_name}.npy"
    embeddings_array = np.array(embeddings)
    np.save(embeddings_path, embeddings_array)

    # Save document IDs (indices)
    doc_ids = np.arange(len(embeddings))
    doc_ids_path = task_dir / f"{method_name}_doc_ids.npy"
    np.save(doc_ids_path, doc_ids)

    logger.info(
        f"Saved embeddings to {embeddings_path} (shape: {embeddings_array.shape})"
    )
    logger.info(f"Saved document IDs to {doc_ids_path}")


def save_similarity_matrix_to_npy(
    similarity_matrix: np.ndarray,
    output_dir: Path,
    task_name: str,
    method_name: str,
) -> None:
    """Save similarity matrix (NxN) to NPY file for heatmap visualization.

    For methods like BM25 that compute pairwise similarity scores but don't produce
    embeddings, we save the full NxN similarity matrix. The heatmap visualizer will
    detect the NxN shape and use it directly instead of computing similarities.

    Args:
        similarity_matrix: NxN similarity matrix (e.g., from BM25)
        output_dir: Base embeddings output directory
        task_name: Name of the task
        method_name: Name of the method
    """
    # Create task subdirectory
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Save similarity matrix using same filename as embeddings for consistency
    # Visualizer will detect NxN shape vs NxD shape
    matrix_path = task_dir / f"{method_name}.npy"
    np.save(matrix_path, similarity_matrix)

    # Save document IDs (indices)
    n_docs = similarity_matrix.shape[0]
    doc_ids = np.arange(n_docs)
    doc_ids_path = task_dir / f"{method_name}_doc_ids.npy"
    np.save(doc_ids_path, doc_ids)

    logger.info(
        f"Saved similarity matrix to {matrix_path} (shape: {similarity_matrix.shape})"
    )
    logger.info(f"Saved document IDs to {doc_ids_path}")


class Benchmark:
    """Benchmark orchestrator.

    Manages a set of tasks and evaluates them using various methods.
    """

    def __init__(
        self,
        tasks: list,
        *,
        method_log_output_dir: str | None = None,
        embeddings_output_dir: str | None = None,
    ):
        """Initialize benchmark with tasks.

        Args:
            tasks: List of Task objects to evaluate
            method_log_output_dir: Directory to write per-triplet per-method logs (optional)
            embeddings_output_dir: Directory to write embeddings NPY files (optional)
        """
        self.tasks = tasks
        self.method_log_output_dir = (
            Path(method_log_output_dir) if method_log_output_dir else None
        )
        if self.method_log_output_dir:
            self.method_log_output_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_output_dir = (
            Path(embeddings_output_dir) if embeddings_output_dir else None
        )
        if self.embeddings_output_dir:
            self.embeddings_output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self, method_configs: dict[str, list[dict]]
    ) -> tuple[dict[str, Any], dict[str, dict[str, float | None]]]:
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
            Tuple of (results, instruction_sensitivity):
            - results: Dict with structure:
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
            - instruction_sensitivity: Dict with structure:
              {
                  "task_name": {
                      "method_name": float | None,
                      ...
                  },
                  ...
              }
        """
        results = {}
        skipped_methods = []  # Track skipped methods for summary

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

                        # Handle multi-trial results
                        if method_results.get("_multi_trial"):
                            # Expand trials into separate rows
                            for trial_result in method_results["trials"]:
                                trial_name = trial_result.get(
                                    "method_name", method_name
                                )
                                triplet_logs = trial_result.get("triplet_logs")
                                embeddings = trial_result.get("embeddings")
                                similarity_matrix = trial_result.get(
                                    "similarity_matrix"
                                )

                                if self.method_log_output_dir and triplet_logs:
                                    save_method_triplet_logs_jsonl(
                                        triplet_logs=triplet_logs,
                                        output_dir=self.method_log_output_dir,
                                        task_name=task_name,
                                        method_name=trial_name,
                                    )

                                # Save embeddings if available
                                if (
                                    self.embeddings_output_dir
                                    and embeddings is not None
                                ):
                                    save_embeddings_to_npy(
                                        embeddings=embeddings,
                                        output_dir=self.embeddings_output_dir,
                                        task_name=task_name,
                                        method_name=trial_name,
                                    )

                                # Save similarity matrix if available (e.g., from BM25)
                                if (
                                    self.embeddings_output_dir
                                    and similarity_matrix is not None
                                ):
                                    save_similarity_matrix_to_npy(
                                        similarity_matrix=similarity_matrix,
                                        output_dir=self.embeddings_output_dir,
                                        task_name=task_name,
                                        method_name=trial_name,
                                    )

                                # Store trial result as separate row (exclude large data)
                                task_results[trial_name] = {
                                    k: v
                                    for k, v in trial_result.items()
                                    if k
                                    not in (
                                        "triplet_logs",
                                        "embeddings",
                                        "similarity_matrix",
                                    )
                                }
                        else:
                            # Single result (normal case)
                            triplet_logs = method_results.get("triplet_logs")
                            embeddings = method_results.get("embeddings")
                            similarity_matrix = method_results.get("similarity_matrix")

                            if self.method_log_output_dir and triplet_logs:
                                save_method_triplet_logs_jsonl(
                                    triplet_logs=triplet_logs,
                                    output_dir=self.method_log_output_dir,
                                    task_name=task_name,
                                    method_name=method_name,
                                )

                            # Save embeddings if available
                            if self.embeddings_output_dir and embeddings is not None:
                                save_embeddings_to_npy(
                                    embeddings=embeddings,
                                    output_dir=self.embeddings_output_dir,
                                    task_name=task_name,
                                    method_name=method_name,
                                )

                            # Save similarity matrix if available (e.g., from BM25)
                            if (
                                self.embeddings_output_dir
                                and similarity_matrix is not None
                            ):
                                save_similarity_matrix_to_npy(
                                    similarity_matrix=similarity_matrix,
                                    output_dir=self.embeddings_output_dir,
                                    task_name=task_name,
                                    method_name=method_name,
                                )

                            # Keep returned results lean: logs and embeddings written to disk
                            task_results[method_name] = {
                                k: v
                                for k, v in method_results.items()
                                if k
                                not in (
                                    "triplet_logs",
                                    "embeddings",
                                    "similarity_matrix",
                                )
                            }

                        # Check if method was skipped before trying to log accuracy
                        if method_results.get("skipped"):
                            reason = method_results.get("reason", "Unknown reason")
                            logger.info(f"      Skipped: {reason}")
                            skipped_methods.append(
                                {
                                    "task": task_name,
                                    "method": method_name,
                                    "reason": reason,
                                }
                            )
                        elif method_results.get("_multi_trial"):
                            # Log summary for multi-trial results
                            num_trials = method_results.get("num_trials", 0)
                            successful_trials = [
                                t
                                for t in method_results["trials"]
                                if not t.get("error") and not t.get("skipped")
                            ]
                            if successful_trials:
                                avg_accuracy = sum(
                                    t.get("accuracy", 0.0) for t in successful_trials
                                ) / len(successful_trials)
                                logger.info(
                                    f"      Multi-trial result: {len(successful_trials)}/{num_trials} trials successful, "
                                    f"avg accuracy: {avg_accuracy:.2%}"
                                )
                            else:
                                logger.warning(
                                    f"      Multi-trial result: {num_trials} trials, all failed or skipped"
                                )
                        else:
                            # Single result (normal case)
                            if method_name in task_results:
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

        # Print summary of skipped methods
        if skipped_methods:
            logger.info("\n" + "=" * 80)
            logger.info("SKIPPED METHODS SUMMARY")
            logger.info("=" * 80)
            for skip in skipped_methods:
                logger.info(f"  [{skip['task']}] {skip['method']}: {skip['reason']}")
            logger.info("=" * 80 + "\n")
        else:
            logger.info("No methods were skipped")

        # Compute instruction sensitivity
        logger.info("Computing instruction sensitivity...")
        instruction_sensitivity = compute_instruction_sensitivity(
            results=results,
            method_configs=method_configs,
            tasks=self.tasks,
        )

        return results, instruction_sensitivity
