"""Benchmark orchestration and evaluation.

This module coordinates running benchmarks across multiple tasks and methods.
"""

import logging
from typing import Any

from multiview.benchmark.methods import (
    evaluate_with_bm25,
    evaluate_with_embeddings,
    evaluate_with_lm_judge_pair,
    evaluate_with_lm_judge_triplet,
)

logger = logging.getLogger(__name__)


class Benchmark:
    """Benchmark orchestrator.

    Manages a set of tasks and evaluates them using various methods.
    """

    def __init__(self, tasks: list):
        """Initialize benchmark with tasks.

        Args:
            tasks: List of Task objects to evaluate
        """
        self.tasks = tasks

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
                        {"name": "bm25_annotated", "use_annotations": True},
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

                    # Call appropriate evaluation method
                    try:
                        if method_type == "lm_judge_triplet":
                            # Convert IDs to text dicts (doc IDs NEVER in prompts!)
                            triplet_dicts = [
                                {
                                    "anchor": task.documents[anchor_id],
                                    "positive": task.documents[positive_id],
                                    "negative": task.documents[negative_id],
                                    # Include IDs for annotation lookup
                                    "anchor_id": anchor_id,
                                    "positive_id": positive_id,
                                    "negative_id": negative_id,
                                }
                                for anchor_id, positive_id, negative_id in task.triplets
                            ]
                            method_results = self._evaluate_lm_judge_triplet(
                                triplet_dicts, task, method_config
                            )
                        elif method_type == "lm_judge_pair":
                            # Convert IDs to text dicts (doc IDs NEVER in prompts!)
                            triplet_dicts = [
                                {
                                    "anchor": task.documents[anchor_id],
                                    "positive": task.documents[positive_id],
                                    "negative": task.documents[negative_id],
                                    # Include IDs for annotation lookup
                                    "anchor_id": anchor_id,
                                    "positive_id": positive_id,
                                    "negative_id": negative_id,
                                }
                                for anchor_id, positive_id, negative_id in task.triplets
                            ]
                            method_results = self._evaluate_lm_judge_pair(
                                triplet_dicts, task, method_config
                            )
                        elif method_type == "embeddings":
                            # Pass documents and triplet IDs directly
                            method_results = self._evaluate_embeddings(
                                task.documents, task.triplets, task, method_config
                            )
                        elif method_type == "bm25":
                            # Pass documents and triplet IDs directly
                            method_results = self._evaluate_bm25(
                                task.documents, task.triplets, task, method_config
                            )
                        else:
                            logger.warning(f"Unknown method type: {method_type}")
                            continue

                        task_results[method_name] = method_results
                        logger.info(
                            f"      Result: {method_results['accuracy']:.2%} accuracy "
                            f"({method_results['n_correct']}/{method_results['n_total']} correct)"
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

    def _evaluate_lm_judge_triplet(
        self,
        triplet_dicts: list[dict],
        task: Any,
        method_config: dict,
    ) -> dict[str, Any]:
        """Evaluate using LM judge triplet comparison.

        Args:
            triplet_dicts: List of triplet dicts with anchor/positive/negative keys
            task: Task object (for accessing criterion metadata)
            method_config: Method configuration dict

        Returns:
            Evaluation results dict
        """
        preset = method_config.get(
            "preset", "lmjudge_triplet_plaintext_binaryhard_gemini"
        )
        model_override = method_config.get("model_override")

        # Get criterion info from task
        criterion = task.criterion_name
        criterion_description = task.config.get("criterion_description")

        # Determine if preset requires annotations based on preset name
        requires_annotations = "with_annotation" in preset.lower()

        # Pass annotations if preset requires them
        annotations = None
        if requires_annotations:
            if task.document_annotations is not None:
                annotations = task.document_annotations
                logger.info("Using annotations for evaluation (preset requires them)")
            else:
                logger.warning(
                    f"Preset '{preset}' requires annotations but task has none. "
                    "Evaluation may fail or produce incorrect results."
                )

        # Generate cache alias
        cache_alias = (
            f"{task.get_task_name()}_eval_{method_config.get('name', 'lmjudge')}"
        )

        # Override model if specified
        if model_override:
            # Get preset and override model
            from multiview.inference.presets import get_preset

            preset_config = get_preset(preset).with_overrides(model_name=model_override)  # noqa: F841
            # Use the config object directly instead of preset name
            # Note: This requires passing preset_config to evaluate_with_lm_judge_triplet
            logger.info(f"Overriding model to: {model_override}")
            # For now, we'll just log this - actual override would need API changes
            # TODO: Support model overrides in evaluate_with_lm_judge_triplet

        return evaluate_with_lm_judge_triplet(
            triplets=triplet_dicts,
            criterion=criterion,
            criterion_description=criterion_description,
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            annotations=annotations,
        )

    def _evaluate_bm25(
        self,
        documents: list[str],
        triplet_ids: list[tuple[int, int, int]],
        task: Any,
        method_config: dict,
    ) -> dict[str, Any]:
        """Evaluate using BM25 scoring.

        Args:
            documents: List of all document texts
            triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
            task: Task object (for accessing annotations if needed)
            method_config: Method configuration dict

        Returns:
            Evaluation results dict
        """
        use_annotations = method_config.get("use_annotations", False)

        # Note: Annotation support temporarily simplified
        # TODO: Handle annotations by passing annotation texts as documents parameter
        if use_annotations:
            logger.warning(
                "Annotation support not yet implemented with ID-based triplets, using raw documents"
            )
            use_annotations = False

        return evaluate_with_bm25(
            documents=documents,
            triplet_ids=triplet_ids,
            use_annotations=use_annotations,
        )

    def _evaluate_lm_judge_pair(
        self,
        triplet_dicts: list[dict],
        task: Any,
        method_config: dict,
    ) -> dict[str, Any]:
        """Evaluate using pairwise LM judge scoring.

        Args:
            triplet_dicts: List of triplet dicts with anchor/positive/negative keys
            task: Task object (for accessing criterion metadata)
            method_config: Method configuration dict

        Returns:
            Evaluation results dict
        """
        preset = method_config.get("preset", "lmjudge_pair_plaintext_likerthard_gemini")
        model_override = method_config.get("model_override")

        # Get criterion info from task
        criterion = task.criterion_name
        criterion_description = task.config.get("criterion_description")

        # Determine if preset requires annotations based on preset name
        requires_annotations = "with_annotation" in preset.lower()

        # Pass annotations if preset requires them
        annotations = None
        if requires_annotations:
            if task.document_annotations is not None:
                annotations = task.document_annotations
                logger.info("Using annotations for evaluation (preset requires them)")
            else:
                logger.warning(
                    f"Preset '{preset}' requires annotations but task has none. "
                    "Evaluation may fail or produce incorrect results."
                )

        # Generate cache alias
        cache_alias = (
            f"{task.get_task_name()}_eval_{method_config.get('name', 'lmjudge_pair')}"
        )

        # Override model if specified
        if model_override:
            from multiview.inference.presets import get_preset

            preset_config = get_preset(preset).with_overrides(model_name=model_override)  # noqa: F841
            logger.info(f"Overriding model to: {model_override}")
            # TODO: Support model overrides in evaluate_with_lm_judge_pair

        return evaluate_with_lm_judge_pair(
            triplets=triplet_dicts,
            criterion=criterion,
            criterion_description=criterion_description,
            lm_judge_preset=preset,
            cache_alias=cache_alias,
            annotations=annotations,
        )

    def _evaluate_embeddings(
        self,
        documents: list[str],
        triplet_ids: list[tuple[int, int, int]],
        task: Any,
        method_config: dict,
    ) -> dict[str, Any]:
        """Evaluate using embedding-based cosine similarity.

        Args:
            documents: List of all document texts
            triplet_ids: List of (anchor_id, positive_id, negative_id) tuples
            task: Task object (not used, but kept for consistency)
            method_config: Method configuration dict

        Returns:
            Evaluation results dict
        """
        preset = method_config.get("preset", "openai_embedding_small")
        model_override = method_config.get("model_override")
        preset_overrides = method_config.get("preset_overrides")

        # Generate cache alias
        cache_alias = (
            f"{task.get_task_name()}_eval_{method_config.get('name', 'embeddings')}"
        )

        # Override model if specified
        if model_override:
            from multiview.inference.presets import get_preset

            preset_config = get_preset(preset).with_overrides(model_name=model_override)  # noqa: F841
            logger.info(f"Overriding model to: {model_override}")
            # TODO: Support model overrides in evaluate_with_embeddings

        return evaluate_with_embeddings(
            documents=documents,
            triplet_ids=triplet_ids,
            embedding_preset=preset,
            cache_alias=cache_alias,
            preset_overrides=preset_overrides,
        )
