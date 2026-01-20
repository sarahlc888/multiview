"""Create evaluation artifacts: triplets, documents, and annotations."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from multiview.benchmark.artifacts import (
    load_documents_from_jsonl,
    save_task_annotations,
    save_task_documents,
    save_task_triplets,
)
from multiview.benchmark.synthesis.validation import validate_synthesis
from multiview.benchmark.task import Task
from multiview.inference.cost_tracker import print_summary as print_cost_summary
from multiview.utils.script_utils import (
    log_triplet_quality_distribution,
    setup_benchmark_config,
)

logger = logging.getLogger(__name__)


def process_task(task: Task, output_base: Path) -> dict | None:
    """Process a single task: load docs, create triplets, save artifacts."""
    task_name = task.get_task_name()

    # Artifact subdirectories (where create_eval.py saves artifacts)
    triplets_dir = output_base / "triplets"
    documents_dir = output_base / "documents"

    # Try to use cached triplets
    if task.can_use_cached_triplets():
        logger.info(f"Using cached triplets for {task_name}")
        # Load triplets directly without re-checking cache
        if not task.try_load_cached_triplets(str(triplets_dir)):
            raise RuntimeError(f"Failed to load cached triplets for {task_name}")
        # Load cached documents (includes synthetic docs from previous run)
        task.documents = load_documents_from_jsonl(
            output_dir=str(documents_dir), task_name=task_name
        )
    else:
        logger.info(f"Generating triplets for {task_name}")
        task.load_documents()
        task.augment_with_synthetic_documents()
        if task.triplet_style != "random":
            task.annotate_documents()
        task.create_triplets()

    # Rate and filter quality if enabled (check per-task config, not just defaults)
    quality_stats = None
    if task.config.get("rate_triplet_quality", False):
        quality_stats = task.rate_and_filter_quality(
            min_quality=task.config.get("min_triplet_quality"),
            output_dir=output_base / "triplets",
        )

    # Validate synthesis if performed
    if (
        task.document_annotations
        and task.num_synthetic_docs > 0
        and task.synthesis_metadata
    ):
        validate_synthesis(
            task.documents,
            task.document_annotations,
            task.synthesis_metadata,
            output_base / "validation",
            task_name,
            task.triplets,
            task.triplet_quality_ratings,
        )

    # Save artifacts
    save_task_documents(task, output_base / "documents")
    if task.document_annotations:
        save_task_annotations(task, output_base / "annotations")
    save_task_triplets(task, output_base / "triplets")

    return (
        {
            "task_name": task_name,
            "stats": quality_stats,
            "min_quality": task.config.get("min_triplet_quality"),
        }
        if quality_stats
        else None
    )


@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
def main(cfg: DictConfig):
    seed, output_base = setup_benchmark_config(cfg)
    logger.info(f"Creating triplets for: {cfg.run_name} with {seed=}")

    # Setup output directories
    for subdir in ["triplets", "documents", "annotations"]:
        (output_base / subdir).mkdir(parents=True, exist_ok=True)

    # Artifact subdirectories
    triplets_cache_dir = str(output_base / "triplets")

    # Process all tasks
    all_task_stats = []
    for task_spec in cfg.tasks.task_list:
        # Set triplet_cache_dir to point to the triplets subdirectory
        task = Task(
            config={
                **cfg.tasks.defaults,
                **task_spec,
                "run_name": cfg.run_name,
                "triplet_cache_dir": triplets_cache_dir,
                "reuse_cached_triplets": cfg.get("reuse_cached_triplets", True),
            }
        )
        if stats := process_task(task, output_base):
            all_task_stats.append(stats)

    # Log summary
    logger.info(f"\n{'='*80}\nARTIFACT CREATION COMPLETE\n{'='*80}")
    logger.info(f"Output: {output_base}")

    if all_task_stats:
        logger.info(f"\n{'='*80}\nTRIPLET QUALITY SUMMARY\n{'='*80}")
        for task_info in all_task_stats:
            log_triplet_quality_distribution(**task_info)

    print_cost_summary()


if __name__ == "__main__":
    main()
