"""Load multiview triplets as DSPy Examples for GEPA tuning workflows."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import dspy

from multiview.docsets.criteria_metadata import load_criteria_metadata

logger = logging.getLogger(__name__)


def _extract_text(doc: str | dict) -> str:
    """Return the text content of a document (string or dict with ``text`` key)."""
    if isinstance(doc, dict):
        return doc.get("text", str(doc))
    return str(doc)


def load_triplets_as_dspy_examples(
    triplets_dir: str | Path,
    *,
    min_quality: int = 0,
    use_criterion_description: bool = True,
    augment_flip: bool = True,
) -> list[dspy.Example]:
    """Load triplets from a single benchmark task directory as DSPy Examples.

    Each example has fields ``A``, ``B``, ``C``, ``label``, ``criteria``, and
    optionally ``criterion_description``.  A random 50% pos/neg flip is applied
    for augmentation when *augment_flip* is ``True``.

    Args:
        triplets_dir: Path to a task directory containing ``triplets.json`` and
            ``triplet_config.json``.
        min_quality: Minimum ``quality_assessment.rating`` to include (0 = no filter).
        use_criterion_description: Resolve a human-readable description for the
            criterion via ``available_criteria.yaml``.
        augment_flip: Randomly swap positive/negative with label adjustment.

    Returns:
        List of ``dspy.Example`` objects with inputs ``A``, ``B``, ``C``, ``criteria``.
    """
    triplets_dir = Path(triplets_dir)
    triplets_file = triplets_dir / "triplets.json"
    config_file = triplets_dir / "triplet_config.json"

    if not triplets_file.exists():
        logger.warning("No triplets.json in %s – skipping", triplets_dir)
        return []

    with open(triplets_file) as f:
        triplets: list[dict[str, Any]] = json.load(f)

    # Read config for criterion / dataset name
    criterion = ""
    dataset_name = ""
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        criterion = config.get("criterion", "")
        dataset_name = config.get("document_set", "")

    # Resolve criterion description
    criterion_description = ""
    if use_criterion_description and criterion and dataset_name:
        try:
            all_meta = load_criteria_metadata()
            dataset_meta = all_meta.get(dataset_name, {})
            crit_meta = dataset_meta.get(criterion, {})
            criterion_description = crit_meta.get("description", "")
        except Exception:
            logger.debug(
                "Could not resolve criterion description for %s/%s",
                dataset_name,
                criterion,
            )

    # Build a descriptive criteria string
    criteria_str = criterion
    if criterion_description:
        criteria_str = f"{criterion}: {criterion_description}"

    examples: list[dspy.Example] = []
    for triplet in triplets:
        # Quality filter
        qa = triplet.get("quality_assessment", {})
        if isinstance(qa, dict) and min_quality > 0:
            rating = qa.get("rating", 5)
            if rating < min_quality:
                continue

        anchor = _extract_text(triplet["anchor"])
        positive = _extract_text(triplet["positive"])
        negative = _extract_text(triplet["negative"])

        if augment_flip and random.random() < 0.5:
            # Flip: label=1 means A is closer to C
            examples.append(
                dspy.Example(
                    {
                        "A": anchor,
                        "B": negative,
                        "C": positive,
                        "label": 1,
                        "criteria": criteria_str,
                    }
                ).with_inputs("A", "B", "C", "criteria")
            )
        else:
            # Normal: label=0 means A is closer to B
            examples.append(
                dspy.Example(
                    {
                        "A": anchor,
                        "B": positive,
                        "C": negative,
                        "label": 0,
                        "criteria": criteria_str,
                    }
                ).with_inputs("A", "B", "C", "criteria")
            )

    logger.info(
        "Loaded %d examples from %s (criterion=%s)",
        len(examples),
        triplets_dir,
        criterion,
    )
    return examples


def load_triplets_from_benchmark(
    benchmark_dir: str | Path,
    *,
    task_filter: str | None = None,
    min_quality: int = 0,
    use_criterion_description: bool = True,
    augment_flip: bool = True,
) -> list[dspy.Example]:
    """Scan all task sub-directories under a benchmark run and aggregate examples.

    Args:
        benchmark_dir: Root benchmark output directory (contains ``triplets/``).
        task_filter: Optional substring filter on task directory names.
        min_quality: Minimum quality rating (passed through).
        use_criterion_description: Whether to resolve criterion descriptions.
        augment_flip: Random pos/neg flip augmentation.

    Returns:
        Aggregated list of ``dspy.Example`` objects from all matching tasks.
    """
    benchmark_dir = Path(benchmark_dir)
    triplets_root = benchmark_dir / "triplets"

    if not triplets_root.is_dir():
        raise FileNotFoundError(
            f"No triplets/ directory found under {benchmark_dir}. "
            "Please provide a valid benchmark output directory."
        )

    all_examples: list[dspy.Example] = []
    for task_dir in sorted(triplets_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter and task_filter not in task_dir.name:
            continue

        examples = load_triplets_as_dspy_examples(
            task_dir,
            min_quality=min_quality,
            use_criterion_description=use_criterion_description,
            augment_flip=augment_flip,
        )
        all_examples.extend(examples)

    logger.info(
        "Loaded %d total examples from %s (%d task dirs)",
        len(all_examples),
        benchmark_dir,
        sum(1 for d in triplets_root.iterdir() if d.is_dir()),
    )
    return all_examples
