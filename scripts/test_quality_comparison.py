#!/usr/bin/env python3
"""Test script to compare quality ratings with and without annotations.

Demonstrates automatic comparison feature - rates triplets twice to assess
whether annotation summaries help the LM make better quality judgments.

Usage: uv run python scripts/test_quality_comparison.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multiview.benchmark.task import Task

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Run quality rating comparison."""
    task = Task(
        config={
            "document_set": "gsm8k",
            "criterion": "arithmetic",
            "triplet_style": "lm_all",
            "max_docs": 20,
            "max_triplets": 5,
        }
    )

    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    # Compare ratings
    comparison = task.compare_quality_ratings_with_without_annotations()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Exact agreement: {comparison['agreement']['exact_match_rate']:.1%}")
    logger.info(f"Within-1 agreement: {comparison['agreement']['within_1_rate']:.1%}")

    if comparison["differences"]:
        logger.info(f"\nDifferences: {len(comparison['differences'])}")
        for diff in comparison["differences"][:5]:
            logger.info(
                f"  Triplet {diff['triplet_idx']}: "
                f"{diff['rating_without_annotation']} → {diff['rating_with_annotation']} "
                f"({diff['difference']:+d})"
            )
        if len(comparison["differences"]) > 5:
            logger.info(f"  ... and {len(comparison['differences']) - 5} more")


if __name__ == "__main__":
    main()
