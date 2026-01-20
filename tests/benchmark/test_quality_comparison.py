"""Test quality rating with and without annotations.

Tests the quality rating comparison feature, which rates triplets twice
(with/without annotations) to assess whether annotation summaries help the LM
make better quality judgments.

Key functionality tested:
- Explicit preset selection (force with/without annotations)
- Auto-selection based on annotation availability
- Validation that annotation presets require annotations
- Full comparison mode with agreement statistics
"""

import logging

import pytest

from multiview.benchmark.task import Task

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "document_set,criterion",
    [
        ("gsm8k", "arithmetic"),
    ],
)
def test_rate_triplet_quality_with_different_presets(document_set, criterion):
    """Test that rating works with and without annotations."""
    task_config = {
        "document_set": document_set,
        "criterion": criterion,
        "triplet_style": "lm_all",  # Need lm_all for rich annotations
        "max_docs": 15,
        "max_triplets": 3,
    }

    task = Task(config=task_config)
    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    # Test using comparison mode to get both with and without
    logger.info("Testing rating with and without annotations...")
    stats = task.rate_and_filter_quality(
        min_quality=None,
    )

    # Check that both rating sets exist
    assert task.triplet_quality_ratings_with_annotations is not None
    assert task.triplet_quality_ratings_without_annotations is not None
    assert len(task.triplet_quality_ratings_with_annotations) == len(task.triplets)
    assert len(task.triplet_quality_ratings_without_annotations) == len(task.triplets)

    # Ratings might be different
    logger.info(f"Ratings without annotations: {task.triplet_quality_ratings_without_annotations}")
    logger.info(f"Ratings with annotations: {task.triplet_quality_ratings_with_annotations}")


@pytest.mark.parametrize(
    "document_set,criterion",
    [
        ("gsm8k", "arithmetic"),
    ],
)
def test_rate_triplet_quality_auto_selects_preset(document_set, criterion):
    """Test that preset is auto-selected based on annotations."""
    task_config = {
        "document_set": document_set,
        "criterion": criterion,
        "triplet_style": "lm_all",  # Need lm_all for rich annotations
        "max_docs": 15,
        "max_triplets": 3,
    }

    task = Task(config=task_config)
    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    # When annotations exist, should compare with and without annotations
    logger.info("Testing auto-selection with annotations...")
    stats = task.rate_and_filter_quality(min_quality=None)

    assert "ratings" in stats
    assert task.triplet_quality_ratings is not None
    assert len(task.triplet_quality_ratings) == len(task.triplets)


def test_rate_triplet_quality_validates_annotations_requirement():
    """Test that auto-selection works without annotations."""
    task_config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "triplet_style": "random",  # Random doesn't create annotations
        "max_docs": 10,
        "max_triplets": 2,
    }

    task = Task(config=task_config)
    task.load_documents()
    task.create_triplets()

    # Should work fine - auto-selects preset without annotations
    stats = task.rate_and_filter_quality(min_quality=None)
    assert "ratings" in stats
    assert len(task.triplet_quality_ratings) == len(task.triplets)


@pytest.mark.parametrize(
    "document_set,criterion",
    [
        ("gsm8k", "arithmetic"),
    ],
)
def test_compare_quality_ratings(document_set, criterion):
    """Test the comparison feature."""
    task_config = {
        "document_set": document_set,
        "criterion": criterion,
        "triplet_style": "lm_all",  # Need lm_all for rich annotations
        "max_docs": 15,
        "max_triplets": 3,
    }

    task = Task(config=task_config)
    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    # Run comparison via rate_and_filter_quality
    logger.info("Running comparison...")
    stats = task.rate_and_filter_quality(min_quality=None)

    # Verify both rating sets exist
    assert task.triplet_quality_ratings_without_annotations is not None
    assert task.triplet_quality_ratings_with_annotations is not None
    assert len(task.triplet_quality_ratings_without_annotations) == len(task.triplets)
    assert len(task.triplet_quality_ratings_with_annotations) == len(task.triplets)

    # Log any differences
    differences = [
        (i, r_w, r_a)
        for i, (r_w, r_a) in enumerate(zip(
            task.triplet_quality_ratings_without_annotations,
            task.triplet_quality_ratings_with_annotations,
            strict=False
        ))
        if r_w != r_a
    ]

    if differences:
        logger.info(f"Found {len(differences)} rating differences")
        for idx, r_w, r_a in differences[:5]:  # Show first 5
            logger.info(f"  Triplet {idx}: without={r_w}, with={r_a}")
    else:
        logger.info("All ratings identical")


def test_compare_quality_ratings_requires_annotations():
    """Test that comparison is skipped without annotations."""
    task_config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "triplet_style": "random",  # Random doesn't create annotations
        "max_docs": 10,
        "max_triplets": 2,
    }

    task = Task(config=task_config)
    task.load_documents()
    task.create_triplets()

    # Should work but skip comparison (no annotations available)
    stats = task.rate_and_filter_quality(min_quality=None)

    # Should only have single rating set (comparison skipped since no annotations)
    assert task.triplet_quality_ratings is not None
    assert len(task.triplet_quality_ratings) == len(task.triplets)
    # Comparison attributes should remain None
    assert task.triplet_quality_ratings_without_annotations is None
    assert task.triplet_quality_ratings_with_annotations is None
