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
    """Test that different presets work correctly."""
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

    # Test rating WITHOUT annotations (explicit preset)
    logger.info("Testing rating WITHOUT annotations...")
    results_without = task.rate_triplet_quality(
        lm_judge_preset="lmjudge_quality_rating_gemini",
        min_quality=None,
    )

    assert "ratings" in results_without
    assert "counts" in results_without
    assert len(results_without["ratings"]) == len(task.triplets)

    # Test rating WITH annotations (explicit preset)
    logger.info("Testing rating WITH annotations...")
    results_with = task.rate_triplet_quality(
        lm_judge_preset="lmjudge_quality_rating_with_annotation_gemini",
        min_quality=None,
    )

    assert "ratings" in results_with
    assert "counts" in results_with
    assert len(results_with["ratings"]) == len(task.triplets)

    # Ratings might be different
    logger.info(f"Ratings without annotations: {results_without['ratings']}")
    logger.info(f"Ratings with annotations: {results_with['ratings']}")


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

    # When preset is None, should auto-select WITH annotations
    logger.info("Testing auto-selection with annotations...")
    results = task.rate_triplet_quality(lm_judge_preset=None, min_quality=None)

    assert "ratings" in results
    assert len(results["ratings"]) == len(task.triplets)


def test_rate_triplet_quality_validates_annotations_requirement():
    """Test that using annotation preset without annotations fails."""
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

    # Should raise error when trying to use annotation preset without annotations
    with pytest.raises(ValueError, match="requires annotations but task has none"):
        task.rate_triplet_quality(
            lm_judge_preset="lmjudge_quality_rating_with_annotation_gemini"
        )


@pytest.mark.parametrize(
    "document_set,criterion",
    [
        ("gsm8k", "arithmetic"),
    ],
)
def test_compare_quality_ratings(document_set, criterion):
    """Test the comparison method."""
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

    # Run comparison
    logger.info("Running comparison...")
    comparison = task.compare_quality_ratings_with_without_annotations()

    # Verify structure
    assert "ratings_without_annotations" in comparison
    assert "ratings_with_annotations" in comparison
    assert "agreement" in comparison
    assert "differences" in comparison

    # Check agreement stats
    agreement = comparison["agreement"]
    assert "n_triplets" in agreement
    assert "exact_matches" in agreement
    assert "exact_match_rate" in agreement
    assert "within_1_matches" in agreement
    assert "within_1_rate" in agreement

    # Check that n_triplets matches
    assert agreement["n_triplets"] == len(task.triplets)

    # Check that exact_match_rate is between 0 and 1
    assert 0 <= agreement["exact_match_rate"] <= 1
    assert 0 <= agreement["within_1_rate"] <= 1

    # Check differences structure
    for diff in comparison["differences"]:
        assert "triplet_idx" in diff
        assert "rating_without_annotation" in diff
        assert "rating_with_annotation" in diff
        assert "difference" in diff

    logger.info(f"Agreement rate: {agreement['exact_match_rate']:.1%}")
    logger.info(f"Within-1 rate: {agreement['within_1_rate']:.1%}")


def test_compare_quality_ratings_requires_annotations():
    """Test that comparison fails without annotations."""
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

    # Should raise error since there are no annotations
    with pytest.raises(ValueError, match="must have annotations"):
        task.compare_quality_ratings_with_without_annotations()
