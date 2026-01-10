"""Integration test for pipeline quality validation.

Tests that the full pipeline generates data meeting minimum quality thresholds.
Enables quick iteration to tune pipeline until quality bar is met.

Run with: pytest tests/benchmark/test_quality_validation.py -v -s
"""

import pytest

from multiview.benchmark.benchmark import Benchmark
from multiview.benchmark.task import Task

pytestmark = [pytest.mark.integration, pytest.mark.external]

# Quality thresholds
MIN_IDEAL_PCT = 40.0  # At least 40% rated 4 (ideal)
MAX_INVALID_PCT = 10.0  # At most 10% rated 1 (invalid)


def test_pipeline_quality_thresholds():
    """Test pipeline generates data meeting quality thresholds.

    Runs same flow as run_eval.py and validates results dict.
    """
    # Config (small for fast iteration)
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic_operations",
        "criterion_description": "Types of arithmetic operations used",
        "max_docs": 10,
        "max_triplets": 5,
        "triplet_style": "lm_all",
        "candidate_strategy": "multi",
        "use_spurious_hard_negs": True,
        "n_schema_samples": 5,
    }

    # Run pipeline (same as run_eval.py)
    task = Task(config=config)
    task.load_documents()
    task.augment_with_synthetic_documents()
    task.annotate_documents()
    task.create_triplets()
    quality_stats = task.rate_triplet_quality(min_quality=None)

    # Evaluate
    benchmark = Benchmark([task])
    results = benchmark.evaluate({
        "bm25": [{"name": "bm25"}],
        "lm_judge_triplet": [{
            "preset": "lmjudge_triplet_with_annotation_gemini",
            "name": "gemini_with_annotation",
        }],
    })

    # Validate quality thresholds
    percentages = quality_stats["percentages"]
    ideal_pct = percentages[4]
    invalid_pct = percentages[1]

    assert ideal_pct >= MIN_IDEAL_PCT, \
        f"Ideal {ideal_pct:.1f}% < {MIN_IDEAL_PCT}%"
    assert invalid_pct <= MAX_INVALID_PCT, \
        f"Invalid {invalid_pct:.1f}% > {MAX_INVALID_PCT}%"

    # Print summary
    print("\n" + "=" * 60)
    print("Quality Distribution:")
    for level in [4, 3, 2, 1]:
        label = ["", "Invalid", "Ambiguous", "Trivial", "Ideal"][level]
        print(f"  {level} ({label:10s}): {quality_stats['counts'][level]:3d} "
              f"({percentages[level]:5.1f}%)")

    print("\nEvaluation Results:")
    task_name = list(results.keys())[0]
    for method, metrics in results[task_name].items():
        print(f"  {method:25s}: {metrics['accuracy']:6.2%} "
              f"({metrics['n_correct']}/{metrics['n_total']})")
    print("=" * 60)
