from multiview.benchmark.metrics_utils import (
    metrics_from_correctness,
    metrics_from_outcomes,
    outcomes_from_pair_scores,
)


def test_outcomes_from_pair_scores_basic() -> None:
    pos = [0.9, 0.1, 0.5]
    neg = [0.1, 0.9, 0.5]
    assert outcomes_from_pair_scores(pos, neg) == [1, -1, 0]


def test_outcomes_from_pair_scores_tie_tolerance() -> None:
    pos = [1.0000, 1.0000, 1.0000]
    neg = [1.0001, 1.0000, 0.9999]
    # With tolerance, tiny diffs become ties
    assert outcomes_from_pair_scores(pos, neg, tie_tol=1e-3) == [0, 0, 0]
    # Without tolerance, diffs decide outcomes
    assert outcomes_from_pair_scores(pos, neg, tie_tol=0.0) == [-1, 0, 1]


def test_metrics_from_outcomes_excludes_ties_from_accuracy() -> None:
    metrics = metrics_from_outcomes([1, 1, 0, -1])  # judged=3, correct=2
    assert metrics["n_total"] == 4
    assert metrics["n_correct"] == 2
    assert metrics["n_incorrect"] == 1
    assert metrics["n_ties"] == 1
    assert metrics["accuracy"] == 2 / 3


def test_metrics_from_outcomes_all_ties_accuracy_zero() -> None:
    metrics = metrics_from_outcomes([0, 0, 0])
    assert metrics["n_total"] == 3
    assert metrics["n_correct"] == 0
    assert metrics["n_incorrect"] == 0
    assert metrics["n_ties"] == 3
    assert metrics["accuracy"] == 0.0


def test_metrics_from_correctness_with_ties_excludes_ties() -> None:
    correct = [True, False, True, False]
    is_tie = [False, False, True, True]
    metrics = metrics_from_correctness(correct, is_tie=is_tie, exclude_ties=True)
    assert metrics["n_total"] == 4
    assert metrics["n_ties"] == 2
    assert metrics["n_correct"] == 1
    assert metrics["n_incorrect"] == 1
    assert metrics["accuracy"] == 1 / 2
