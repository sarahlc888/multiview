from multiview.benchmark.evaluation_utils import finalize_method_results


def test_finalize_pairwise_scores_adds_metrics_and_vectors() -> None:
    raw = {"positive_scores": [0.9, 0.1, 0.5], "negative_scores": [0.1, 0.9, 0.5]}
    out = finalize_method_results(raw)
    assert out["correct"] == [True, False, False]
    assert out["is_tie"] == [False, False, True]
    assert out["n_total"] == 3
    assert out["n_correct"] == 1
    assert out["n_incorrect"] == 2  # Ties counted as incorrect (exclude_ties=False)
    assert out["n_ties"] == 1
    assert out["accuracy"] == 1 / 3  # 1 correct out of 3 total


def test_finalize_outcomes_adds_metrics_and_vectors() -> None:
    raw = {"outcomes": [1, -1, 0, 1]}
    out = finalize_method_results(raw)
    assert out["correct"] == [True, False, False, True]
    assert out["is_tie"] == [False, False, True, False]
    assert out["n_total"] == 4
    assert out["n_correct"] == 2
    assert out["n_incorrect"] == 2  # Ties counted as incorrect (exclude_ties=False)
    assert out["n_ties"] == 1
    assert out["accuracy"] == 2 / 4  # 2 correct out of 4 total


def test_finalize_already_final_returns_copy() -> None:
    raw = {"accuracy": 0.5, "n_correct": 1, "n_incorrect": 1, "n_ties": 0, "n_total": 2}
    out = finalize_method_results(raw)
    # finalize_method_results returns a copy, not the same object
    assert out == raw
    assert out is not raw
