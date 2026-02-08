"""Metrics for the rewriter GEPA workflow."""

from __future__ import annotations

from typing import Any


def metric(
    example: Any,
    pred: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    """Simple 0/1 triplet accuracy (works with ``dspy.Evaluate``)."""
    return 1.0 if int(example["label"]) == int(pred.closer) else 0.0


def gepa_metric(
    example: Any,
    pred: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> Any:
    """GEPA-friendly metric with margin feedback.

    Returns a ``ScoreWithFeedback`` when the DSPy GEPA module is available,
    otherwise a plain dict.
    """
    gold = int(example["label"])
    got = int(pred.closer)
    score = 1.0 if gold == got else 0.0
    margin = float(pred.cos_AB - pred.cos_AC)

    if score == 1.0:
        feedback = (
            f"Correct for aspect='{example['criteria']}'. Margin={margin:+.3f}. "
            "Keep rewrites concise."
        )
    else:
        if gold == 1:
            advice = "Focus on how to make the rewrite of A closer to the rewrite of C."
        else:
            advice = "Focus on how to make the rewrite of A closer to the rewrite of B."
        feedback = (
            f"Wrong sign under '{example['criteria']}'. Margin={margin:+.3f}. "
            f"{advice}"
        )

    try:
        from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

        return ScoreWithFeedback(score=score, feedback=feedback)
    except Exception:
        return {"score": score, "feedback": feedback}
