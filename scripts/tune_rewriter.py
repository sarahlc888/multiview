#!/usr/bin/env python
"""Entry point for rewriter GEPA prompt tuning.

Optimises a document-rewriting prompt so that, after rewriting anchor /
positive / negative documents according to a criterion, the embedding
cosine distance correctly reflects triplet order.

Usage:
    python scripts/tune_rewriter.py                                 # defaults
    python scripts/tune_rewriter.py data.benchmark_dir=outputs/benchmark_haiku \\
        gepa.rollout_budget=10
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import dspy
import hydra
from omegaconf import DictConfig

from multiview.tuning.data_loading import load_triplets_from_benchmark
from multiview.tuning.rewriter_gepa.metric import gepa_metric
from multiview.tuning.rewriter_gepa.module import CheckTripletSimple
from multiview.tuning.utils import (
    LearningCurveTracker,
    extract_score,
    save_detailed_results,
    save_prompt_details,
    save_readout_file,
    setup_logging,
    setup_proposal_prompt_logging,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="tune_rewriter", version_base=None)
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    random.seed(cfg.seed)

    # ── output dir ────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir) / f"gepa_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    log_file = setup_logging(output_dir)
    print(f"Verbose logs: {log_file}")

    # ── DSPy LM + embedder ───────────────────────────────────────────────
    lm = dspy.LM(cfg.lm.model)
    embedder = dspy.Embedder(cfg.embedder.model)
    dspy.configure(lm=lm)

    # ── data ──────────────────────────────────────────────────────────────
    all_examples = load_triplets_from_benchmark(
        cfg.data.benchmark_dir,
        task_filter=cfg.data.get("task_filter"),
        min_quality=cfg.data.min_quality,
        use_criterion_description=cfg.data.use_criterion_description,
    )
    random.shuffle(all_examples)

    n = len(all_examples)
    train_end = int(cfg.data.train_ratio * n)
    val_end = int((cfg.data.train_ratio + cfg.data.val_ratio) * n)
    train_set = all_examples[:train_end]
    val_set = all_examples[train_end:val_end]
    test_set = all_examples[val_end:]
    print(f"Data: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    # ── learning curve tracker ────────────────────────────────────────────
    rollout_budget = cfg.gepa.rollout_budget
    learning_curve = LearningCurveTracker(
        framework="dspy_gepa",
        benchmark="triplet_similarity",
        total_budget=rollout_budget,
    )

    # ── baseline ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    program = CheckTripletSimple(embedder=embedder)

    evaluate = dspy.Evaluate(
        devset=val_set,
        metric=gepa_metric,
        display_table=True,
        num_threads=1,
    )
    baseline_result = evaluate(program)
    baseline_score = extract_score(baseline_result)
    print(f"Baseline: {baseline_score:.4f} ({baseline_score * 100:.1f}%)")

    learning_curve.record(
        rollout_count=0, performance=baseline_score, checkpoint_pct=0.0
    )

    # ── GEPA optimisation ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("GEPA OPTIMIZATION")
    print("=" * 80)
    print(f"Budget: {rollout_budget} metric calls")

    reflection_lm = dspy.LM(cfg.gepa.reflection_lm_model)
    prompt_log_file, original_forward, prompt_log_handle = (
        setup_proposal_prompt_logging(reflection_lm, output_dir)
    )

    try:
        gepa = dspy.GEPA(
            metric=gepa_metric,
            reflection_lm=reflection_lm,
            track_stats=True,
            max_metric_calls=rollout_budget,
            reflection_minibatch_size=cfg.gepa.reflection_minibatch_size,
        )
        optimized = gepa.compile(program, trainset=train_set, valset=val_set)
    finally:
        reflection_lm.forward = original_forward
        prompt_log_handle.close()
        print(f"Saved proposal prompts to: {prompt_log_file}")

    print("Optimization complete")

    # ── final evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    val_result = evaluate(optimized)
    val_score = extract_score(val_result)

    improvement = val_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
    print(f"Final val score: {val_score:.4f} ({val_score * 100:.1f}%)")
    print(f"Improvement: {improvement:+.4f} ({improvement_pct:+.1f}% relative)")

    total_time = time.time() - start_time
    learning_curve.record(
        rollout_count=rollout_budget, performance=val_score, checkpoint_pct=1.0
    )

    # ── save results ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    learning_curve.save(output_dir)

    detailed_results_file = save_detailed_results(
        optimized_module=optimized,
        output_dir=output_dir,
        baseline_score=baseline_score,
        val_score=val_score,
        total_time=total_time,
        rollout_budget=rollout_budget,
        val_n=len(val_set),
        train_n=len(train_set),
    )

    with open(detailed_results_file) as f:
        detailed_results = json.load(f)

    prompt_details = save_prompt_details(
        optimized_module=optimized,
        output_dir=output_dir,
        trainset=train_set,
    )

    readout_file = save_readout_file(
        output_dir=output_dir,
        baseline_score=baseline_score,
        val_score=val_score,
        total_time=total_time,
        rollout_budget=rollout_budget,
        train_n=len(train_set),
        val_n=len(val_set),
        detailed_results=detailed_results,
        prompt_details=prompt_details,
        log_file=log_file,
        prompt_log_file=prompt_log_file,
    )

    stats = {
        "total_time": total_time,
        "total_rollouts": rollout_budget,
        "actual_rollouts": detailed_results.get("actual_rollouts", rollout_budget),
        "baseline_score": float(baseline_score),
        "val_score": val_score,
        "val_n": len(val_set),
        "train_n": len(train_set),
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults directory: {output_dir}")
    print(f"  Detailed results: {detailed_results_file.name}")
    print("  Optimized prompt: optimized_prompt.txt")
    print(f"  Readout: {readout_file.name}")


if __name__ == "__main__":
    main()
