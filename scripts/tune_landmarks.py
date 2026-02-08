#!/usr/bin/env python
"""Entry point for landmark GEPA prompt tuning.

Optimises a query-generation prompt so that the produced queries yield
embeddings with better triplet agreement.

Usage:
    python scripts/tune_landmarks.py                                # defaults
    python scripts/tune_landmarks.py data.benchmark_dir=outputs/benchmark_haiku \\
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

from multiview.tuning.landmark_gepa.metric import gepa_metric
from multiview.tuning.landmark_gepa.module import QueryGeneratorModule
from multiview.tuning.utils import (
    LearningCurveTracker,
    extract_score,
    save_detailed_results,
    save_readout_file,
    setup_logging,
    setup_proposal_prompt_logging,
)

logger = logging.getLogger(__name__)


def _collect_per_task_data(
    benchmark_dir: str | Path,
    task_filter: str | None = None,
) -> dict[str, dict]:
    """Read documents and triplets from each task directory, keyed by criterion.

    Returns a dict mapping ``criterion`` → ``{"documents": [...], "triplet_ids": [...]}``
    so that each criterion's queries are only scored against its own triplets.
    """
    benchmark_dir = Path(benchmark_dir)
    triplets_root = benchmark_dir / "triplets"

    per_task: dict[str, dict] = {}

    for task_dir in sorted(triplets_root.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter and task_filter not in task_dir.name:
            continue

        triplets_file = task_dir / "triplets.json"
        config_file = task_dir / "triplet_config.json"
        if not triplets_file.exists():
            continue

        with open(triplets_file) as f:
            triplets = json.load(f)

        # Read criterion from config
        criterion = task_dir.name
        if config_file.exists():
            with open(config_file) as f:
                cfg_data = json.load(f)
            criterion = cfg_data.get("criterion", task_dir.name)

        # Use task dir name as dict key (unique per directory)
        task_key = task_dir.name

        # Build a flat document list + triplet indices for this task
        doc_list: list[str] = []
        local_docs: dict[int, int] = {}
        triplet_ids: list[tuple[int, int, int]] = []

        _ID_TO_TEXT_KEY = {
            "anchor_id": "anchor",
            "positive_id": "positive",
            "negative_id": "negative",
        }
        for t in triplets:
            for id_key, text_key in _ID_TO_TEXT_KEY.items():
                local_id = t[id_key]
                if local_id not in local_docs:
                    doc = t[text_key]
                    text = (
                        doc.get("text", str(doc)) if isinstance(doc, dict) else str(doc)
                    )
                    local_docs[local_id] = len(doc_list)
                    doc_list.append(text)

            triplet_ids.append(
                (
                    local_docs[t["anchor_id"]],
                    local_docs[t["positive_id"]],
                    local_docs[t["negative_id"]],
                )
            )

        per_task[task_key] = {
            "criterion": criterion,
            "documents": doc_list,
            "triplet_ids": triplet_ids,
        }

    return per_task


def _resolve_criteria_str(
    criterion: str, all_criteria_meta: dict, use_description: bool
) -> str:
    """Build a criteria string, optionally appending its description."""
    if not use_description:
        return criterion
    for _ds, crit_map in all_criteria_meta.items():
        if criterion in crit_map:
            desc = crit_map[criterion].get("description", "")
            if desc:
                return f"{criterion}: {desc}"
    return criterion


@hydra.main(config_path="../configs", config_name="tune_landmarks", version_base=None)
def main(cfg: DictConfig) -> None:
    start_time = time.time()
    random.seed(cfg.seed)

    # ── output dir ────────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir) / f"gepa_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    log_file = setup_logging(output_dir)
    print(f"Verbose logs: {log_file}")

    # ── DSPy LM ──────────────────────────────────────────────────────────
    lm = dspy.LM(cfg.lm.model)
    dspy.configure(lm=lm)

    # ── load data ─────────────────────────────────────────────────────────
    from multiview.docsets.criteria_metadata import load_criteria_metadata
    from multiview.inference.inference import run_inference

    per_task = _collect_per_task_data(
        cfg.data.benchmark_dir,
        task_filter=cfg.data.get("task_filter"),
    )
    print(f"Collected {len(per_task)} criteria from benchmark")

    all_criteria_meta = load_criteria_metadata()
    per_criterion = cfg.data.get("per_criterion_triplets", True)

    if per_criterion:
        # Each criterion scored only against its own triplets (cleaner signal).
        print("Mode: per-criterion triplets")
        print("Pre-computing document embeddings per criterion...")

        dspy_examples: list[dspy.Example] = []
        for task_key, task_data in sorted(per_task.items()):
            criterion = task_data["criterion"]
            documents = task_data["documents"]
            triplet_ids = task_data["triplet_ids"]

            doc_embeddings = run_inference(
                inputs={"document": documents},
                config=cfg.embedding_preset,
                cache_alias=f"tune_landmarks_{task_key}",
                verbose=False,
            )
            print(
                f"  {task_key}: {len(documents)} docs, "
                f"{len(triplet_ids)} triplets, {len(doc_embeddings)} embeddings"
            )

            criteria_str = _resolve_criteria_str(
                criterion, all_criteria_meta, cfg.data.use_criterion_description
            )
            dspy_examples.append(
                dspy.Example(
                    {
                        "criteria": criteria_str,
                        "domain": cfg.domain,
                        "num_queries": cfg.num_queries,
                        "documents": documents,
                        "triplet_ids": triplet_ids,
                        "doc_embeddings": doc_embeddings,
                        "embedding_preset": cfg.embedding_preset,
                    }
                ).with_inputs("criteria", "domain", "num_queries")
            )
    else:
        # All criteria share merged documents + triplets (cross-criteria signal).
        print("Mode: shared triplets (cross-criteria)")
        all_docs: list[str] = []
        all_triplet_ids: list[tuple[int, int, int]] = []
        doc_offset = 0
        for task_data in per_task.values():
            for tid in task_data["triplet_ids"]:
                all_triplet_ids.append(
                    (tid[0] + doc_offset, tid[1] + doc_offset, tid[2] + doc_offset)
                )
            all_docs.extend(task_data["documents"])
            doc_offset += len(task_data["documents"])

        print(f"Merged: {len(all_docs)} docs, {len(all_triplet_ids)} triplets")
        print("Pre-computing document embeddings...")

        doc_embeddings = run_inference(
            inputs={"document": all_docs},
            config=cfg.embedding_preset,
            cache_alias="tune_landmarks_docs",
            verbose=False,
        )
        print(f"Computed {len(doc_embeddings)} document embeddings")

        dspy_examples = []
        for _task_key, task_data in sorted(per_task.items()):
            criterion = task_data["criterion"]
            criteria_str = _resolve_criteria_str(
                criterion, all_criteria_meta, cfg.data.use_criterion_description
            )
            dspy_examples.append(
                dspy.Example(
                    {
                        "criteria": criteria_str,
                        "domain": cfg.domain,
                        "num_queries": cfg.num_queries,
                        "documents": all_docs,
                        "triplet_ids": all_triplet_ids,
                        "doc_embeddings": doc_embeddings,
                        "embedding_preset": cfg.embedding_preset,
                    }
                ).with_inputs("criteria", "domain", "num_queries")
            )

    random.shuffle(dspy_examples)
    n = len(dspy_examples)
    train_end = int(cfg.data.train_ratio * n)
    val_end = int((cfg.data.train_ratio + cfg.data.val_ratio) * n)
    train_set = dspy_examples[:train_end]
    val_set = dspy_examples[train_end:val_end]
    test_set = dspy_examples[val_end:]

    # Ensure we have at least 1 example in each split
    if not val_set and test_set:
        val_set = [test_set.pop(0)]
    if not val_set and train_set:
        val_set = [train_set.pop()]
    print(f"Data: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")

    # ── learning curve tracker ────────────────────────────────────────────
    rollout_budget = cfg.gepa.rollout_budget
    learning_curve = LearningCurveTracker(
        framework="dspy_gepa",
        benchmark="landmark_query",
        total_budget=rollout_budget,
    )

    # ── baseline ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION")
    print("=" * 80)
    program = QueryGeneratorModule()

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

    # For landmark, save prompt details from the generate predictor
    prompt_details: dict = {}
    if hasattr(optimized, "generate") and hasattr(optimized.generate, "predict"):
        predictor = optimized.generate.predict
        if hasattr(predictor, "signature") and hasattr(
            predictor.signature, "instructions"
        ):
            prompt_details["instructions"] = str(predictor.signature.instructions)
        if hasattr(predictor, "demos"):
            prompt_details["num_demos"] = len(predictor.demos)

    # Save optimized prompt text
    with open(output_dir / "optimized_prompt.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DSPy GEPA Optimized Query Generation Prompt\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Instructions:\n{prompt_details.get('instructions', 'None')}\n\n")
        f.write(f"Number of Few-Shot Examples: {prompt_details.get('num_demos', 0)}\n")

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

    # Save the optimized module for later loading
    optimized.save(str(output_dir / "optimized_module"))

    print(f"\nResults directory: {output_dir}")
    print(f"  Detailed results: {detailed_results_file.name}")
    print("  Optimized prompt: optimized_prompt.txt")
    print(f"  Readout: {readout_file.name}")


if __name__ == "__main__":
    main()
