"""Utilities for GEPA tuning: logging, learning curves, and result saving."""

from __future__ import annotations

import json
import logging
import time
import types
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LearningCurveTracker:
    """Simple learning curve tracker for GEPA optimization."""

    def __init__(self, framework: str, benchmark: str, total_budget: int):
        self.framework = framework
        self.benchmark = benchmark
        self.total_budget = total_budget
        self.checkpoints: list[dict] = []

    def record(self, rollout_count: int, performance: float, checkpoint_pct: float):
        self.checkpoints.append(
            {
                "rollout_count": rollout_count,
                "performance": performance,
                "checkpoint_pct": checkpoint_pct,
            }
        )

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{self.framework}_{self.benchmark}_learning_curve.json"
        filepath = output_dir / filename
        data = {
            "framework": self.framework,
            "benchmark": self.benchmark,
            "total_budget": self.total_budget,
            "checkpoints": self.checkpoints,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)


def setup_logging(output_dir: Path) -> Path:
    """Redirect DSPy verbose logs to file, keep terminal clean.

    Returns:
        Path to the log file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "dspy_gepa.log"

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    dspy_logger = logging.getLogger("dspy")
    dspy_logger.setLevel(logging.DEBUG)
    dspy_logger.addHandler(file_handler)
    dspy_logger.propagate = False

    for logger_name in ["dspy.teleprompt", "dspy.evaluate"]:
        sub_logger = logging.getLogger(logger_name)
        sub_logger.setLevel(logging.DEBUG)
        sub_logger.addHandler(file_handler)
        sub_logger.propagate = False

    return log_file


def setup_proposal_prompt_logging(
    reflection_lm: Any, output_dir: Path
) -> tuple[Path, Any, Any]:
    """Monkey-patch *reflection_lm* to capture GEPA proposal prompts.

    Returns:
        ``(log_file_path, original_forward_method, log_file_handle)``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_log_file = output_dir / "dspy_gepa_proposal_prompts.log"
    prompt_log_handle = open(prompt_log_file, "w")  # noqa: SIM115

    gepa_call_count = {"count": 0}
    original_forward = reflection_lm.forward

    def logged_forward(self, prompt=None, messages=None, **kwargs):
        gepa_call_count["count"] += 1

        prompt_log_handle.write("=" * 80 + "\n")
        prompt_log_handle.write(
            f"DSPy GEPA PROPOSAL PROMPT (Call #{gepa_call_count['count']})\n"
        )
        prompt_log_handle.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        prompt_log_handle.write("=" * 80 + "\n")

        if messages:
            prompt_log_handle.write(
                f"\n--- CONVERSATION PROMPT ({len(messages)} messages) ---\n"
            )
            for i, msg in enumerate(messages):
                prompt_log_handle.write(f"\n[Message {i + 1}]\n")
                if isinstance(msg, dict):
                    prompt_log_handle.write(f"Role: {msg.get('role', 'unknown')}\n")
                    content = msg.get("content", str(msg))
                    if isinstance(content, str) and len(content) > 50_000:
                        prompt_log_handle.write(
                            f"Content (truncated): {content[:50_000]}...\n"
                        )
                    else:
                        prompt_log_handle.write(f"Content: {content}\n")
                else:
                    prompt_log_handle.write(f"{msg}\n")
            prompt_log_handle.write("\n--- END CONVERSATION ---\n\n")
        elif prompt:
            prompt_log_handle.write("\n--- FULL PROMPT SENT TO LLM ---\n")
            if isinstance(prompt, str) and len(prompt) > 50_000:
                prompt_log_handle.write(f"{prompt[:50_000]}...\n")
            else:
                prompt_log_handle.write(f"{prompt}\n")
            prompt_log_handle.write("\n--- END PROMPT ---\n\n")
        else:
            prompt_log_handle.write("\n--- PROMPT (no prompt/messages provided) ---\n")
            prompt_log_handle.write(f"kwargs: {kwargs}\n")
            prompt_log_handle.write("\n--- END PROMPT ---\n\n")

        prompt_log_handle.flush()

        result = original_forward(prompt=prompt, messages=messages, **kwargs)

        prompt_log_handle.write("--- LLM RESPONSE ---\n")
        if hasattr(result, "choices") and result.choices:
            for i, choice in enumerate(result.choices):
                prompt_log_handle.write(f"Choice {i + 1}:\n")
                if hasattr(choice, "message"):
                    if hasattr(choice.message, "content"):
                        prompt_log_handle.write(
                            f"  Content: {choice.message.content}\n"
                        )
                    if hasattr(choice.message, "tool_calls"):
                        prompt_log_handle.write(
                            f"  Tool Calls: {json.dumps([tc.model_dump() if hasattr(tc, 'model_dump') else str(tc) for tc in choice.message.tool_calls], indent=2, default=str)}\n"
                        )
                else:
                    prompt_log_handle.write(f"  {choice}\n")
        else:
            prompt_log_handle.write(f"{result}\n")
        prompt_log_handle.write("--- END RESPONSE ---\n\n")
        prompt_log_handle.flush()

        return result

    reflection_lm.forward = types.MethodType(logged_forward, reflection_lm)

    return prompt_log_file, original_forward, prompt_log_handle


def extract_score(eval_result: Any) -> float:
    """Extract score from DSPy evaluation result, normalised to 0-1."""
    if isinstance(eval_result, int | float):
        return float(eval_result) / 100.0 if eval_result > 1 else float(eval_result)
    elif isinstance(eval_result, dict):
        return eval_result.get("accuracy", eval_result.get("score", 0.0))
    elif hasattr(eval_result, "score"):
        return float(eval_result.score) / 100.0
    return 0.0


def save_detailed_results(
    optimized_module: Any,
    output_dir: Path,
    baseline_score: float,
    val_score: float,
    total_time: float,
    rollout_budget: int,
    val_n: int,
    train_n: int,
) -> Path:
    """Save detailed GEPA optimisation results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_results_file = output_dir / "dspy_gepa_detailed_results.json"

    detailed_results: dict[str, Any] = {
        "best_score": val_score,
        "baseline_score": float(baseline_score),
        "total_rollouts": rollout_budget,
        "total_time": total_time,
        "val_n": val_n,
        "train_n": train_n,
        "candidates": [],
        "evolution": [],
    }

    if hasattr(optimized_module, "detailed_results"):
        gepa_results = optimized_module.detailed_results

        if (
            hasattr(gepa_results, "total_metric_calls")
            and gepa_results.total_metric_calls is not None
        ):
            detailed_results["actual_rollouts"] = gepa_results.total_metric_calls

        if hasattr(gepa_results, "log_dir") and gepa_results.log_dir is not None:
            detailed_results["log_dir"] = str(gepa_results.log_dir)
            detailed_results["note"] = (
                "Candidate programs and full optimization logs are saved in the log_dir"
            )

        if hasattr(gepa_results, "candidates") and hasattr(
            gepa_results, "val_aggregate_scores"
        ):
            for i, (candidate, score, discovery_count) in enumerate(
                zip(
                    gepa_results.candidates,
                    gepa_results.val_aggregate_scores,
                    getattr(
                        gepa_results,
                        "discovery_eval_counts",
                        [0] * len(gepa_results.candidates),
                    ),
                    strict=False,
                )
            ):
                candidate_info: dict[str, Any] = {
                    "candidate_num": i,
                    "score": float(score),
                    "discovery_rollout": discovery_count,
                    "is_best": i == getattr(gepa_results, "best_idx", 0),
                    "instructions": {},
                }

                if isinstance(candidate, dict):
                    for pred_name, instruction in candidate.items():
                        candidate_info["instructions"][pred_name] = str(instruction)
                elif hasattr(candidate, "named_predictors"):
                    for pred_name, predictor in candidate.named_predictors():
                        if hasattr(predictor, "signature") and hasattr(
                            predictor.signature, "instructions"
                        ):
                            candidate_info["instructions"][pred_name] = str(
                                predictor.signature.instructions
                            )

                detailed_results["candidates"].append(candidate_info)

        if hasattr(gepa_results, "parents"):
            for i, parent_list in enumerate(gepa_results.parents):
                detailed_results["evolution"].append(
                    {
                        "candidate_num": i,
                        "parents": parent_list if parent_list else [],
                    }
                )

    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    return detailed_results_file


def save_prompt_details(
    optimized_module: Any,
    output_dir: Path,
    trainset: list | None = None,
    predictor_attr: str = "rewrite_query",
) -> dict[str, Any]:
    """Extract and save prompt details from an optimised module.

    Works for any module with a *predictor_attr* (default ``rewrite_query``)
    that exposes ``predict.signature.instructions`` and ``predict.demos``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_details: dict[str, Any] = {}
    module_predictor = getattr(optimized_module, predictor_attr, None)
    if module_predictor is not None and hasattr(module_predictor, "predict"):
        predictor = module_predictor.predict
        if hasattr(predictor, "signature") and hasattr(
            predictor.signature, "instructions"
        ):
            prompt_details["instructions"] = str(predictor.signature.instructions)
        if hasattr(predictor, "demos"):
            demos = predictor.demos
            prompt_details["num_demos"] = len(demos)
            prompt_details["demo_examples"] = []
            for demo in demos[:5]:
                demo_dict: dict[str, str] = {}
                for attr in (
                    "input_text",
                    "aspect",
                    "aspect_of_input_text",
                    "rationale",
                ):
                    if hasattr(demo, attr):
                        demo_dict[attr] = str(getattr(demo, attr))
                prompt_details["demo_examples"].append(demo_dict)

    # Try a forward pass to capture the full rendered prompt
    try:
        if trainset and len(trainset) > 0 and module_predictor is not None:
            sample = trainset[0]
            sample_input = getattr(sample, "A", "test")
            sample_aspect = getattr(sample, "criteria", "test")
            sample_pred = module_predictor(
                input_text=sample_input, aspect=sample_aspect
            )
            prompt_details["sample_prediction"] = {
                "input_text": sample_input,
                "aspect": sample_aspect,
                "aspect_of_input_text": str(
                    getattr(sample_pred, "aspect_of_input_text", sample_pred)
                ),
                "rationale": str(sample_pred.rationale)
                if hasattr(sample_pred, "rationale")
                else None,
            }
    except Exception as e:
        prompt_details["sample_prediction_error"] = str(e)

    module_info = {
        "instructions": prompt_details.get("instructions"),
        "demos": prompt_details.get("num_demos", 0),
        "prompt_details": prompt_details,
    }
    with open(output_dir / "best_module.json", "w") as f:
        json.dump(module_info, f, indent=2)

    with open(output_dir / "optimized_prompt.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DSPy GEPA Optimized Prompt\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Instructions:\n{prompt_details.get('instructions', 'None')}\n\n")
        f.write(
            f"Number of Few-Shot Examples: {prompt_details.get('num_demos', 0)}\n\n"
        )
        if prompt_details.get("demo_examples"):
            f.write("Few-Shot Examples:\n")
            for i, demo in enumerate(prompt_details["demo_examples"], 1):
                f.write(f"\nExample {i}:\n")
                for key, val in demo.items():
                    f.write(f"  {key}: {val}\n")
        if prompt_details.get("sample_prediction"):
            f.write("\n" + "=" * 80 + "\n")
            f.write("Sample Prediction:\n")
            f.write("=" * 80 + "\n")
            sp = prompt_details["sample_prediction"]
            for key, val in sp.items():
                if val is not None:
                    f.write(f"{key}: {val}\n")

    return prompt_details


def save_readout_file(
    output_dir: Path,
    baseline_score: float,
    val_score: float,
    total_time: float,
    rollout_budget: int,
    train_n: int,
    val_n: int,
    detailed_results: dict[str, Any],
    prompt_details: dict[str, Any],
    log_file: Path,
    prompt_log_file: Path | None = None,
) -> Path:
    """Save comprehensive human-readable readout."""
    output_dir.mkdir(parents=True, exist_ok=True)
    readout_file = output_dir / "dspy_gepa_readout.txt"

    with open(readout_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DSPy GEPA OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write("Benchmark: Triplet Similarity\n")
        f.write("Framework: DSPy GEPA\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Rollout Budget: {rollout_budget}\n")
        f.write(f"Training Examples: {train_n}\n")
        f.write(f"Validation Examples: {val_n}\n")
        f.write("=" * 80 + "\n\n")

        # Summary
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Baseline Score: {baseline_score:.4f} ({baseline_score * 100:.1f}%)\n")
        f.write(f"Best Score:     {val_score:.4f} ({val_score * 100:.1f}%)\n")
        improvement = (
            ((val_score - baseline_score) / baseline_score) * 100
            if baseline_score > 0
            else 0
        )
        f.write(
            f"Improvement:    {improvement:+.1f}% relative "
            f"({(val_score - baseline_score) * 100:+.1f} pp absolute)\n"
        )
        f.write(f"Total Time:     {total_time:.1f}s ({total_time / 60:.1f}m)\n")
        f.write(
            f"Actual Rollouts: {detailed_results.get('actual_rollouts', rollout_budget)}\n"
        )
        f.write("\n")

        # Best prompt
        f.write("=" * 80 + "\n")
        f.write("BEST PROMPT\n")
        f.write("=" * 80 + "\n")
        if prompt_details.get("instructions"):
            f.write(f"\nInstructions:\n{prompt_details['instructions']}\n\n")
        f.write(f"Few-Shot Examples: {prompt_details.get('num_demos', 0)}\n")
        if prompt_details.get("demo_examples"):
            f.write("\nExample Few-Shot Demos:\n")
            for i, demo in enumerate(prompt_details["demo_examples"][:5], 1):
                f.write(f"\n  Example {i}:\n")
                for key, val in demo.items():
                    f.write(f"    {key}: {val}\n")
        f.write("\n")

        # All candidates
        if detailed_results.get("candidates"):
            f.write("=" * 80 + "\n")
            f.write(f"ALL CANDIDATES ({len(detailed_results['candidates'])})\n")
            f.write("=" * 80 + "\n\n")
            for cand in detailed_results["candidates"]:
                cand_num = cand.get("candidate_num", "?")
                score = cand.get("score", 0.0)
                discovery_rollout = cand.get("discovery_rollout", "?")
                is_best = cand.get("is_best", False)
                best_marker = " << BEST" if is_best else ""
                f.write(
                    f"[Candidate {cand_num}] Score: {score:.4f} | "
                    f"Discovery Rollout: {discovery_rollout}{best_marker}\n"
                )
                f.write("-" * 80 + "\n")
                instructions = cand.get("instructions", {})
                if instructions:
                    for pred_name, instr_text in instructions.items():
                        f.write(f"\n  {pred_name}:\n")
                        instr_str = str(instr_text)
                        if len(instr_str) > 500:
                            f.write(f"    {instr_str[:500]}...\n")
                            f.write("    [Truncated - full text in JSON file]\n")
                        else:
                            f.write(f"    {instr_str}\n")
                f.write("\n")

        # Evolution
        if detailed_results.get("evolution"):
            f.write("=" * 80 + "\n")
            f.write("CANDIDATE EVOLUTION (PARENT LINEAGE)\n")
            f.write("=" * 80 + "\n\n")
            for evo in detailed_results["evolution"]:
                cand_num = evo.get("candidate_num", "?")
                parents = evo.get("parents", [])
                f.write(f"Candidate {cand_num}: Parents = {parents}\n")
            f.write("\n")

        # File references
        f.write("=" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Detailed JSON results: {output_dir / 'dspy_gepa_detailed_results.json'}\n"
        )
        f.write(f"Optimized prompt: {output_dir / 'optimized_prompt.txt'}\n")
        f.write(f"Verbose log: {log_file}\n")
        if prompt_log_file and prompt_log_file.exists():
            f.write(f"Proposal prompts log: {prompt_log_file}\n")
        if detailed_results.get("log_dir"):
            f.write(f"Full optimization logs: {detailed_results['log_dir']}\n")
        f.write(
            f"Learning curve: {output_dir / 'dspy_gepa_triplet_similarity_learning_curve.json'}\n"
        )
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    return readout_file
