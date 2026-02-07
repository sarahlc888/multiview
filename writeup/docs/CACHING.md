# Caching

This project uses two distinct cache layers.

## 1) Inference Cache (LM/Embedding Calls)

Purpose:
- Avoid repeated model API/local inference calls for identical prompts + config.

Where it lives:
- Root cache dir: `src/multiview/constants.py` sets `INFERENCE_CACHE_DIR` to `/Users/sarahchen/code/.cache/inference_cache`.
- If `run_name` is set, cache files are placed under a run subdirectory.

How cache keys work:
- File-level key: `cache_alias + hashed inference config`.
- Entry-level key: hash of each packed prompt (`sha256`, truncated).
- Prompts are deduplicated before provider calls, then remapped back.

What is considered in the inference config hash:
- Provider/model and generation params (`temperature`, `max_tokens`)
- Prompt template/instruction/prefill template
- Parser + parser kwargs
- `cache_alias`

Concurrency behavior:
- Cache files use lock files + atomic write/replace.
- On save, cache is reloaded and merged to preserve concurrent writes.

Controls:
- Global: `USE_CACHE` env var (`true/1/yes` enables; otherwise disabled).
- Per-call: `force_refresh=True` bypasses cache for that call.

Observability:
- Cache hits are tracked via `multiview.inference.cost_tracker` as avoided requests/cost.

## 2) Triplet Artifact Reuse (Task-Level)

Purpose:
- Reuse previously generated triplets/doc artifacts for a task when generation config matches.

Where it lives:
- Under benchmark outputs, typically `outputs/<run_name>/triplets/<task_name>/`.
- Key files:
  - `triplets.json`
  - `triplet_config.json`

Validation logic:
- Reuse is allowed only when `triplet_config_matches(cached, current)` is true.
- Comparison includes triplet-generation-relevant fields (dataset, criterion, style, seed, hints, quality filters, etc.).

Controls:
- `reuse_cached_triplets` (default `true`):
  - `true`: try loading cached triplets first.
  - `false`: always regenerate triplets.
- `triplet_cache_dir`: explicit triplet cache directory.

Important distinction:
- `reuse_cached_triplets=false` only disables triplet artifact reuse.
- It does NOT disable inference cache for LM calls made during annotation/triplet generation/evaluation.

## Cache Alias Strategy

Method evaluation aliases are generated as:
- `<task_name>_eval_<method_name><hash_suffix>`

Notes:
- `hash_suffix` is added when custom method config changes behavior.
- Multi-trial methods inject a trial index into the hashed config to avoid cross-trial cache collisions.

## Practical Workflow

1. Generate artifacts:
- `uv run python scripts/create_eval.py run_name=<run_name>`

2. Evaluate on cached artifacts:
- `uv run python scripts/run_eval.py run_name=<run_name>`

3. If you want fresh model calls for debugging:
- Set `USE_CACHE=false` in environment, or use per-call `force_refresh=True` in code paths.

## Common Failure Modes

- Triplet cache not reused:
  - `triplet_config.json` missing, `triplets.json` missing, or config mismatch.
- Inference cache misses unexpectedly:
  - Prompt text changed, prompt packing changed, cache alias changed, or inference config changed.
- Stale/incorrect parse artifacts:
  - Re-run with `force_refresh=True` for affected method call.
