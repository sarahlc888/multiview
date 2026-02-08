# Taxonomy Detail vs Performance

## Thesis

More detailed taxonomies improve pseudologit (and in-one-word) accuracy on triplet evaluation. If you add more compute for the taxonomy, the downstream embeddings sharpen.

This is a Phase 1 finding (ROADMAP: "Systematic method comparison") — proving schema sensitivity is core to the multiview story.

## Goal

Polish into a clear GitHub showcase. One compelling GSM8K example with a bar chart, screenshot it for the README, then move on.

## Key findings to support

- [ ] Different taxonomies lead to different performance.
- [ ] More detailed taxonomies improve pseudologit performance.

## What's built (infrastructure)

### Schema comparison framework

Config: `configs/benchmark_schema_comparison.yaml`
Run: `uv run python -m multiview.benchmark.run configs/benchmark_schema_comparison.yaml`

Four schema modes implemented:

| Mode | Description | Config key |
|------|-------------|------------|
| No schema | Generic "categorize in one word", no categories given | `category_context: "Question: Categorize this text in one word."` |
| Oracle schema | Ground-truth schema from annotation phase | omit `category_context` (auto-extracted from `task.document_annotations`) |
| Proposed schema | Freshly LLM-generated schema, oracle withheld | `generate_schema: true` |
| Custom schema | Hand-specified categories for ablation | `category_context: "Categories: a, b, c\n..."` or `classes_file:` |

Multi-trial support: `num_trials: N` produces N rows (`method_trial1`, `method_trial2`, ...) with independent schema generations per trial. Statistics via `compute_trial_statistics()` or pandas.

### Key code locations

- **Schema generation:** `src/multiview/benchmark/annotations/class_schema.py` — `generate_category_schema()` samples N docs, uses LLM to produce categories
- **Tag schema (lm_tags):** `src/multiview/benchmark/annotations/tag_schema.py` — `generate_tag_schema()` for binary tags
- **Evaluation integration:** `src/multiview/benchmark/evaluation_utils.py` ~lines 657-758 — in_one_word and pseudologit with schema modes, trial loop, cache key differentiation
- **Trial statistics:** `multiview.benchmark.evaluation_utils.compute_trial_statistics()`
- **Docs:** `writeup/docs/SCHEMA_COMPARISON.md` (comprehensive reference)

### Current config state

`benchmark_schema_comparison.yaml` (`run_name: "benchmark_fuzzy_debug2"`):
- **Active:** pseudologit_proposed (3 trials), qwen3_8b embeddings (with/without instructions), bm25
- **Commented out:** in_one_word (no schema, oracle, proposed), pseudologit_oracle, custom schemas, query_relevance_vectors, document_rewrite
- **Task:** GSM8K / `final_expression` / `lm_tags` style
- **Scale:** 20 docs, 5 triplets (debug size — production would be 200 docs / 50 triplets)

### Existing results

Benchmark outputs exist across several runs but **no taxonomy granularity comparison has been collected yet**:
- `outputs/benchmark_fuzzy_debug_backup/`
- `outputs/benchmark_discrete/`
- `outputs/gsm8k_arithmetic_256_4methods/`

Results format: `{ task_name: { method_name: { accuracy, n_correct, n_incorrect, n_ties } } }`

## What's NOT done (the actual experiment)

- [ ] **Design the granularity ablation.** Define 3+ taxonomy detail levels for GSM8K (e.g., coarse: 4 categories, medium: 8, detailed: 12+). Can use custom schema mode with hand-specified categories, or generate at different `n_schema_samples` settings.
- [ ] **Run the comparison.** Uncomment relevant methods in config, scale up to production size, collect results for each granularity level.
- [ ] **Make the bar chart.** No bar chart / method comparison visualization exists yet. The existing viz infrastructure (`visualization_utils.py`, `corpus_viz.py`) is scatter/heatmap/dendrogram — need a new simple bar chart (accuracy by taxonomy detail, grouped by method). Matplotlib or similar.
- [ ] **Screenshot for README.** One clear chart showing the taxonomy granularity vs accuracy trend.
- [ ] **Link configs for reproducibility.** Ensure the config that produced the chart is checked in and referenced.

## Exit criteria

- [ ] One chart clearly shows pseudologit performance by taxonomy detail level (coarse / medium / detailed).
- [ ] Chart is in the README or easily reachable from it.
- [ ] Supporting config is committed and runnable.
