# Roadmap
## Done

- [x] Generated 256 triplets for `gsm8k/arithmetic_operations` and evaluated 4 methods.
- [x] Scaled to 10 docset-criteria combinations.
- [x] Implemented schema comparison framework (`configs/benchmark_schema_comparison.yaml`) with multi-trial support and trial statistics.
- [x] Shipped core embedding visualizer flows (2D reducers, leaderboard modes, triplet interaction; see `visualizer/NEW_FEATURES.md`).

## TODO
- [ ] No-Nonsense Viz Ship





### Evidence Milestone: Taxonomy Detail vs Performance

Key finding to support:
- [ ] Different taxonomies lead to different performance.
- [ ] More detailed taxonomies improve pseudologit performance.

Action items:
- [ ] Run/collect results for multiple pseudologit taxonomy settings (coarse -> medium -> detailed).
- [ ] Add a bar chart view comparing pseudologit variants by accuracy.
- [ ] Include a concise evidence note with the chart: taxonomy granularity vs accuracy trend.

Exit criteria:
- [ ] One chart in dashboard/report clearly shows pseudologit performance by taxonomy detail level.
- [ ] Supporting run metadata/configs are linked for reproducibility.

### Demo Milestone

- [ ] Build a "semantic filter" demo.
- [ ] Build an Anthropic interview transcript demo focused on value alignment.
- [ ] Define and scope one additional demo candidate (currently open question).

Exit criteria:
- [ ] Each demo has a runnable path and a short script for what it proves.

### Findings to ship
- If you add more compute for the taxonomy, the downstream embeddings sharpen for vecEOL and pseudologits

### Tuning

- [ ] Productionized fine-tuning pipeline for triplet-based training.
