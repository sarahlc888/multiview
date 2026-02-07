# Roadmap

## North Star

Build a reliable benchmark for criteria-specific semantic similarity, then train and validate embedding models that measurably improve triplet accuracy on held-out tasks.

## Current Status Snapshot

### Completed

- [x] Generated 256 triplets for `gsm8k/arithmetic_operations` and evaluated 4 methods.
- [x] Scaled to 10 docset-criteria combinations.
- [x] Implemented schema comparison framework (`configs/benchmark_schema_comparison.yaml`) with multi-trial support and trial statistics.
- [x] Shipped core embedding visualizer flows (2D reducers, leaderboard modes, triplet interaction; see `visualizer/NEW_FEATURES.md`).

## Immediate Milestone (Next 7-10 Days): No-Nonsense Viz Ship

Goal: Ship the coolest core capability ASAP from `npm run dev`: x-by-y style embedding + visualization for multiple views, with minimal polish work.

Top priority (explicit):
- [ ] vibe code the viz dashboard and stop working on fluff
- [ ] ship a version that uses zero shot methods ONLY to embed multiple different views of the Met dataset

### Scope (must ship)

- [ ] Build API document-rewriter path using Gemini:
  - Input: source document text
  - Output: summary constrained to <=1 sentence
  - Use rewritten sentence as embedding input (document rewriter setting)
- [ ] Prototype on `ut_zappos50k` first (toy/easy path), not Met.
- [ ] Zero-shot methods only for this ship (no fine-tuning work in milestone).
- [ ] Enable fast view toggling in dashboard:
  - Self-organizing map (SOM)
  - Dendrogram
  - Existing 2D view(s) already supported
- [ ] Render item as thumbnail (preferred) or point fallback.
- [ ] On hover, show associated text (rewritten sentence and/or source snippet).

### De-scope (do not spend time here)

- [ ] No training infrastructure changes.
- [ ] No long-form reporting or writeup polish.
- [ ] No advanced feature work beyond required toggles and hover behavior.

### Acceptance checks

- [ ] `npm run dev` demonstrates end-to-end flow on Zappos with at least 3 distinct criteria/views.
- [ ] Each item has a Gemini-generated <=1 sentence rewrite stored/available for embedding.
- [ ] Embeddings in this path are generated from rewritten text only.
- [ ] User can switch between SOM and dendrogram without leaving the page.
- [ ] Hover interaction reveals text and image/point context reliably.

### Next after ship

- [ ] Port the same zero-shot document-rewriter workflow from Zappos to Met dataset views.

## Next Milestones (After No-Nonsense Viz Ship)

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

## Findings to ship
- If you add more compute for the taxonomy, the downstream embeddings sharpen for vecEOL and pseudologits

### Open

- [ ] Productionized fine-tuning pipeline for triplet-based training.
