# TODO: No-Nonsense Zappos Viz Ship

## Goal
Ship the x-by-y embedding + visualization demo ASAP from `npm run dev`, using Gemini document rewriting (`<=1 sentence`) and embedding rewritten text only.

## Scope Lock
- [ ] Use `ut_zappos50k` only for prototype.
- [ ] Zero-shot methods only (no training/fine-tuning work).
- [ ] Keep UI scope minimal: reducer toggle (`som`, `dendrogram`, plus existing 2D), thumbnail-or-point rendering, hover text.

## Implementation Checklist
- [ ] Enforce one-sentence rewrite prompt in `src/multiview/prompts/eval/document_rewrite_summary.txt`.
- [ ] Create ship config `configs/benchmark_zappos_viz_ship.yaml` from `configs/benchmark_fuzzy_debug2.yaml`.
- [ ] In `configs/benchmark_zappos_viz_ship.yaml`, keep only Zappos tasks and set at least 3 criteria/views.
- [ ] In `configs/benchmark_zappos_viz_ship.yaml`, keep only methods needed for ship (`document_rewrite` + baseline embedding comparator).
- [ ] In `configs/benchmark_zappos_viz_ship.yaml`, set reducers to include `som` and `dendrogram`.
- [ ] In `configs/benchmark_zappos_viz_ship.yaml`, use `visualization.output_dir: "viz"` for direct dashboard pickup.

## Commands
- [ ] Set API keys:
```bash
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
```
- [ ] Create triplets/artifacts:
```bash
uv run python scripts/create_eval.py --config-name benchmark_zappos_viz_ship
```
- [ ] Run eval + auto-generate viz:
```bash
uv run python scripts/run_eval.py --config-name benchmark_zappos_viz_ship
```
- [ ] Start dashboard:
```bash
cd visualizer
npm install
npm run dev
```

## Verification (Done Criteria)
- [ ] Dashboard loads run data and supports reducer toggle including `som` and `dendrogram`.
- [ ] At least 3 Zappos criteria/views are present and explorable.
- [ ] Hover shows text for each item (rewritten sentence and/or source snippet).
- [ ] Thumbnails render for Zappos (point fallback acceptable only when image missing).
- [ ] Rewrites are `<=1 sentence`.
- [ ] Embeddings for document-rewriter path are computed from rewritten text only.

## Explicit Non-Goals
- [ ] No model training.
- [ ] No report polish.
- [ ] No extra dashboard features outside core toggle + hover + render path.
