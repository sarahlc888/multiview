# Task: Automatically t-SNE embeddings as a byproduct of eval

## Goal
When embeddings are computed during eval, automatically generate t-SNE visualizations as a QA/exploration tool.

## Type
OTHER (infrastructure)

## Notes from original task
- "we do get embeddings as a byproduct of a bunch of these, so automatically tsne those up?"
- Currently embeddings are saved as .npy files in outputs/

## Reference files
- `scripts/run_eval.py` — where embeddings get computed
- `src/multiview/utils/visualization_utils.py` — existing viz utilities
- `outputs/` — where .npy embedding files are saved

## Steps
- [ ] Read visualization_utils.py to see what t-SNE/UMAP support exists
- [ ] Add a post-eval hook or flag that generates 2D projections from embedding .npy files
- [ ] Save as HTML or PNG alongside the embedding files
- [ ] Consider adding this to the main benchmark pipeline as an optional step

## Exit criteria
- [ ] Running eval with a flag produces a t-SNE plot in outputs/
