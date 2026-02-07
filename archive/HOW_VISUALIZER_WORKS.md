# How the Visualizer Works

## Purpose

The visualizer lets you inspect benchmark methods across tasks and criteria using these views:
- Scatter layouts (`pca`, `tsne`, `umap`, `som`)
- Hierarchical view (`dendrogram`)
- Similarity matrix view (`heatmap`)
- Force graph (`graph`)

Supported method output types:
- Embeddings (`N x D`)
- Precomputed similarity matrices (`N x N`, e.g., BM25)

## Data Flow

1. Methods run in benchmark evaluation and emit embeddings or similarity matrices.
2. Export writes artifacts to `viz/<experiment>/<task>/<method>/`.
3. Each method directory includes `manifest.json` and mode artifacts.
4. `viz/index.json` indexes experiments, datasets, criteria, methods, and modes.
5. The web app loads index + manifest and renders the selected mode.

## Exported Artifacts

Common:
- `embeddings.npy` (dense embeddings) or `N x N` similarity matrix in that slot
- `manifest.json` (metadata + available layouts)

Mode-specific:
- Scatter modes: `layout_<mode>.npy`
- Dendrogram: `linkage_matrix.npy`
- Heatmap: no layout file required
- Graph: no layout file required

## Mode Behavior

Scatter (`pca`, `tsne`, `umap`, `som`):
- Uses precomputed 2D coordinates.

Dendrogram:
- Uses saved linkage matrix to build hierarchical tree structure.

Heatmap:
- `N x D` input: computes cosine similarity in browser.
- `N x N` input: uses matrix directly as pairwise similarities.
- Does not need coordinates.

Graph:
- Builds force-directed layout in browser from embeddings/similarity.
- Does not need a precomputed layout.

## BM25/Matrix Methods

BM25-like methods produce `N x N` matrices.

Current behavior:
- Matrix is persisted as visualization artifact.
- Non-heatmap reducers are skipped for matrix methods.
- UI exposes `heatmap` only for these methods.

## Triplet-Aware Graph

If triplets are present:
- Graph shows triplet navigation UI.
- Anchor/positive/negative nodes are highlighted with role colors.
- Sidebar includes triplet text and metadata (margin/correctness/quality fields).

## Leaderboard and Task Binding

Correct leaderboard display depends on consistent task identity across:
- `manifest.json`
- `viz/index.json`
- `results.json`

Task names with suffixes (style/count variants) must stay aligned end-to-end.

## Constraints

- Similarity matrices are not valid inputs for PCA/t-SNE/UMAP/SOM in this pipeline.
- Older exports may miss newer artifacts (for example `linkage_matrix.npy`), so re-export if needed.
