# Multiview Visualizer

Interactive web viewer for multiview corpus visualizations.

## Quick Start

### 1. Generate visualization data

```bash
# Single export (circles)
uv run python scripts/analyze_corpus.py \
  --dataset gsm8k \
  --embedding-preset hf_qwen3_embedding_8b \
  --criterion arithmetic \
  --reducer tsne \
  --export-format web \
  --output viz/gsm8k/arithmetic \
  --max-docs 100

# With thumbnails (images instead of circles)
uv run python scripts/analyze_corpus.py \
  --dataset gsm8k \
  --embedding-preset hf_qwen3_embedding_8b \
  --criterion arithmetic \
  --reducer tsne \
  --marker-type thumbnail \
  --export-format web \
  --output viz/gsm8k/arithmetic \
  --max-docs 100
```

### 2. Start the visualizer

```bash
cd visualizer
npm install
npm run dev
```

The visualizer will open at `http://localhost:5173`.

## Features

- **Multiple visualization modes**: t-SNE, SOM, UMAP, PCA, Dendrogram
- **Interactive controls**:
  - Select dataset from dropdown
  - Switch between criteria
  - Toggle visualization modes instantly (pre-computed)
- **Interactive visualization**:
  - Pan and zoom
  - Hover over points to see document text
  - Drag nodes (force-directed graph)
  - **Thumbnail support**: Shows images instead of circles when available
    - GSM8K: Computational graph thumbnails
    - Automatically scales and highlights on hover

## Directory Structure

```
visualizer/
  src/
    App.tsx              # Main application component
    render/
      ModeRenderer.tsx   # Mode switcher
      ScatterPlot.tsx    # t-SNE/SOM/UMAP/PCA renderer
      DendrogramView.tsx # Dendrogram renderer
      ForceDirectedGraph.tsx  # Force-directed graph renderer
    types/
      manifest.ts        # Type definitions
    utils/
      dataLoader.ts      # Data loading utilities
      npyLoader.ts       # NumPy .npy file loader
  scripts/
    build-dataset-index.cjs  # Dataset index builder

viz/                     # Visualization data (generated)
  {dataset}/
    {criterion}/
      manifest.json      # Metadata
      documents.txt      # Document texts
      embeddings.npy     # Embeddings
      layout_tsne.npy    # t-SNE coordinates
      layout_som.npy     # SOM coordinates
      ...
  index.json            # Dataset index
```

## Data Format

Each visualization export contains:

- `manifest.json`: Metadata (dataset, criterion, available modes, thumbnail paths)
- `documents.txt`: One document per line (newlines escaped as `\n`)
- `embeddings.npy`: NumPy array of embeddings (shape: `[n_docs, embedding_dim]`)
- `layout_{mode}.npy`: 2D coordinates for each mode (shape: `[n_docs, 2]`)
- `linkage_matrix.npy`: (optional) Linkage matrix for dendrogram
- `thumbnails/`: (optional) Thumbnail images (PNG files)

## Scripts

- `npm run dev`: Start development server (auto-builds index)
- `npm run build`: Build for production
- `npm run preview`: Preview production build
- `npm run index`: Manually rebuild dataset index

## Tips

- Export data with `--max-docs 100-500` for quick testing
- Pre-compute all modes you want to view (instant switching)
- The visualizer reads from `viz/` directory (served as public files)
