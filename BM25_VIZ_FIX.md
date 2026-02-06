# BM25 Visualization Fix

## Problem

BM25 was computing a full NxN similarity matrix but discarding it, only returning triplet-specific scores. This meant:
1. The similarity matrix wasn't saved to disk
2. The heatmap visualization mode couldn't work with BM25 methods
3. No way to visualize pairwise similarities for non-embedding methods

## Solution

Added support for saving and visualizing pre-computed similarity matrices (NxN) in addition to embeddings (NxD).

### Changes Made

#### 1. BM25 Evaluation (`src/multiview/eval/bm25.py`)
- **Line 85**: Return `similarity_matrix` in results dict
- Now returns both triplet scores AND full similarity matrix for visualization

#### 2. Document Rewrite Evaluation (`src/multiview/eval/document_summary.py`)
- **Lines 137, 144**: Track BM25 similarity matrix in `bm25_similarity_matrix` variable
- **Lines 241-254**: Return similarity matrix when using BM25 mode
- Properly expands matrix to full document count with NaN for unused docs

#### 3. Benchmark Orchestration (`src/multiview/benchmark/benchmark.py`)
- **Lines 55-86**: Added `save_similarity_matrix_to_npy()` function
  - Saves NxN matrices using same file naming as embeddings
  - Visualizer detects shape to determine how to process
- **Lines 177, 203**: Extract `similarity_matrix` from method results
- **Lines 186-194, 220-228**: Save similarity matrices to disk (multi-trial case)
- **Lines 233-241**: Save similarity matrices to disk (single result case)
- **Lines 206, 243**: Exclude `similarity_matrix` from in-memory results (keep lean)

#### 4. Heatmap Visualizer (`visualizer/src/render/HeatmapView.tsx`)
- **Lines 54-58**: Added `isPrecomputedSimilarityMatrix()` helper
  - Detects NxN shape (precomputed matrix) vs NxD (embeddings)
- **Lines 82-108**: Updated computation logic
  - Uses precomputed matrix directly for BM25
  - Computes cosine similarity for embeddings
  - Handles document limiting for both cases
- **Lines 162-165**: Updated title to distinguish matrix types
- **Lines 211-213**: Updated legend label

## How It Works

### For Embedding Methods (NxD shape)
1. Save embeddings as usual (NxD array where D = embedding dimension)
2. Visualizer loads embeddings
3. Heatmap computes cosine similarity on-the-fly
4. Shows "Cosine Similarity Matrix"

### For BM25/Sparse Methods (NxN shape)
1. Compute full NxN similarity matrix during evaluation
2. Save matrix to same location/format as embeddings
3. Visualizer detects NxN shape (where N = number of documents)
4. Heatmap uses precomputed matrix directly
5. Shows "Pairwise Similarity Matrix"

## Benefits

1. **Heatmap mode now works for BM25** - can visualize document similarities
2. **Memory efficient** - only stores once, not recomputed for visualization
3. **Consistent API** - similarity matrices saved alongside embeddings
4. **Automatic detection** - visualizer auto-detects format by array shape
5. **Extensible** - other sparse/non-embedding methods can use same pattern

## Testing

To test:
```bash
# Run benchmark with BM25 method (generates visualizations automatically)
uv run python scripts/run_eval.py --config configs/benchmark_fuzzy_debug.yaml

# Open visualizer and check heatmap mode for BM25 methods
```

The heatmap should:
- Load without errors for BM25 methods
- Show "Pairwise Similarity Matrix" title
- Display color-coded similarity values
- Support hover tooltips with document pairs

## Future Work

- Consider applying same pattern to reranker methods
- Could add explicit "matrix_type" metadata to manifest
- May want separate viz modes for sparse vs dense methods
