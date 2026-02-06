# BM25 Visualization Fix - Part 2: Preventing Invalid Visualization Modes

## Problem

Building on the previous BM25 fix which enabled heatmap visualization for similarity matrices, there was still an issue:

**BM25 was showing PCA/t-SNE/UMAP visualization options in the web visualizer**, even though these dimensionality reduction methods don't make sense for NxN similarity matrices (they only work on NxD embeddings where D << N).

## Root Cause

1. **No format detection before visualization**: The visualization pipeline didn't check whether a `.npy` file contained:
   - NxD embeddings (suitable for PCA/t-SNE/UMAP)
   - NxN similarity matrix (only suitable for heatmap)

2. **All reducers applied to all methods**: The auto-visualization code tried to generate PCA/t-SNE/UMAP visualizations for ALL methods, including BM25

3. **Heatmap not in default reducers**: The default reducer list was `["tsne", "umap", "pca", "som", "dendrogram"]` - missing "heatmap"!

## Solution

### 1. Detect Similarity Matrices Early (visualize_corpus.py:338-344)

Added shape validation when loading embeddings from benchmark:

```python
if embeddings.ndim == 2 and embeddings.shape[0] == embeddings.shape[1] == len(documents):
    raise ValueError(
        f"Detected NxN similarity matrix ({embeddings.shape[0]}x{embeddings.shape[1]}) for {method_name}. "
        f"Similarity matrices cannot be visualized with dimensionality reduction methods (PCA/TSNE/UMAP). "
        f"Only heatmap visualization is supported for this method."
    )
```

This prevents the pipeline from even attempting to run PCA/t-SNE on similarity matrices.

### 2. Filter Reducers for BM25 Methods (visualization_utils.py)

Added a new helper function to identify similarity matrix methods:

```python
def is_similarity_matrix_method(method_name: str) -> bool:
    """Check if a method produces similarity matrices rather than embeddings."""
    similarity_matrix_patterns = ["bm25", "lexical"]
    method_lower = method_name.lower()
    return any(pattern in method_lower for pattern in similarity_matrix_patterns)
```

Updated `run_visualization()` to skip non-heatmap reducers for BM25:

```python
# Check if this is a similarity matrix method (like BM25)
if is_similarity_matrix_method(method_name):
    if reducer != "heatmap":
        # Silently skip non-heatmap reducers for similarity matrix methods
        return None
```

### 3. Add Heatmap to Default Reducers (visualization_utils.py:404)

```python
if reducers is None:
    reducers = ["tsne", "umap", "pca", "som", "dendrogram", "heatmap"]
```

Now heatmap is generated for ALL methods by default, not just BM25.

### 4. Support Heatmap Without Coords (visualize_corpus.py)

Heatmap doesn't need 2D coordinates (it works directly from embeddings/similarity matrix):

**Line 1560**: Allow web export when coords_2d is None for heatmap:
```python
if coords_2d is None and args.reducer != "heatmap":
    logger.error("No 2D coordinates available for web export")
```

**Lines 1046-1054**: Skip saving layout file for heatmap:
```python
if coords_2d is not None:
    layout_path = output_dir / f"layout_{reducer_name}.npy"
    np.save(layout_path, coords_2d)
    manifest["layouts"][reducer_name] = f"layout_{reducer_name}.npy"
elif reducer_name == "heatmap":
    # Heatmap works directly from embeddings, just add to manifest
    manifest["layouts"]["heatmap"] = None  # No layout file needed
```

## Result

### Before Fix
- BM25 showed all visualization modes: PCA, t-SNE, UMAP, SOM, Dendrogram, Heatmap
- Clicking PCA/t-SNE would fail or produce meaningless results
- Confusing user experience

### After Fix
- BM25 only shows "HEATMAP" button
- PCA/t-SNE/UMAP/SOM/Dendrogram buttons are hidden
- Clear indication that only heatmap is available for this method
- Heatmap works correctly with the NxN similarity matrix

### For Other Methods
- Still get all visualization modes (PCA, t-SNE, UMAP, SOM, Dendrogram, Heatmap)
- Heatmap now available for embedding methods too (computes cosine similarity on-the-fly)
- No regression in functionality

## Files Modified

1. **scripts/visualize_corpus.py**:
   - Added NxN matrix detection (raises error if trying to reduce)
   - Allow heatmap export without coords_2d
   - Handle heatmap in manifest generation

2. **src/multiview/utils/visualization_utils.py**:
   - Added `is_similarity_matrix_method()` helper
   - Filter reducers in `run_visualization()` for BM25
   - Added "heatmap" to default reducers list

## Testing

To verify the fix:

1. Generate visualizations for a benchmark with BM25:
   ```bash
   uv run python scripts/run_eval.py --config-name benchmark_fuzzy
   ```

2. Check manifest for BM25 method:
   ```bash
   cat viz/benchmark_fuzzy/*/bm25*/manifest.json
   ```
   Should show:
   ```json
   {
     "layouts": {
       "heatmap": null
     }
   }
   ```
   (Only heatmap, no tsne/pca/umap)

3. Open web visualizer and select BM25 method:
   - Should only show "HEATMAP" button
   - PCA/t-SNE/UMAP buttons should be hidden

4. For embedding methods (e.g., qwen3):
   - Should show all buttons: PCA, t-SNE, UMAP, SOM, Dendrogram, Heatmap
   - All modes should work correctly

## Implementation Notes

### Why Pattern Matching Instead of File Shape?

We use pattern matching (`is_similarity_matrix_method()`) rather than checking file shape because:
1. **Early filtering**: Skip visualization attempts before even loading the file
2. **Performance**: Don't need to load large NxN matrices just to skip them
3. **Clear intent**: Method names clearly indicate their output format

### Why Raise Error vs Return None?

When we detect an NxN matrix in `load_from_benchmark()`, we raise an error rather than returning None because:
1. **Clear failure mode**: Error message explains exactly what's wrong
2. **Prevents silent failures**: Caller knows immediately that visualization isn't possible
3. **Better debugging**: Stack trace shows where detection happened

The `run_visualization()` function catches this error and handles it appropriately (by skipping that visualization).
