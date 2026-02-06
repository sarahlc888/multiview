# BM25 "Data Not Available" Fix

## Problem

After implementing the initial BM25 visualization fix, BM25 was showing "does not have data available" error in the visualizer.

## Root Cause

The previous fix had several issues:

1. **Error on load**: Code was raising a `ValueError` when loading NxN similarity matrices, even for heatmap visualization (which should work with them)

2. **Missing heatmap reducer**: The `create_reducer()` function didn't have a case for "heatmap", causing it to raise `ValueError("Unknown reducer: heatmap")`

3. **Reducer application**: Code tried to call `reducer.fit_transform()` even when reducer was None (heatmap case)

## Solution

### 1. Don't Error on NxN Matrix Load (visualize_corpus.py:340-350)

Changed from raising an error to just logging when an NxN similarity matrix is detected:

```python
is_similarity_matrix = (
    embeddings.ndim == 2 and
    embeddings.shape[0] == embeddings.shape[1] == len(documents)
)

if is_similarity_matrix:
    logger.info(
        f"Loaded NxN similarity matrix ({embeddings.shape[0]}x{embeddings.shape[1]}) for {method_name}. "
        f"This method supports heatmap visualization only."
    )
    # Don't raise error - heatmap visualization will handle this correctly
```

### 2. Handle Heatmap in create_reducer() (visualize_corpus.py:587-589)

Added explicit heatmap case that returns None:

```python
elif args.reducer == "heatmap":
    # Heatmap doesn't need a reducer - works directly from embeddings/similarity matrix
    return None
```

### 3. Skip Reduction for Heatmap (visualize_corpus.py:1560-1571)

Added check to skip dimensionality reduction when reducer is "heatmap":

```python
if args.reducer == "heatmap":
    if not quiet:
        logger.debug("Heatmap mode - skipping dimensionality reduction")
    coords_2d = None  # Heatmap doesn't need 2D coords
else:
    embeddings_arr = np.array(embeddings, dtype=np.float32)
    coords_2d = reducer.fit_transform(embeddings_arr)
```

### 4. Fixed Indentation Bug (visualize_corpus.py:354-372)

Fixed indentation in NaN filtering code block that was causing syntax errors.

## Complete Flow

### For BM25 + Heatmap:
1. ✅ Not skipped early (is_similarity_matrix_method but reducer == "heatmap")
2. ✅ Loads NxN similarity matrix successfully (no error)
3. ✅ create_reducer returns None
4. ✅ Skips dimensionality reduction (coords_2d stays None)
5. ✅ Export works with coords_2d = None for heatmap
6. ✅ Manifest saved with `"layouts": {"heatmap": null}`
7. ✅ Visualizer displays heatmap correctly

### For BM25 + PCA/t-SNE/UMAP:
1. ✅ Skipped early in run_visualization()
2. ✅ No manifest entry created
3. ✅ UI won't show these buttons

### For Embedding Methods + Heatmap:
1. ✅ Loads NxD embeddings
2. ✅ create_reducer returns None
3. ✅ Skips dimensionality reduction
4. ✅ Heatmap computes cosine similarity on-the-fly
5. ✅ Works correctly

### For Embedding Methods + PCA/t-SNE/UMAP:
1. ✅ Works as before with proper reducers
2. ✅ No regression

## Testing

Regenerate visualizations:

```bash
# If you have auto_visualize enabled in config, just re-run eval:
uv run python scripts/run_eval.py --config-name benchmark_fuzzy

# Or manually generate visualizations:
uv run python -c "from multiview.utils.visualization_utils import generate_visualizations_for_benchmark; generate_visualizations_for_benchmark('benchmark_fuzzy')"
```

Check results:
```bash
# Check BM25 manifest
cat viz/benchmark_fuzzy/*/bm25*/manifest.json

# Should show:
# {
#   "layouts": {
#     "heatmap": null
#   },
#   ...
# }

# Open visualizer
cd visualizer && npm run dev
# Select BM25 method - should only show HEATMAP button
# Click HEATMAP - should display correctly
```

## Files Modified

1. **scripts/visualize_corpus.py**:
   - Line 340-350: Don't raise error for NxN matrices
   - Line 354-372: Fixed indentation in NaN filtering
   - Line 587-589: Added heatmap case to create_reducer()
   - Line 1560-1571: Skip reduction for heatmap

2. **src/multiview/utils/visualization_utils.py**:
   - (Already fixed in previous iteration)

## Result

BM25 now works correctly:
- Only shows "HEATMAP" visualization option
- Heatmap loads and displays the similarity matrix correctly
- No "data not available" error
- No invalid PCA/t-SNE/UMAP options shown
