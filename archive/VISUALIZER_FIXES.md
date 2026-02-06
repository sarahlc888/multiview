# Visualizer Fixes for Dendrogram, SOM, and Heatmap

## Issues Identified

1. **Dendrogram**: Linkage matrix was not being saved because of incorrect attribute name
2. **Heatmap**: Required a layout file when it should work directly with embeddings
3. **SOM**: Should work but may have issues due to layout loading logic
4. **Index**: Heatmap and graph were not included in available modes list

## Fixes Applied

### 1. Fixed Dendrogram Linkage Matrix Attribute Name

**File**: `scripts/visualize_corpus.py`

**Issue**: Code checked for `linkage_matrix_` (with underscore) but the reducer stores it as `linkage_matrix` (without underscore).

**Fix**: Changed two locations:
- Line 1249: `visualizer.reducer.linkage_matrix` (was `linkage_matrix_`)
- Line 1439: `reducer.linkage_matrix` (was `linkage_matrix_`)

**Result**: Dendrogram linkage matrix will now be properly saved to `linkage_matrix.npy` in visualization exports.

### 2. Updated Data Loader to Handle Modes Without Layouts

**File**: `visualizer/src/utils/dataLoader.ts`

**Issue**: The loader required all modes to have a layout entry in the manifest, but `heatmap` and `graph` don't use 2D coordinates.

**Fix**:
- Added special handling for `heatmap` and `graph` modes
- These modes no longer require a layout file
- They skip layout loading and use empty Float32Array for coords

**Result**: Heatmap and graph modes will work even without layout files in the manifest.

### 3. Updated Index Creation to Always Include Heatmap and Graph

**File**: `src/multiview/utils/visualization_utils.py`

**Issue**: The index only included modes that had layout files generated, excluding heatmap and graph.

**Fix**:
- Always append "heatmap" and "graph" to the modes list
- These are now always available for any method with embeddings

**Result**: Heatmap and graph buttons will always appear in the visualizer UI.

## How Each Mode Works

### Dendrogram
- **Requires**: Linkage matrix (hierarchical clustering structure)
- **Uses**: `linkageMatrix` from data, builds tree structure with D3
- **Note**: With the fix, new visualizations will include the linkage matrix

### SOM (Self-Organizing Map)
- **Requires**: 2D coordinates from SOM grid assignment
- **Uses**: `coords` from layout file, renders as scatter plot
- **Status**: Should work correctly (treated as scatter plot variant)

### Heatmap
- **Requires**: Raw embeddings or pre-computed similarity matrix
- **Uses**: `embeddings` from data (NxD or NxN)
- **Detects**: Automatically determines if embeddings or similarity matrix
- **Supports**: Both regular embeddings and BM25 similarity matrices

### Graph
- **Requires**: Embeddings and optional graph structure
- **Uses**: `embeddings` from data, builds force-directed graph
- **Status**: Always available, renders based on embedding similarity

## Testing the Fixes

### For Existing Data
Existing visualizations may not have the linkage matrix saved. To regenerate with the fix:

```bash
# Re-run visualization generation for a specific benchmark
uv run python scripts/run_eval.py --config-name benchmark_fuzzy_debug
```

### For New Data
New evaluation runs will automatically include:
- Proper linkage matrix for dendrogram mode
- Heatmap and graph modes in the available modes list

### Verifying the Fix

1. Check that the manifest includes `linkage_matrix`:
```bash
cat viz/benchmark_*/task_name/method_name/manifest.json | jq .linkage_matrix
```

2. Verify heatmap and graph appear in the UI mode selector (always visible)

3. Test each mode:
   - Dendrogram: Should show hierarchical tree structure
   - SOM: Should show grid-organized scatter plot
   - Heatmap: Should show NxN similarity matrix as colored grid

## Architecture Notes

### Why Heatmap and Graph Don't Need Layouts

- **Heatmap**: Displays a 2D grid representing pairwise similarities between all documents. The "layout" is implicit (row i, col j = similarity between doc i and doc j). Only needs embeddings or similarity matrix.

- **Graph**: Uses force-directed layout computed dynamically in the browser based on embedding distances. Layout is generated on-the-fly, not pre-computed.

### Why Dendrogram Needs Linkage Matrix

Unlike other reducers that just produce 2D coordinates, dendrogram needs the full hierarchical structure (which documents merge at what distances). This is stored in the linkage matrix format from scipy.
