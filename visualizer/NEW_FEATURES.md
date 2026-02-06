# New Visualizer Features

## ✅ What's Working

### 1. Bar Chart Leaderboard
- Toggle between **Table** and **Chart** views using buttons in the top-right
- Chart view features:
  - Horizontal bars colored by rank (🥇 gold, 🥈 silver, 🥉 bronze)
  - Accuracy percentages on the right
  - Correct/total counts displayed inside bars
  - Click any bar to switch to that method's visualization

### 2. Interactive Triplet Hover
- Hover over anchor/positive/negative boxes in the sidebar to:
  - See a 👈 pointer emoji in the sidebar
  - Highlight the corresponding point on the visualization
  - Make the point larger and add an outer glow
  - Fade other triplet points to 40% opacity
  - Make the label larger and bolder

### 3. Fixed Issues
- ✅ Embeddings now load directly from `outputs/` directory
- ✅ NaN values (from document rewrite methods) are automatically filtered
- ✅ Triplets are included in all visualizations

## 🚀 How to Test

1. **Start the visualizer:**
   ```bash
   cd visualizer
   npm run dev
   ```
   Then open http://localhost:5173

2. **Test Bar Chart:**
   - Select a benchmark/dataset/criterion
   - Click the "Chart" button (top-right of leaderboard)
   - Click different bars to switch between methods

3. **Test Interactive Triplet Hover:**
   - Select a method with embeddings (e.g., qwen3_8b_baseline)
   - Make sure "Show" checkbox is checked in triplet panel
   - Hover over the colored boxes (Anchor/Positive/Negative)
   - Watch the visualization highlight the corresponding points
   - Use Prev/Next buttons to navigate through triplets

## 📁 Data Available

Current visualizations in `viz/benchmark_fuzzy_debug2/`:
- `gsm8k__final_expression__tag__5` (3 methods)
- `ut_zappos50k__color__tag__5` (3 methods)
- `ut_zappos50k__functional_type__tag__5` (3 methods)

Each includes:
- t-SNE layouts
- Embeddings
- Triplets with quality ratings
- Benchmark results for leaderboard

## 🎨 Visual Features

**Leaderboard Chart View:**
- Gold (#FFD700) → Bronze (#CD7F32) gradient for top 3
- Blue gradient for others
- Smooth transitions and hover effects

**Triplet Hover:**
- Anchor: Blue (#0066cc)
- Positive: Green (#00aa00)
- Negative: Red (#cc0000)
- Outer glow ring when hovered
- Size changes (8px → 10px)
- Opacity fading for context

## 🔧 Technical Details

**Files Modified:**
- `visualizer/src/components/Leaderboard.tsx` - Added chart view
- `visualizer/src/render/ScatterPlot.tsx` - Added hover interaction
- `visualizer/src/types/manifest.ts` - Added 'graph' layout type
- `visualizer/src/App.css` - Added animations
- `visualizer/tsconfig.json` - Relaxed linting for D3 callbacks

**Triplet Data Flow:**
1. `outputs/benchmark_fuzzy_debug2/triplets/{task}/triplets.json` (source)
2. Copied to `viz/benchmark_fuzzy_debug2/{task}/{method}/triplets.json`
3. Loaded by `dataLoader.loadTriplets()`
4. Rendered in ScatterPlot sidebar
