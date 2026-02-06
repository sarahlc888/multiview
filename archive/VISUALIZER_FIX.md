# Visualizer Leaderboard Fix ✅

## Problem
The leaderboard wasn't populating because:
1. Manifests had `criterion: "default"` instead of actual criterion names
2. Index.json used "default" as the criterion key
3. App.tsx constructed task name as `dataset__criterion` but results.json used full task names like `dataset__criterion__style__num`

## Solution
1. **Fixed criterion parsing** in `visualize_corpus.py`:
   - Extract criterion from task name (e.g., `gsm8k__final_expression__tag__5` → `final_expression`)
   - Pass correct criterion to manifest export

2. **Updated index structure**:
   - Now uses actual criterion names: `gsm8k` → `final_expression` (not "default")

3. **Fixed task name lookup** in `App.tsx`:
   - Extract full task name from method path instead of constructing from dataset + criterion
   - Now correctly matches task names in results.json

## Testing

1. **Start the visualizer:**
   ```bash
   cd visualizer
   npm run dev
   ```
   Open http://localhost:5173

2. **Verify leaderboard:**
   - Select: Experiment="benchmark_fuzzy_debug2", Dataset="gsm8k", Criterion="final_expression"
   - You should see a populated leaderboard with methods:
     - qwen3_8b_baseline: 80.0%
     - dr_gemini_lite_openai_small: 60.0%
     - qrv_gemini_openai_k10_dev25: 40.0%
     - bm25_baseline: 20.0%

3. **Test bar chart:**
   - Click "Chart" button → see horizontal bars
   - Click different bars → switches to that method's visualization

4. **Test triplet hover:**
   - Select qwen3_8b_baseline
   - Hover over Anchor/Positive/Negative boxes
   - Points should highlight on the visualization

## Files Modified
- `scripts/visualize_corpus.py`: Criterion parsing and propagation
- `visualizer/src/App.tsx`: Task name extraction from path
- `viz/index.json`: Regenerated with correct criteria
- `viz/benchmark_fuzzy_debug2/`: Regenerated with correct manifests

## Current Data
- ✅ gsm8k__final_expression__tag__5 (3 methods with embeddings, 5 triplets each)
- ⚠️ ut_zappos50k tasks failed (documents are dicts, need string conversion)
