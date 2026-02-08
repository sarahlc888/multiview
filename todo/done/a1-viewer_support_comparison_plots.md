# Comparison Plots: Side-by-Side Criteria Viewer

## Goal

Add the capability for the visualizer to select **2 criteria at a time** and show a side-by-side comparison. This makes it easy to see how the same documents cluster differently under different evaluation criteria (e.g., "difficulty" vs "topic").

The vite app (`npm run dev` in `visualizer/`) already supports single-criterion views. This task extends it to dual-criterion mode.

---

## Feature 1: Side-by-Side Embedding Graphs

**What:** Two synchronized scatter plots (or force graphs, SOMs, etc.) rendered next to each other — same documents, different criteria embeddings.

**UI changes (in `visualizer/src/`):**
- Add a **second criterion selector** dropdown alongside the existing one. When both are populated, switch to split-pane layout.
- The two panels should share:
  - Hover/selection state — hovering a point in panel A highlights the same document in panel B
  - Zoom level (optional, could be toggled with a "sync zoom" checkbox)
  - Color scheme — same document gets the same color in both panels so the user can visually track movement
- Each panel header shows its criterion name

**Implementation notes:**
- The existing `ScatterPlot.tsx` component handles single-view rendering. For split mode, render two `ScatterPlot` instances side-by-side in a flex container.
- Shared hover state: lift hovered-doc-id into a parent component and pass it down as a prop to both scatter plots. The receiving scatter plot highlights that point (e.g., enlarged radius, border ring).
- Layout data already lives in `manifest.json` under `layouts` — each criterion's manifest has its own layout coordinates. Load two manifests in parallel.
- `HeatmapView`, `ForceDirectedGraph`, `DendrogramView`, and `SOMGridView` should also support the linked-highlight prop for consistency, but scatter is the priority.

**Data flow:**
```
User selects criterion A + criterion B
  → fetch viz/{benchmark}/{taskA}/{method}/manifest.json
  → fetch viz/{benchmark}/{taskB}/{method}/manifest.json
  → render ScatterPlot(layoutA, docs) | ScatterPlot(layoutB, docs)
  → shared state: hoveredDocId, selectedDocIds
```

---

## Feature 2: Pareto Accuracy Plot (Triplet Mode)

**What:** When running in triplet evaluation mode, show a scatter plot where each **method** is a point: X = accuracy on criterion A, Y = accuracy on criterion B. This reveals which methods trade off between criteria and which dominate.

**UI:**
- Appears as a third panel (or a tab) when two criteria are selected and triplet results are available.
- Each point is labeled with the method name.
- Draw the Pareto frontier line connecting non-dominated methods.
- Optionally show error bars using the multi-trial `std_accuracy` values from `results.json`.

**Data source:**
- `results.json` already stores per-task, per-method accuracy:
  ```json
  {
    "task_name_A": { "method_1": { "accuracy": 0.72 }, ... },
    "task_name_B": { "method_1": { "accuracy": 0.65 }, ... }
  }
  ```
- Read accuracy for each method under both selected tasks, zip them into (x, y) pairs.

**Implementation notes:**
- New component: `ParetoPlot.tsx` — a simple D3 or recharts scatter.
  - X-axis label = criterion A name, Y-axis label = criterion B name
  - Each dot = one method, labeled on hover (tooltip with method name + both accuracies)
  - Pareto frontier: sort points by X descending, walk forward keeping only points where Y increases — connect those with a stepped or smooth line.
- If multi-trial data is present (`mean_accuracy`, `std_accuracy`), render error bars (ellipses or crosshairs).
- Pull trial grouping logic from `Leaderboard.tsx:groupTrialMethods()` to aggregate trial runs before plotting.

**Stretch: instruction sensitivity overlay**
- `evaluation_utils.py` already computes `instruction_sensitivity` (delta between instruction-based and baseline methods). Could color-code or annotate points by sensitivity magnitude.

---

## Implementation Plan

### Phase 1 — Dual criterion selection + split scatter
1. Add second criterion dropdown to the experiment selector UI
2. Create a `ComparisonView` wrapper component that renders two visualization panels in a flex row
3. Add shared hover/selection state (lifted to `ComparisonView`)
4. Wire up `ScatterPlot` to accept + render `highlightedDocId` prop
5. Test with GSM8K — compare e.g. "difficulty" vs "computation_type"

### Phase 2 — Pareto accuracy plot
1. Create `ParetoPlot.tsx` component
2. Load `results.json` for both selected criteria
3. Compute Pareto frontier from (accuracy_A, accuracy_B) pairs
4. Render scatter + frontier line + method labels
5. Add error bars from trial statistics if available

### Phase 3 — Polish
- Sync zoom toggle between panels
- Extend linked highlighting to `ForceDirectedGraph`, `HeatmapView`, `SOMGridView`, `DendrogramView`
- Add URL params so dual-criterion views are shareable/bookmarkable
- Mobile/narrow viewport fallback: stack panels vertically instead of side-by-side
