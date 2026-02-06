# Triplet Highlighting in Force-Directed Graph

## Summary

Successfully implemented triplet highlighting for the Force-Directed Graph visualization. Users can now:

1. **View triplets** - A sidebar appears on the right when triplets are available
2. **Navigate triplets** - Use Prev/Next buttons to cycle through all triplets
3. **See node highlighting** - Current triplet nodes are color-coded:
   - **Anchor**: Blue (#0066cc)
   - **Positive**: Green (#00aa00)
   - **Negative**: Red (#cc0000)
   - **Other nodes**: Gray (#888)
4. **Toggle triplet display** - Show/hide triplets using the checkbox in the sidebar
5. **View triplet details** - See full document text, margin, correctness, and quality assessments

## Changes Made

### 1. ForceDirectedGraph.tsx
- Added `triplets` prop to interface (line ~10)
- Added import for `TripletData` type
- Added state management for triplet navigation:
  - `selectedTripletIndex` - current triplet being displayed
  - `showTriplets` - whether to show triplet highlighting
  - `hasTriplets` - computed flag for triplet availability
- Modified node rendering logic to color nodes based on current triplet role (lines ~205-230)
- Updated render dependencies to include triplet state (line ~237)
- Added triplets sidebar UI (lines ~472-650) with:
  - Header with show/hide toggle
  - Navigation controls (Prev/Next buttons)
  - Triplet counter display
  - Color-coded document sections for anchor/positive/negative
  - Margin and correctness badges
  - Quality assessment display

### 2. ModeRenderer.tsx
- Updated graph case to pass `triplets={data.triplets}` prop (line ~77)

## Implementation Details

### Layout Structure
```
[Controls (280px LEFT)] | [Canvas (flex:1 CENTER)] | [Triplets Sidebar (350px RIGHT)]
```

The controls sidebar remains on the left unchanged. The triplets sidebar appears conditionally on the right when triplets are available.

### Node Coloring Logic
When a triplet is displayed:
- Nodes involved in the triplet get their role-specific colors
- All other nodes are grayed out (#888)
- When triplets are hidden, all nodes show their default rainbow colors

### Triplets Sidebar
- Shows current triplet count (e.g., "1 / 20")
- Displays margin and correctness information when available
- Shows anchor, positive, and negative documents with:
  - Color-coded backgrounds matching node colors
  - Truncated text (300 chars) with scrollable containers
  - Monospace font for better readability
- Includes quality assessment when available

## Testing

Build completed successfully with no TypeScript errors:
```bash
cd visualizer && npm run build
# ✓ built in 1.21s
```

To test the feature:
1. Start the dev server: `cd visualizer && npm run dev`
2. Navigate to a dataset with triplets (e.g., `benchmark_fuzzy_debug2/gsm8k__final_expression__tag__5/pseudologit_proposed_trial2`)
3. Switch to the "Graph" visualization mode
4. Verify:
   - Triplets sidebar appears on the right
   - Node colors update when navigating between triplets
   - Toggle works to show/hide highlighting
   - All triplet information displays correctly

## Data Integration

The feature integrates with existing data structures:
- Reads triplets from `data.triplets` passed via ModeRenderer
- Supports all TripletData fields including:
  - `anchor_id`, `positive_id`, `negative_id` - for node highlighting
  - `anchor`, `positive`, `negative` - document text display
  - `margin`, `correct`, `is_tie` - evaluation metrics
  - `positive_score`, `negative_score` - similarity scores
  - `quality_assessment_with_annotations` - quality ratings

## Performance Considerations

- Minimal overhead: Simple ID comparison in render loop
- Canvas redraw triggered by state changes via RAF loop
- No performance issues expected for typical dataset sizes
