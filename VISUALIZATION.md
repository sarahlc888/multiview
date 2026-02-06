# Visualization Guide
  Full Feature List

  ✅ Experiment selector - Browse multiple benchmark runs
  ✅ Method dropdown - Compare different embeddings
  ✅ Leaderboard - See method rankings with accuracy
  ✅ Graph mode - Physics-based force layout with sliders
  ✅ Image thumbnails - Actual images for image datasets
  ✅ Auto-generation - Runs automatically after evaluation
  ✅ Index merging - Never lose old experiments


Two visualization modes:

### 1. Automatic Benchmark Visualization (Recommended)
Visualize **evaluation results** with triplets - automatic with `run_eval.py`:
- Shows how embedding methods organize your evaluated corpus
- Reuses cached embeddings from evaluation
- Displays triplet relationships
- Fully automatic, no manual steps needed

### 2. Manual Corpus Exploration
Visualize **unlabeled corpora** for investigation - use `visualize_corpus.py`:
- Investigate structure of a new corpus before creating benchmarks
- No triplets needed
- Generates fresh embeddings
- Manual exploration tool

**For benchmark results, visualizations are automatic.** Only use `visualize_corpus.py` when investigating new unlabeled corpora.

---

## Dashboard Architecture

The web dashboard provides an integrated view for exploring benchmark results:

### Leaderboard & Method Selection

**Leaderboard View:**
- Shows ranking of all embedding methods by accuracy
- Acts as an interactive selector for which embedding to visualize
- Click any method to load its visualization

**Bar Chart Version:**
- Alternative visualization of the same leaderboard data
- Easier to compare relative performance at a glance
- Also acts as a selector - click bars to switch embeddings

**Purpose:** The leaderboard serves dual purposes:
1. **Performance overview** - See which embeddings work best
2. **Embedding selector** - Choose which embedding type to explore in detail

### Dimensionality Reduction Options

For each selected embedding, multiple reduction methods are available:

- **t-SNE** - Preserves local structure, good for cluster discovery
- **UMAP** - Fast, preserves both local and global structure
- **PCA** - Linear projection, interpretable axes
- **SOM** - Self-organizing map with grid layout
- **Force Graph** - Physics-based layout with adjustable parameters

**Instant switching:** All layouts are pre-computed during visualization generation, so switching between them is instantaneous.

### Triplets Sidebar

**Triplet Display:**
- Sidebar shows all triplets for the current task
- Each triplet displays:
  - **Anchor** document (reference point)
  - **Positive** document (should be similar to anchor)
  - **Negative** document (should be dissimilar from anchor)
- Quality assessment with rating and reasoning

**Interactive Highlighting:**
- Hover over any triplet in the sidebar to highlight it in the visualization
- When hovering:
  - **Anchor point** highlighted in blue
  - **Positive point** highlighted in green
  - **Negative point** highlighted in red
  - Connecting lines drawn between the three points
- Click to lock the highlight, click again to unlock

**Navigation:**
- Prev/Next buttons to cycle through triplets
- Toggle triplets on/off to declutter the view
- Search/filter triplets by quality rating

### Workflow

```
1. Select experiment → 2. View leaderboard → 3. Click embedding method →
4. Choose dim reduction → 5. Explore triplets → 6. Hover to highlight
```

This integrated approach lets you quickly assess which embeddings work best, then dive deep into understanding *why* through the triplet visualizations.

---

## Part 1: Automatic Benchmark Visualization

Visualizations are **automatically generated** after running evaluations, enabled by default.

### Quick Start

```bash
# Run evaluation (auto-generates visualizations)
uv run python scripts/run_eval.py --config-name benchmark_fuzzy_debug
```

This will:
1. Run the evaluation
2. Automatically generate visualizations for all tasks
3. Create a web viewer at `viz/benchmark_fuzzy_debug/`
4. Auto-enable thumbnails for compatible datasets (images, GSM8K)

### Configuration

Edit your benchmark config (e.g., `configs/benchmark_fuzzy_debug.yaml`):

```yaml
# Auto-visualization configuration
auto_visualize: true  # Set to false to disable
visualization:
  reducers: ["tsne", "umap"]  # Which reducers to use
  thumbnails: true            # Enable image/graph thumbnails
  output_dir: "viz"           # Output directory
```

### Disable Auto-Visualization

Set `auto_visualize: false` in your config, or run with:

```bash
uv run python scripts/run_eval.py --config-name benchmark_fuzzy_debug auto_visualize=false
```

### Re-generate with Different Settings

To regenerate visualizations with different reducers without re-running the evaluation:

```yaml
visualization:
  reducers: ["tsne", "umap", "pca"]  # Add more reducers
```

Then run:
```bash
uv run python scripts/run_eval.py --config-name benchmark_fuzzy_debug
```

The evaluation will be skipped (already cached), but visualizations will regenerate with the new settings.

### Output Structure

Visualizations are saved to `viz/` by default:

```
viz/
├── index.json                              # Web viewer index
└── benchmark_fuzzy_debug/
    ├── gsm8k__arithmetic/
    │   ├── qwen3_8b_with_instructions/
    │   │   ├── manifest.json               # Metadata
    │   │   ├── documents.txt               # Document texts
    │   │   ├── embeddings.npy              # Raw embeddings
    │   │   ├── layout_tsne.npy             # 2D t-SNE coordinates
    │   │   └── thumbnails/                 # Computational graphs
    │   └── qwen3_8b_no_instructions/
    ├── haiku__imagery/
    │   └── qwen3_8b_with_instructions/
    └── ut_zappos50k__heel_height/
        └── qwen3_8b_with_instructions/
            ├── layout_tsne.npy
            └── thumbnails/                 # Actual shoe images
```

### Standard Workflow

```bash
# 1. Create evaluation artifacts
uv run python scripts/create_eval.py --config-name my_benchmark

# 2. Run evaluation (auto-generates visualizations)
uv run python scripts/run_eval.py --config-name my_benchmark

# Output:
#   outputs/my_benchmark/results/     - Evaluation results
#   viz/my_benchmark/                 - Visualizations (automatic!)
```

---

## Part 2: Manual Corpus Exploration

Use `visualize_corpus.py` for manual exploration of unlabeled corpora.

### Quick Start

```bash
# Single export (with circles)
uv run python scripts/visualize_corpus.py \
  --dataset gsm8k \
  --embedding-preset hf_qwen3_embedding_8b \
  --criterion arithmetic \
  --reducer tsne \
  --export-format web \
  --output viz/gsm8k/arithmetic \
  --max-docs 100

# With thumbnails (for GSM8K computational graphs)
uv run python scripts/visualize_corpus.py \
  --dataset gsm8k \
  --embedding-preset hf_qwen3_embedding_8b \
  --criterion arithmetic \
  --reducer tsne \
  --marker-type thumbnail \
  --export-format web \
  --output viz/gsm8k/arithmetic \
  --max-docs 100

# Export additional modes to the same location (they will merge)
uv run python scripts/visualize_corpus.py \
  --dataset gsm8k \
  --embedding-preset hf_qwen3_embedding_8b \
  --criterion arithmetic \
  --reducer som \
  --export-format web \
  --output viz/gsm8k/arithmetic \
  --max-docs 100
```

### Example Workflow

```bash
# 1. Export multiple modes for one dataset/criterion using multiple commands
for reducer in tsne som umap pca; do
  uv run python scripts/visualize_corpus.py \
    --dataset gsm8k \
    --embedding-preset hf_qwen3_embedding_8b \
    --criterion arithmetic \
    --reducer $reducer \
    --export-format web \
    --output viz/gsm8k/arithmetic \
    --max-docs 200
done

# 2. Start visualizer (see below)
cd visualizer
npm run dev

# 3. In browser:
#    - Select "gsm8k" from dataset dropdown
#    - Select "arithmetic" from criterion dropdown
#    - Toggle between t-SNE, SOM, UMAP, PCA modes
#    - Hover over points to see document text
#    - Pan/zoom to explore
```

### Special Method: In-One-Word Embeddings

The in-one-word method is a specialized embedding technique for exploring category-based document structure.

#### What is In-One-Word?

The in-one-word method:
1. Generates category annotations (e.g., arithmetic operations for GSM8K)
2. Prompts the model to categorize documents "in one word"
3. Extracts hidden states or logprob-weighted embeddings
4. Uses these as embeddings for visualization

This shows how problems cluster by their category (e.g., addition vs. multiplication).

#### Quick Start

**GPU-based visualization:**

```bash
uv run python scripts/visualize_corpus.py \
    --dataset gsm8k \
    --embedding-preset inoneword_hf_qwen3_4b \
    --criterion arithmetic \
    --reducer tsne \
    --export-format web \
    --output viz/gsm8k_inoneword \
    --max-docs 100
```

This will:
- Sample 100 random GSM8K problems
- Auto-generate arithmetic category annotations
- Create in-one-word embeddings using local HF model
- Visualize with t-SNE, colored by arithmetic category

**With computational graph thumbnails:**

```bash
uv run python scripts/visualize_corpus.py \
    --dataset gsm8k \
    --embedding-preset inoneword_hf_qwen3_4b \
    --criterion arithmetic \
    --marker-type thumbnail \
    --export-format web \
    --output viz/gsm8k_inoneword_graphs \
    --max-docs 100
```

#### Available In-One-Word Presets

All presets use local HuggingFace models with hidden state extraction:

- **`inoneword_hf_qwen3_8b`** - Qwen 3-8B hidden states (GPU required, recommended)
- **`inoneword_hf_qwen3_4b`** - Qwen 3-4B hidden states (GPU required, faster, less memory)
- **`inoneword_hf_generic`** - Generic preset (customizable)

#### Comparison with Standard Embeddings

To compare in-one-word vs standard embeddings:

```bash
# Standard embeddings
uv run python scripts/visualize_corpus.py \
    --dataset gsm8k \
    --embedding-preset hf_qwen3_embedding_8b \
    --export-format web \
    --output viz/gsm8k_standard \
    --max-docs 100

# In-one-word embeddings
uv run python scripts/visualize_corpus.py \
    --dataset gsm8k \
    --embedding-preset inoneword_hf_qwen3_4b \
    --criterion arithmetic \
    --export-format web \
    --output viz/gsm8k_inoneword \
    --max-docs 100
```

The in-one-word version should show clearer category clustering since it's explicitly conditioned on the criterion.

#### Expected Results

The visualization will show:
- Each point is a document (e.g., GSM8K problem)
- Colors indicate categories (e.g., arithmetic operations)
- Similar documents cluster together in 2D space
- Legend shows category names

For GSM8K arithmetic, categories might include:
- addition
- subtraction
- multiplication
- division
- fractions
- percentages
- ratios
- mixed_operations

#### Caching

Results are automatically cached:
- Annotations: `outputs/cache/<run_name>_annotation/`
- Embeddings: `outputs/cache/<run_name>_embeddings/`

To force refresh: add `--force-refresh`

#### Troubleshooting

**Error: "In-one-word presets require --criterion"**
- Solution: Add `--criterion arithmetic` (or other criterion name)

**Out of Memory (GPU)**
- Reduce `--max-docs` to 50 or fewer
- Use smaller model: `inoneword_hf_qwen3_4b` instead of 8B
- Reduce batch size in preset overrides

---

## Part 3: Interactive Web Viewer

### Starting the Viewer

```bash
cd visualizer
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

### Dashboard Components

#### 1. Experiment & Method Selection

**Experiment Selector:**
- Browse multiple benchmark runs
- Each experiment shows its configuration and task count

**Leaderboard (Table View):**
- Shows all embedding methods ranked by accuracy
- Displays performance metrics for each method
- Click any row to visualize that embedding

**Leaderboard (Bar Chart View):**
- Same data as table, visualized as horizontal bars
- Easier to compare relative performance
- Click any bar to switch to that embedding

#### 2. Dimensionality Reduction Views

Choose from multiple pre-computed layouts:

**ScatterPlot Modes** (t-SNE, UMAP, PCA, SOM):
- Interactive pan/zoom
- Hover tooltips showing document text
- Color-coded points or thumbnail images
- **Thumbnail support**:
  - GSM8K: Computational graph thumbnails
  - Image datasets: Actual images (shoes, artwork, etc.)
  - Fallback: Colored circles

**Force Graph Mode:**
- Physics-based force-directed layout
- Draggable nodes
- Adjustable parameters via sliders:
  - Link strength
  - Charge force
  - Collision radius

**Dendrogram View:**
- Hierarchical clustering tree
- Shows clustering relationships
- Requires linkage matrix

#### 3. Triplet Sidebar

**Triplet Browser:**
- Lists all triplets for the current task
- Each triplet shows:
  - Anchor document text
  - Positive document text
  - Negative document text
  - Quality rating and reasoning

**Interactive Highlighting:**
- **Hover** over a triplet to highlight it in the visualization:
  - Anchor point → blue
  - Positive point → green
  - Negative point → red
  - Connecting lines drawn between points
- **Click** to lock the highlight
- **Click again** to unlock

**Navigation & Controls:**
- Prev/Next buttons to cycle through triplets
- Show/hide toggle to declutter the view
- Quality filter to show only high/low-quality triplets

#### 4. Document Details

- Hover over any point to see full document text
- Click to pin the tooltip
- Works with both regular points and highlighted triplets

---

## Part 4: Technical Details

### Data Format

Each export creates:

```
viz/
  {dataset}/
    {criterion}/
      manifest.json         # Metadata + available modes
      documents.txt         # One doc per line (newlines escaped)
      embeddings.npy        # Raw embeddings (n_docs × dim)
      layout_tsne.npy       # 2D coordinates for t-SNE
      layout_som.npy        # 2D coordinates for SOM
      linkage_matrix.npy    # (optional) For dendrogram
      thumbnails/           # (optional) Thumbnail images
        thumb_0.png
        thumb_1.png
        ...
  index.json               # Auto-generated index
```

### Manifest Format

```json
{
  "version": 1,
  "dataset": "gsm8k",
  "criterion": "arithmetic",
  "n_docs": 100,
  "embedding_dim": 4096,
  "documents_path": "documents.txt",
  "embeddings_path": "embeddings.npy",
  "layouts": {
    "tsne": "layout_tsne.npy",
    "som": "layout_som.npy"
  },
  "thumbnails": [
    "thumbnails/thumb_0.png",
    "thumbnails/thumb_1.png",
    ...
  ]
}
```

### Key Files

**Python (data export):**
- `scripts/visualize_corpus.py` - Manual corpus visualization
- `scripts/run_eval.py` - Automatic benchmark visualization

**JavaScript (viewer):**
- `visualizer/src/App.tsx` - Main application
- `visualizer/src/render/ModeRenderer.tsx` - Mode switcher
- `visualizer/src/render/ScatterPlot.tsx` - Scatter plot renderer
- `visualizer/src/render/DendrogramView.tsx` - Dendrogram renderer
- `visualizer/src/utils/npyLoader.ts` - NumPy file loader
- `visualizer/src/utils/dataLoader.ts` - Data loading utilities
- `visualizer/scripts/build-dataset-index.cjs` - Index builder

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python: Data Export                                    │
│  - Load documents                                       │
│  - Generate embeddings                                  │
│  - Run dimensionality reduction                         │
│  - Export .npy files + manifest.json                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  viz/ directory                                         │
│  - manifest.json (metadata)                             │
│  - documents.txt (text)                                 │
│  - embeddings.npy (vectors)                             │
│  - layout_*.npy (2D coords)                             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  React Visualizer                                       │
│  - Load manifest + data files                           │
│  - Render with D3.js                                    │
│  - Interactive controls                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Part 5: Reference

### Thumbnail Support

Thumbnails are **automatically enabled** for:

**Image Datasets:**
- **ut_zappos50k**: Actual shoe images from the dataset
- **met_museum**: Artwork images from Met collection
- **example_images**: Custom image dataset

**GSM8K with Arithmetic:**
- **gsm8k + arithmetic criterion**: Computational graph visualizations

Thumbnails work with:
- Data URIs (base64 encoded images)
- HTTP/HTTPS URLs
- Local file paths

### Reducers

Available dimensionality reduction methods:

- **tsne**: t-SNE (good general purpose, preserves local structure)
- **umap**: UMAP (fast, preserves both local and global structure)
- **pca**: PCA (fast, linear, interpretable axes)
- **som**: Self-organizing map (grid layout)
- **dendrogram**: Hierarchical clustering (requires thumbnails)

### Export Formats

#### Web viewer (default)
Interactive web-based visualization:

```bash
--export-format web
```

Output: `manifest.json`, `*.npy` files, `thumbnails/`

#### Static plots
PNG/SVG images:

```bash
--export-format png
# or
--export-format svg
```

Output: `.png` or `.svg` files

#### Both
Generate both formats:

```bash
--export-format all
```

### Performance

- **Embeddings**: Reused from evaluation cache (instant)
- **t-SNE**: ~1-5 seconds for 100-500 documents
- **UMAP**: ~0.5-2 seconds for 100-500 documents
- **Thumbnails**: Depends on dataset
  - Image datasets: Instant (uses existing images)
  - GSM8K graphs: ~1-2 seconds per graph

---

## Tips

1. **Keep auto_visualize enabled**: It's on by default and has minimal overhead
2. **Start with t-SNE**: Good default for most datasets
3. **Thumbnails are automatic**: Enabled by default for image datasets and GSM8K
4. **Try multiple reducers**: Add them to your config to see different structure
5. **Everything is configured in YAML**: No command-line juggling needed
6. **Start small**: Test with `--max-docs 100` first
7. **Pre-compute modes**: Export all modes you want to explore (switching is instant)
8. **Merge layouts**: Run exports to the same output dir to add more modes
9. **Auto-index**: The dev server auto-rebuilds `index.json` on start

---

## Troubleshooting

### Visualizations not generated

Make sure `auto_visualize: true` in your config (it's the default).

Check the end of `run_eval.py` output for any visualization errors.

### Missing thumbnails for image datasets

Check that your documents have `image_path` fields. For ut_zappos50k, ensure the dataset loaded correctly.

### Visualization takes too long

Use fewer reducers in your config:

```yaml
visualization:
  reducers: ["tsne"]  # Just one reducer
```

Or disable temporarily:

```yaml
auto_visualize: false
```
