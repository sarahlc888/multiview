# Thumbnail Support

The interactive visualizer now supports displaying thumbnail images instead of simple circle markers.

## How It Works

### Python Export
When you use `--marker-type thumbnail`, the export script:
1. Generates thumbnail images using existing infrastructure
2. Copies them to `{export_dir}/thumbnails/`
3. Adds thumbnail paths to `manifest.json`

### JavaScript Viewer
The ScatterPlot component:
1. Checks if `thumbnails` field exists in manifest
2. Renders `<image>` elements instead of `<circle>` elements
3. Scales images on hover for better visibility

## Usage

### Export with Thumbnails

```bash
# GSM8K computational graphs
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

### Batch Export with Thumbnails

```bash
# Export multiple reducers with thumbnails
for reducer in tsne som; do
  uv run python scripts/analyze_corpus.py \
    --dataset gsm8k \
    --embedding-preset hf_qwen3_embedding_8b \
    --criterion arithmetic \
    --reducer $reducer \
    --marker-type thumbnail \
    --export-format web \
    --output viz/gsm8k/arithmetic \
    --max-docs 100
done
```

## Supported Datasets

Currently, thumbnails are automatically generated for:

- **GSM8K** (when criterion is `arithmetic`): Computational graph thumbnails

### Extending to Other Datasets

To add thumbnail support for other datasets:

1. Extend `create_thumbnail_images()` in `analyze_corpus.py`
2. Add dataset-specific thumbnail generation logic
3. Return list of image paths

Example:
```python
def create_thumbnail_images(documents, dataset_name, criterion, output_dir):
    if dataset_name == "gsm8k" and criterion == "arithmetic":
        return create_gsm8k_marker_images(...)
    elif dataset_name == "my_dataset":
        return create_my_custom_thumbnails(...)
    else:
        return []
```

## Manifest Format

When thumbnails are present, the manifest includes:

```json
{
  "version": 1,
  "dataset": "gsm8k",
  "criterion": "arithmetic",
  "n_docs": 10,
  "thumbnails": [
    "thumbnails/thumb_0.png",
    "thumbnails/thumb_1.png",
    ...
  ]
}
```

## File Structure

```
viz/gsm8k/arithmetic/
├── manifest.json          # Includes thumbnail paths
├── documents.txt
├── embeddings.npy
├── layout_tsne.npy
└── thumbnails/
    ├── thumb_0.png        # High-res computational graph
    ├── thumb_1.png
    └── ...
```

## Viewer Behavior

- **Without thumbnails**: Shows blue circle markers (default)
- **With thumbnails**: Shows image markers at 40×40px
- **On hover**: Scales image to 48×48px and shows tooltip
- **Fallback**: If thumbnail path is null/missing, no image shown

## Performance

- Thumbnails are loaded lazily by the browser
- High-resolution images (3570×3570px) are scaled down in SVG
- No performance impact on initial load (only manifest is parsed)

## Testing

Test data with thumbnails is available:
- Dataset: `gsm8k`
- Criterion: `arithmetic_thumb`
- Mode: `tsne`
- Documents: 10
- Thumbnails: 10 computational graphs

To view:
```bash
cd visualizer
npm run dev
# Select "gsm8k" → "arithmetic_thumb" → "TSNE"
```
