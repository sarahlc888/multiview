# Task: Add New Yorker covers docset

## Goal
Create a new image-based docset for New Yorker magazine covers (~5,101 images spanning 1925–2025), enabling visual/cultural analysis across a century of illustration.

## Type
NEW_DOCSET

## Data source
- Already scraped: `/Users/sarahchen/code/pproj/scrape/newyorker_covers/`
- Structure: `{year}/{hex_id}.jpg` (e.g., `2024/6584427c986226975368f9b6.jpg`)
- 101 year directories (1925–2025), ~5,101 JPEG images total
- Images are 320x437 baseline JPEGs
- No metadata files — just images. The hex IDs likely correspond to New Yorker CMS object IDs.

## Reference files
- `src/multiview/docsets/met_museum.py` — closest existing pattern: image-based docset with `get_document_image()`, local file caching
- `src/multiview/docsets/ut_zappos50k.py` — another image docset
- `src/multiview/docsets/base.py` — BaseDocSet abstract class
- `src/multiview/docsets/__init__.py` — register new class in DOCSETS dict + imports
- `configs/available_criteria.yaml` — add criteria section (see `met_museum:` for image criteria patterns)

## Steps

### 1. Create docset: `src/multiview/docsets/newyorker_covers.py`
- Class `NewYorkerCoversDocSet(BaseDocSet)`
- `DATASET_PATH` = path to scraped directory (or make configurable via `config.get("images_dir")`)
- `DESCRIPTION = "New Yorker magazine cover illustrations"`
- `DOCUMENT_TYPE = "magazine cover illustration"`
- `DATASET_NAME = "newyorker_covers"` (for auto-resolving criteria from `available_criteria.yaml`)
- `KNOWN_CRITERIA = []` (no deterministic criteria beyond `word_count` from base)
- `load_documents()` — walk year dirs, collect image paths + extract year from dir name
  - Support `config.get("max_docs")` for limiting corpus size
  - Support `config.get("seed", 42)` for reproducible sampling when max_docs < total
  - Call `self._deduplicate(documents)` before returning
- `get_document_text(doc)` — return `doc.get("text", "")` (minimal: year + filename since no text metadata)
- `get_document_image(doc)` — return `doc.get("image_path")`
- Each document: `{"image_path": str, "text": str, "year": int}`

### 2. Register in `src/multiview/docsets/__init__.py`
- Add import: `from multiview.docsets.newyorker_covers import NewYorkerCoversDocSet`
- Add to `DOCSETS` dict: `"newyorker_covers": NewYorkerCoversDocSet,`
- Add to `__all__` list: `"NewYorkerCoversDocSet"`

### 3. Add criteria to `configs/available_criteria.yaml`
Add `newyorker_covers:` section (see `met_museum:` at line 407 for image criteria patterns):
```yaml
newyorker_covers:
  visual_style:
    description: "Illustration style/technique (watercolor, line drawing, collage, photorealistic, abstract, etc.)"
  subject_matter:
    description: "What is depicted on the cover (city scene, portrait, seasonal, political, surreal, etc.)"
  cultural_moment:
    description: "What cultural era/mood does the cover reflect? (wartime, counterculture, digital age, etc.)"
  composition:
    description: "Visual composition — use of color, negative space, framing, typography integration"
```

### 4. Create Hydra config: `configs/corpus_new_yorker_covers.yaml`
This is the config consumed by `scripts/analyze_corpus.py` via:
```python
@hydra.main(config_path="../configs", config_name="benchmark", version_base=None)
```
The `--config-name corpus_new_yorker_covers` flag makes Hydra look for `configs/corpus_new_yorker_covers.yaml`.

Template (pattern after `configs/benchmark_met_100.yaml`):
```yaml
run_name: "corpus_new_yorker_covers"
logging:
  level: DEBUG
  output_file: "outputs/logs/${run_name}.log"
seed: 42
use_cache: true
reuse_cached_triplets: true
step_through: false

auto_visualize: true
visualization:
  reducers: ["tsne", "umap", "pca", "som", "dendrogram"]
  thumbnails: true
  output_dir: "outputs/viz"

tasks:
  defaults:
    max_docs: 200
    max_triplets: 25
    num_synthetic_docs: 0
    triplet_style: "lm_all"
    candidate_strategy: "multi"
    use_spurious_hard_negs: true
    embedding_preset: "hf_qwen3_embedding_8b"
    max_num_candidates: 10
    n_schema_samples: 10
    rate_triplet_quality: true
    min_triplet_quality: 2
    validate_consistency: true
    consistency_max_invalid: 1
    prelabeled_selection: "hard_negatives"

  task_list:
    - document_set: newyorker_covers
      criterion: visual_style
      triplet_style: lm_tags
    - document_set: newyorker_covers
      criterion: subject_matter
      triplet_style: lm_tags

methods_to_evaluate:
  document_rewrite:
    - name: dr_gemini_lite_openai_small
      embedding_preset: openai_embedding_small
      summary_preset: document_summary_gemini_flash
  embeddings:
    - name: voyage_multimodal_3_5
      preset: voyage_multimodal_3_5
```

## Reference files
| File | Why |
|---|---|
| `src/multiview/docsets/base.py` | `BaseDocSet` ABC — required methods: `load_documents()`, `get_document_text()`, `get_document_image()` |
| `src/multiview/docsets/met_museum.py` | Closest pattern: image docset with local file cache, `image_path` field, `get_document_image()` |
| `src/multiview/docsets/ut_zappos50k.py` | Another image docset reference |
| `src/multiview/docsets/example_images.py` | Minimal image docset (hardcoded list, good for structure reference) |
| `src/multiview/docsets/onion_headlines.py` | Simple HF `load_dataset` pattern with `DATASET_NAME` for criteria auto-resolution |
| `src/multiview/docsets/__init__.py` | Registration: imports, `DOCSETS` dict (line 39), `__all__` (line 77) |
| `configs/available_criteria.yaml` | Criteria definitions — see `met_museum:` (line 407) for image dataset criteria pattern |
| `configs/benchmark_met_100.yaml` | Full config template for image dataset with `visualization`, `tasks.defaults`, `methods_to_evaluate` |
| `configs/benchmark_zappos_100.yaml` | Another image config template (uses `voyage_multimodal_3_5` for multimodal embeddings) |
| `scripts/analyze_corpus.py` | Consumer of config — `@hydra.main(config_path="../configs")`, calls `setup_benchmark_config()` then `generate_visualizations_for_benchmark()` |
| `README.md` line 48 | References `python scripts/analyze_corpus.py --config-name corpus_new_yorker_covers` as example |

## Notes
- This is distinct from b08 (New Yorker Caption Contest / cartoons) — covers are standalone illustrations, not captioned jokes
- The year metadata provides a natural temporal dimension for analysis
- With 5K images this is a substantial multimodal benchmark dataset
- Consider whether to try scraping cover metadata (artist, title, issue date) from the New Yorker archives to enrich `get_document_text()`
- `voyage_multimodal_3_5` is the go-to embedding preset for image datasets (used in both met_museum and zappos configs)

## Exit criteria
- [ ] `src/multiview/docsets/newyorker_covers.py` exists and loads images from the scraped directory
- [ ] Registered in `__init__.py`
- [ ] Criteria defined in `available_criteria.yaml`
- [ ] `configs/corpus_new_yorker_covers.yaml` exists
- [ ] `python scripts/analyze_corpus.py --config-name corpus_new_yorker_covers` runs without config errors
