# Task: Add CARI aesthetics docset

## Goal
Create a docset from the Consumer Aesthetics Research Institute (CARI) data — ~90 named aesthetic movements (synthwave, vaporwave, nu-brutalism, etc.) with ~3,185 associated images scraped from Are.na galleries, plus inter-aesthetic similarity edges.

## Type
NEW_DOCSET

## Data source
- Already scraped: `/Users/sarahchen/code/pproj/bounding_bosch/scraping_data/data/cari/`
- `aesthetics.csv` — 90 aesthetics with name, URL slug, and similar-aesthetics list
- `images.csv` — 3,185 images with URL, aesthetic name, title, source link
- `galleries/*.json` — per-aesthetic gallery JSONs from Are.na (id, title, description, image URLs at thumb/display/original sizes)
- Scraper: `src/extract_cari.py`

## Reference files
- `src/multiview/docsets/met_museum.py` — image docset with `get_document_image()`, local caching
- `src/multiview/docsets/ut_zappos50k.py` — another image docset
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/__init__.py` — register
- `configs/available_criteria.yaml` — add criteria section

## Steps
- [ ] Create `src/multiview/docsets/cari_aesthetics.py` with class `CARIAestheticsDocSet`
  - Load images from gallery JSONs or images.csv
  - Download/cache images locally from Are.na URLs
  - `KNOWN_CRITERIA = ["aesthetic_name"]` — deterministic from the labeled aesthetic category
  - Each document: `{"image_path": str, "text": str, "aesthetic_name": str, "title": str}`
  - `get_document_text(doc)` — return title + aesthetic name + description if available
  - `get_document_image(doc)` — return local cached image path
- [ ] Register in `__init__.py`
- [ ] Add `cari_aesthetics:` section to `configs/available_criteria.yaml`:
  - `aesthetic_name`: Which named aesthetic movement does this image belong to?
  - `visual_style`: What visual techniques/elements are used (neon, pastels, geometric, organic, etc.)?
  - `era`: What decade/era does this aesthetic evoke?
  - `medium`: What medium is this (poster, album cover, interior design, product, web design, etc.)?
- [ ] Create `configs/corpus_cari_aesthetics.yaml`
- [ ] Run eval and visually inspect triplets

## Notes
- The similar-aesthetics graph in `aesthetics.csv` is a natural ground truth for evaluating embeddings — aesthetics listed as "similar" should cluster
- With 90 labeled categories this is excellent for classification-style eval
- Images are hosted on Are.na CDN — will need to download and cache locally

## Exit criteria
- [ ] Docset loads images and generates triplets
- [ ] `aesthetic_name` criterion produces sensible groupings
- [ ] Triplets respect visual similarity (e.g., synthwave closer to vaporwave than to eco-beige)
