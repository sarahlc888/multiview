# Task: Add Whole Earth Catalog docset

## Goal
Create an image-based docset from scraped Whole Earth Catalog page/cover images (~147 images from Archive.org), enabling visual and cultural analysis of the counterculture publication (1968–1998).

## Type
NEW_DOCSET

## Data source
- Already scraped: `/Users/sarahchen/code/pproj/bounding_bosch/scraping_data/data/whole_earth/`
- `output/images_run2.json` — 147 image entries with Archive.org URLs and index
- `input/scraped.html` — source HTML from Archive.org
- Scraper: `src/whole_earth.py`
- Images are Archive.org `cover_medium.jpg` files from various Whole Earth Catalog/Review editions

## Reference files
- `src/multiview/docsets/met_museum.py` — image docset with download/cache pattern, closest reference
- `src/multiview/docsets/newyorker_covers.py` — another cover-image docset (if implemented)
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/__init__.py` — register
- `configs/available_criteria.yaml` — add criteria section

## Steps
- [ ] Create `src/multiview/docsets/whole_earth_catalog.py` with class `WholeEarthCatalogDocSet`
  - Load from `images_run2.json`
  - Download/cache images locally from Archive.org URLs
  - `KNOWN_CRITERIA = []` — no deterministic labels beyond what's in the URL
  - Each document: `{"image_path": str, "text": str, "source_url": str, "index": int}`
  - `get_document_text(doc)` — return minimal text (edition name extracted from URL if possible)
  - `get_document_image(doc)` — return local cached image path
- [ ] Register in `__init__.py`
- [ ] Add `whole_earth_catalog:` section to `configs/available_criteria.yaml`:
  - `visual_style`: Cover design style (photographic, illustrated, collage, typographic, etc.)
  - `era`: What period does this edition represent (early counterculture, 70s, 80s revival, 90s digital, etc.)?
  - `subject_focus`: What is the cover's primary subject (tools, ecology, technology, community, space, etc.)?
- [ ] Create `configs/corpus_whole_earth_catalog.yaml`
- [ ] Run eval and visually inspect triplets

## Notes
- Small dataset (147 images) — good for quick iteration and testing
- The Whole Earth Catalog is culturally significant (Stewart Brand, counterculture → tech culture pipeline)
- Archive.org URLs may need respectful rate limiting when downloading
- Could extract edition names/years from the Archive.org URL slugs for metadata enrichment
- Consider pairing with a text-based Whole Earth dataset if full catalog text becomes available

## Exit criteria
- [ ] Docset loads and caches images from Archive.org
- [ ] Pipeline generates triplets from the cover images
- [ ] Visual style criterion produces sensible groupings across the catalog's 30-year span
