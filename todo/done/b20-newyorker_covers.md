# Task: Add New Yorker covers docset ✅

## Completed
Created image-based docset for New Yorker magazine covers (~5,101 images spanning 1925–2025).

## Data source
- Scraped directory: `/Users/sarahchen/code/pproj/scrape/newyorker_covers/`
- Structure: `{year}/{hex_id}.jpg` (101 year directories, 1925–2025)
- 320x437 baseline JPEGs

## Implementation
- `src/multiview/docsets/newyorker_covers.py` — image docset with year metadata
- Registered in `src/multiview/docsets/__init__.py`
- Criteria: `visual_style`, `subject_matter`, `cultural_moment`, `composition`
- Config: `configs/corpus_new_yorker_covers.yaml`

## Details
See docstring in `src/multiview/docsets/newyorker_covers.py` for usage and configuration.
