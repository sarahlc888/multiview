# Task: Add New Yorker Caption Contest docset

## Goal
Create a new docset for New Yorker caption contest data, enabling analysis of humor by overt and covert features.

## Type
NEW_DOCSET

## Data source
- https://nextml.github.io/caption-contest-data/
- Contains cartoon descriptions + submitted captions + ratings
- Original framing: NYCC "moves" — "What is the angle?"
- "embed based on overt and covert features"
- Note: "a Dashboard is really necessary to show off the cool factor" — viz/dashboard support matters here

## Reference files
- `src/multiview/docsets/base.py` — BaseDocSet abstract class (implement `load_documents()`, `get_document_text()`)
- `src/multiview/docsets/haiku.py` — simple text docset example
- `src/multiview/docsets/onion_headlines.py` — similar humor-analysis domain, good criteria reference
- `src/multiview/docsets/__init__.py` — register new class in DOCSETS dict + imports
- `configs/available_criteria.yaml` — add criteria section

## Steps
- [ ] Explore the caption contest data format (download sample, check fields)
- [ ] Create `src/multiview/docsets/new_yorker_cartoons.py` with class `NewYorkerCartoonsDocSet`
  - DATASET_PATH, DESCRIPTION, DOCUMENT_TYPE
  - `load_documents()` — fetch/cache data from nextml source
  - `get_document_text()` — format as caption text (+ optional cartoon description context)
- [ ] Register in `src/multiview/docsets/__init__.py` (import + add to DOCSETS + __all__)
- [ ] Add `new_yorker_cartoons:` section to `configs/available_criteria.yaml` with criteria:
  - `humor_move`: What comedy strategy does the caption use? (similar to onion's `joke_type`)
  - `overt_features`: What's literally happening in the cartoon/caption
  - `covert_features`: What's the implicit joke or commentary

## Exit criteria
- [ ] New file `src/multiview/docsets/new_yorker_cartoons.py` exists and follows BaseDocSet pattern
- [ ] Registered in `__init__.py`
- [ ] Criteria defined in available_criteria.yaml
- [ ] Can run `python scripts/create_eval.py` with new_yorker_cartoons + humor_move
