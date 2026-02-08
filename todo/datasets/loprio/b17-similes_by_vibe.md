# Task: Add similes-by-vibe dataset

## Goal
Create or find a dataset of similes, with criteria based on their "vibe" / aesthetic quality.

## Type
NEW_DOCSET

## Notes from original task
- "similes by vibe" — listed under "more better data"

## Data source
- Need to research: HuggingFace simile datasets, literary corpora, or generate from existing text
- Each document = one simile

## Reference files
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/haiku.py` — similar short-text literary docset

## Steps
- [ ] Research available simile datasets
- [ ] Create `src/multiview/docsets/similes.py` with class `SimilesDocSet`
- [ ] Register in `__init__.py`
- [ ] Add criteria: `vibe` (the aesthetic/emotional quality of the simile)

## Exit criteria
- [ ] Docset loads similes and pipeline runs
