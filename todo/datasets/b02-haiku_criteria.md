# Task: Verify haiku criteria coverage

## Goal
Confirm that haiku criteria from Task B are already covered, or add missing ones.

## Type
ADD_CRITERIA (likely no-op)

## Current state
- Docset exists: `src/multiview/docsets/haiku.py` (333K haiku from HuggingFace)
- Existing criteria in `configs/available_criteria.yaml` under `haiku:`:
  - `imagery` — "The images featured in the haiku" (maps to Task B's "literal images")
  - `meaning_evoked` — "The greater sense of meaning that the haiku evokes" (maps to Task B's "evoked meaning")
  - `poem_composition` — structural composition

## What Task B asked for
- "Haiku literal images" → already covered by `imagery`
- "Haiku evoked meaning" → already covered by `meaning_evoked`

## Reference files
- `configs/available_criteria.yaml` — haiku section (~line 177)
- `src/multiview/docsets/haiku.py`

## Steps
- [ ] Verify `imagery` and `meaning_evoked` criteria have sufficient hints for good triplet generation
- [ ] Run eval if not already done: `python scripts/create_eval.py` with haiku + imagery
- [ ] Run eval if not already done: `python scripts/create_eval.py` with haiku + meaning_evoked

## Exit criteria
- [ ] Haiku criteria confirmed working or improved
- [ ] At least one eval run exists in `outputs/` for haiku with these criteria
