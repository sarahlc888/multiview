# Task: Add themes criteria to Goodreads quotes

## Goal
Add a `themes` criterion to the goodreads_quotes docset.

## Type
ADD_CRITERIA

## Current state
- Docset exists: `src/multiview/docsets/goodreads_quotes.py`
- No criteria currently defined in `configs/available_criteria.yaml` for goodreads (no `goodreads:` or `goodreads_quotes:` section)

## What to add
From Task B: "goodreads_quotes - don't do positive sum thing... instead do themes"
- **themes**: The thematic content of the quote — what it's about at a level deeper than topic (love, mortality, ambition, self-knowledge, etc.)

## Reference files
- `src/multiview/docsets/goodreads_quotes.py` — read to understand document format
- `configs/available_criteria.yaml` — add new `goodreads_quotes:` section
- Look at how `DATASET_NAME` maps to the yaml key (check `__init__.py` registry key vs class DATASET_NAME)

## Steps
- [ ] Read `goodreads_quotes.py` to understand the DATASET_NAME value
- [ ] Add `goodreads_quotes:` section to `configs/available_criteria.yaml`
- [ ] Add `themes` criterion with description and hints
- [ ] Run eval to verify

## Exit criteria
- [ ] `goodreads_quotes:` section with `themes` criterion exists in available_criteria.yaml
- [ ] Eval runs without errors
