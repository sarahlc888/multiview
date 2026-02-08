# Task: Verify crossword criteria coverage

## Goal
Confirm that crossword criteria from Task B are already covered, or add missing ones.

## Type
ADD_CRITERIA (likely no-op)

## Current state
- Docset exists: `src/multiview/docsets/crossword_clues.py` (CrosswordQA from HuggingFace)
- Existing criteria in `configs/available_criteria.yaml` under `crossword:`:
  - `topic` — domain/subject matter of the answer
  - `clue_type` — archetype of the clue (wordplay, straight def, cryptic, etc.)
  - `type_of_challenge` — what makes the clue difficult

## What Task B asked for
- "Crossword topic" → already covered by `topic`
- "Crossword clue type" → already covered by `clue_type`

## Reference files
- `configs/available_criteria.yaml` — crossword section (~line 59)
- `src/multiview/docsets/crossword_clues.py`

## Steps
- [ ] Verify criteria have sufficient hints (they already have detailed tag_schema_hint)
- [ ] Run eval if not already done

## Exit criteria
- [ ] Crossword criteria confirmed working
- [ ] At least one eval run exists in `outputs/`
