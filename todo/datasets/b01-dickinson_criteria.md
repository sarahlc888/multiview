# Task: Add new Dickinson criteria

## Goal
Add poetry-specific criteria to the existing Dickinson docset that capture what a poem *does* beyond its theme and imagery.

## Type
ADD_CRITERIA

## Current state
- Docset exists: `src/multiview/docsets/dickinson.py` (scrapes Project Gutenberg, ~200 poems)
- Existing criteria in `configs/available_criteria.yaml` under `dickinson:`: `poetic_theme`, `specific_imagery`
- Several commented-out criteria exist (philosophical_stance, emotional_register, structural_strategy, stylistic_signature)

## What to add
New criteria from the original notes:
- **role_of_line**: "What role does each line play in the poem?" — how individual lines function (sets scene, delivers turn, creates tension, resolves, questions, etc.). Original note also says "Does similar work" — i.e., group poems whose lines play similar functional roles.
- **evocation**: "What does the poem evoke?" — the feeling/atmosphere/association that lingers, distinct from literal meaning
- **what_hovers**: "What hovers at the edges?" — what the poem suggests without stating, the unsaid resonance

## Reference files
- `src/multiview/docsets/dickinson.py` — existing docset loader
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `configs/available_criteria.yaml` — add criteria definitions here (see `dickinson:` section ~line 208)
- `src/multiview/docsets/criteria_metadata.py` — how criteria metadata gets loaded from YAML

## Steps
- [ ] Add `role_of_line` criterion to `configs/available_criteria.yaml` under `dickinson:` with description and hints
- [ ] Add `evocation` criterion with description and hints
- [ ] Add `what_hovers` criterion with description and hints
- [ ] Consider whether any of the commented-out criteria (philosophical_stance, emotional_register, etc.) should be uncommented

## Exit criteria
- [ ] New criteria appear in `configs/available_criteria.yaml` under `dickinson:`
- [ ] Can run `python scripts/create_eval.py` with dickinson + new criterion without errors
