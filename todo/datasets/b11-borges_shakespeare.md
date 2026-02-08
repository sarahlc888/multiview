# Task: Add Borges and Shakespeare distribution docsets

## Goal
Create docsets for literary analysis of Borges and Shakespeare — exploring the "distribution" of themes, references, and touchstones in their work.

## Type
NEW_DOCSET

## Notes from original task
- "a person has a finite set of points of reference"
- "there is such a thing as a set of cultural touchstones"
- DEMO: show distribution over borges; over shakespeare
- shakespeare-anything, borges-anything

## Data source
- Shakespeare: Project Gutenberg, Folger Shakespeare Library, or HuggingFace datasets
- Borges: Harder to source (copyright). May need to use quotes, summaries, or analysis excerpts
- Each document = passage, scene, or story excerpt

## Reference files
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/dickinson.py` — scraping + literary text pattern
- `src/multiview/docsets/__init__.py` — register

## Steps
- [ ] Research available Shakespeare text datasets (prefer structured by play/scene)
- [ ] Research available Borges text (may be limited due to copyright — consider using story summaries or critical analyses)
- [ ] Create `src/multiview/docsets/shakespeare.py` with class `ShakespeareDocSet`
- [ ] Create `src/multiview/docsets/borges.py` with class `BorgesDocSet` (or combine into `literary_distributions.py`)
- [ ] Register in `__init__.py`
- [ ] Add criteria to available_criteria.yaml:
  - `recurring_reference`: What cultural/literary touchstones does this passage reference?
  - `thematic_preoccupation`: What recurring theme does this passage embody?

## Exit criteria
- [ ] At least one docset (Shakespeare) loads and runs through the pipeline
- [ ] Borges docset created if suitable data source found (may need to punt if copyright issues)
