# Task: Add Tao Te Ching docset

## Goal
Create a new docset for Tao Te Ching passages (Ursula K. Le Guin translation), enabling analysis of philosophical/spiritual themes.

## Type
NEW_DOCSET

## Data source
- GitHub: https://github.com/nrrb/tao-te-ching/blob/master/Ursula%20K%20Le%20Guin.md
- 81 chapters/verses, each a short passage
- Markdown format, parse into individual chapters

## Reference files
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/dickinson.py` — similar pattern: scrape/parse text, cache locally
- `src/multiview/docsets/__init__.py` — register new class
- `configs/available_criteria.yaml` — add criteria

## Steps
- [ ] Create `src/multiview/docsets/tao_te_ching.py` with class `TaoTeChingDocSet`
  - Fetch the markdown from GitHub, parse into 81 chapters
  - Cache locally under `~/.cache/multiview/tao_te_ching/`
  - Each document = one chapter/verse
- [ ] Register in `src/multiview/docsets/__init__.py`
- [ ] Add `tao_te_ching:` section to `configs/available_criteria.yaml` with criteria:
  - `philosophical_theme`: What philosophical concept does this passage address? (wu wei, the Tao, simplicity, leadership, etc.)
  - `teaching_strategy`: How does the passage convey its teaching? (paradox, metaphor, negation, direct instruction, etc.)
- [ ] Run eval to verify

## Exit criteria
- [ ] `src/multiview/docsets/tao_te_ching.py` exists and loads 81 passages
- [ ] Registered in `__init__.py`
- [ ] Criteria in available_criteria.yaml
- [ ] Pipeline runs without errors
