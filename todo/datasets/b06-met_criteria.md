# Task: Expand MET Museum criteria

## Goal
Add new criteria to the MET Museum docset covering influence, expression, form, content, and animacy — creating richer multimodal evaluation dimensions.

## Type
ADD_CRITERIA

## Current state
- Docset exists: `src/multiview/docsets/met_museum.py` (Met API, image-based, cached locally)
- Existing criteria in `configs/available_criteria.yaml` under `met_museum:`:
  - `subject_matter` — literal content depicted
  - `visual_composition` — structure, color, lighting, perspective
  - `period_or_movement` — historical/artistic context

## What to add
From Task B notes:
- **animacy**: How animate/alive does the artwork feel? Static vs. dynamic, presence of living beings
- **form**: Shape, medium, physical form of the artwork (overlaps with visual_composition — define carefully)
- **content**: What the artwork is "about" at a deeper level than subject_matter (narrative, allegory)
- **expression**: Emotional expression in portraits — what feelings are conveyed
- **influence_network**: "artworks by common ancestors (hidden influence network)" — which artworks seem to share stylistic DNA or lineage
- **sparseness**: "Paintings based on what is unusual about them" — what makes a painting stand out, unusual features
- **number_of_people**: "Portraits based on number of people in it" — solo portrait vs. group, how many figures
- **use_of_color**: "Portraits based on use of color" — color palette choices, dominant hues, contrast

## Reference files
- `configs/available_criteria.yaml` — met_museum section (~line 407)
- `src/multiview/docsets/met_museum.py` — note: image-based docset with `get_document_image()`
- `src/multiview/docsets/base.py` — BaseDocSet pattern

## Steps
- [ ] Add `animacy` criterion to available_criteria.yaml with description and hints
- [ ] Add `expression` criterion (consider scoping to portraits via config filter)
- [ ] Add `influence_network` criterion with description emphasizing stylistic lineage
- [ ] Add `sparseness` / `unusual_features` criterion
- [ ] Evaluate whether `form` and `content` are sufficiently distinct from existing `visual_composition` and `subject_matter`
- [ ] Run eval with new criteria

## Exit criteria
- [ ] New criteria in `configs/available_criteria.yaml` under `met_museum:`
- [ ] Can run `python scripts/create_eval.py` with met_museum + new criterion
