# Task: Add Paris Review interviews docset
loprio because paywall

## Goal
Create a new docset for Paris Review interviews (literary interviews with authors).

## Type
NEW_DOCSET

## Notes from original task
- "Paris Review interviews" — listed alongside museum exhibitions and New Yorker cartoons as new data sources

## Data source
- The Paris Review: https://www.theparisreview.org/interviews
- May require scraping or finding an existing dataset
- Each document = one interview excerpt or Q&A segment

## Reference files
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/dickinson.py` — scraping + caching pattern
- `src/multiview/docsets/__init__.py` — register

## Steps
- [ ] Research whether Paris Review interview data is available (API, dataset, or requires scraping)
- [ ] Create `src/multiview/docsets/paris_review.py` with class `ParisReviewDocSet`
- [ ] Register in `__init__.py`
- [ ] Add criteria to `configs/available_criteria.yaml`:
  - `interview_style`: What kind of interview is this? (craft-focused, biographical, philosophical, etc.)
  - `literary_sensibility`: What literary values/aesthetics does the interviewee express?

## Exit criteria
- [ ] Docset loads interview data
- [ ] Pipeline runs with at least one criterion
