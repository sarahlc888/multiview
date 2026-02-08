# Task: Add bitext mining docset (Schopenhauer-Nietzsche)

## Goal
Create a "deep semantic bitext mining" docset that finds conceptual parallels between quotes from two authors — Schopenhauer and Nietzsche.

## Type
NEW_DOCSET

## Notes from original task
- "matches are sparser" — unlike standard bitext mining, conceptual parallels are rare
- This is a harder task because it requires understanding philosophical concepts across different writing styles
- Consider the "know it when you see it" edge case — very sparse matches

## Data source
- Need to find/scrape collections of Schopenhauer and Nietzsche quotes
- Possible sources: Project Gutenberg, Wikiquote, existing HuggingFace datasets
- Each document = one quote/passage from either author

## Reference files
- `src/multiview/docsets/base.py` — BaseDocSet pattern
- `src/multiview/docsets/dickinson.py` — example of scraping + caching
- `src/multiview/docsets/__init__.py` — register
- `configs/available_criteria.yaml` — criteria

## Steps
- [ ] Research best data source for Schopenhauer and Nietzsche quotes/passages
- [ ] Create `src/multiview/docsets/bitext_mining.py` with class `BitextMiningDocSet`
  - Load quotes from both authors
  - Each document should include author attribution
  - `get_document_text()` returns the quote text
- [ ] Register in `__init__.py`
- [ ] Add `bitext_mining:` section to `configs/available_criteria.yaml`:
  - `conceptual_parallel`: Do these quotes address the same philosophical concept, even in different ways?
  - `philosophical_stance`: What position does the quote take? (pessimism, will-to-power, aesthetic, ethical, etc.)
- [ ] Run eval — expect lower accuracy due to sparse matches

## Exit criteria
- [ ] Docset loads quotes from both authors
- [ ] Pipeline runs; triplet quality may be lower (expected for sparse match scenario)
