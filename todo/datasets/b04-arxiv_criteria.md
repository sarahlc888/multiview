# Task: Verify arxiv_cs criteria coverage

## Goal
Confirm that arxiv criteria from Task B are already covered, or add missing ones.

## Type
ADD_CRITERIA (likely no-op)

## Current state
- Docset exists: `src/multiview/docsets/arxiv_cs.py`
- Existing criteria in `configs/available_criteria.yaml` under `arxiv_cs:`:
  - `topic` — subfield and technical insights (maps to "science topic")
  - `research_sensibility` — type of intellectual work (maps to "approach")
  - `core_contribution` — main intellectual advance

## What Task B asked for
- "Arxiv CS abstract science topic" → covered by `topic`
- "Arxiv CS abstract 'approach'" → covered by `research_sensibility`
- "a good science abstractions thing -- run the pipeline and see what kind of triplets we get"

## Reference files
- `configs/available_criteria.yaml` — arxiv_cs section (~line 153)
- `src/multiview/docsets/arxiv_cs.py`
- `src/multiview/prompts/custom/arxiv_topic.txt` — custom prompt for topic
- `src/multiview/prompts/custom/arxiv_research_sensibility.txt`
- `src/multiview/prompts/custom/arxiv_core_contribution.txt`

## Steps
- [ ] Run the pipeline with arxiv_cs + topic to inspect triplet quality
- [ ] Run the pipeline with arxiv_cs + research_sensibility
- [ ] Check if "implicit taxonomy" idea (from Task B notes) warrants a new criterion

## Exit criteria
- [ ] ArXiv criteria confirmed working with good triplet quality
- [ ] Eval runs exist in `outputs/`
