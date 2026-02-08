# Task: Apply taxonomy criteria to MMLU

## Goal
Add taxonomy-based criteria to the existing MMLU docset.

## Type
ADD_CRITERIA

## Current state
- Docset exists: `src/multiview/docsets/mmlu.py`
- Registered in `__init__.py` as `"mmlu": MMLUDocSet`
- Check what criteria currently exist in available_criteria.yaml

## Notes from original task
- "apply some taxonomy to some corpus - mmlu"
- Related to the taxonomy detail → performance finding (Task A)

## Reference files
- `src/multiview/docsets/mmlu.py` — read to understand document format
- `configs/available_criteria.yaml` — check if `mmlu:` section exists, add if not

## Steps
- [ ] Read `mmlu.py` to understand document structure
- [ ] Check if `mmlu:` section exists in available_criteria.yaml
- [ ] Add taxonomy-based criteria (e.g., `subject_taxonomy`, `reasoning_type`)
- [ ] Run eval

## Exit criteria
- [ ] MMLU criteria defined and eval runs
