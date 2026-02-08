# Task: Run and improve AidanBench evaluation

## Goal
Run the aidanbench evaluation benchmark with filtering for higher-quality (longer) responses.

## Type
RUN_EVAL

## Current state
- Docset exists: `src/multiview/docsets/aidanbench.py`
- Registered in `__init__.py` as `"aidanbench": AidanBenchDocSet`

## Notes from original task
- "quickly run the aidanbench one"
- "responses are too toy though...? maybe filter for examples where the response is a bit longer??"
- "it should provide a perfume taxonomy, why isn't it?"

## Reference files
- `src/multiview/docsets/aidanbench.py` — read to understand current filtering/config
- `scripts/run_eval.py` — entry point for running evaluation
- `scripts/create_eval.py` — entry point for creating triplets

## Steps
- [ ] Read `aidanbench.py` to understand document format and any existing filters
- [ ] Add a `min_response_length` config parameter to filter out toy/short responses
- [ ] Run eval: `python scripts/create_eval.py` with aidanbench
- [ ] Investigate the perfume taxonomy issue — why isn't it generated?
- [ ] Run `python scripts/run_eval.py` with results

## Exit criteria
- [ ] Eval runs with filtered (longer) responses
- [ ] Perfume taxonomy issue diagnosed
- [ ] Results in `outputs/`
