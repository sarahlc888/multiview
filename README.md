# Multiview: scalable evaluations for conditional semantic similarity

- Standard embedding models prioritize general notions of semantic similarity.
- This benchmark evaluates embedding models on **criteria-specific** semantic similarity.

See [WRITEUP.md](writeup/WRITEUP.md) for more information.

## Usage
1. Set up
    ```
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```
2. Update `CACHE_ROOT` in `src/multiview/constants.py` to a valid local path
### Prepare triplets for evaluation
```bash
python scripts/create_eval.py --config-name benchmark
```
### Run evaluation
```bash
python scripts/run_eval.py --config-name benchmark
```
## Benchmark format

Given a set of documents and a freetext criteria, we create triplets `(anchor, positive, negative)` for which the anchor document is more similar to the positive than negative, in the context of the criteria.

Our evaluation metric is accuracy rate over triplets.

For each document set, we create triplets for at least two different criteria.


## Examples
See [TASK_TABLE.md](TASK_TABLE.md) for example triplets for all tasks.

## Development

- Overview
    - `Benchmark: List[Task]`
    - `Task: List[Triplet]`
    - `Triplet: Tuple[String]` composed of `(anchor, positive, negative)`

### View triplets
- Show random triplets from documents
    ```bash
    pytest tests/benchmark/test_triplet_utils.py::test_create_random_triplets"[arxiv_abstract_sentences]" -vs
    ```
- Show triplets from labeled data
    ```bash
	pytest tests/benchmark/test_quality_validation.py::test_prelabeled_triplets -k  abstractsim -vs
	pytest tests/benchmark/test_quality_validation.py::test_prelabeled_triplets -k  infinite_prompts -vs
	```
### Validate triplets
- Validate that triplets adapted from labeled data are high quality
    ```bash
    pytest tests/benchmark/test_quality_validation.py::test_prelabeled_triplets -k infinite_prompts -vs --run-external
    ```
### To add a new criteria for an existing document set
Update `available_tasks.yaml` with the criteria name, criteria description, and optional "hints" for the LM judge.

### To add a new document set
Create a new file in `docsets/`
