# Multiview: scalable evaluations for conditional semantic similarity

Standard embedding models prioritize general notions of semantic similarity. This benchmark evaluates embedding models on **criteria-specific** semantic similarity.

See [WRITEUP.md](writeup/WRITEUP.md) for more information.

## Example triplet
- Document set: machine-generated haikus from TODO
- Criteria: `**poem_composition**<br/>The relationship between the haiku's parts. What is the composition of the poem? Does the haiku have a 'turn'? Does the last line make a general pronouncement or a question?`
- Anchor
    ```
    weeds choke the garden / yet bloom with defiant grace / beauty in the low
    ```
- Positive
    ```
    starry night's vastness / infinite and dark and deep / human soul's smallness
    ```
- Negative
    ```
    tailor's voice is low / as he speaks to the fabric / coaxing it to form
    ```

See [TASKS_TABLE.md](TASKS_TABLE.md) for example triplets for all tasks.



## Benchmark format

Given a set of documents and a freetext criteria, we create triplets `(anchor, positive, negative)` for which the anchor document is more similar to the positive than negative, in the context of the criteria.

Our evaluation metric is accuracy rate over triplets.

For each document set, we create triplets for at least two different criteria.

## Usage
1. Set up
    ```
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```
2. Update `CACHE_ROOT` in `src/multiview/constants.py` to a valid local path
### Create triplets for evaluation
```bash
python scripts/create_eval.py --config-name benchmark
```
### Run evaluation
```bash
python scripts/run_eval.py --config-name benchmark
```
### View results
After running `create_eval.py`, a `viewer.html` file is automatically generated alongside each `triplets.json`:

```bash
# Open viewer for a specific task
open outputs/benchmark_fuzzy_debug/triplets/haiku__poem_composition__tag__50/viewer.html

# Or find all viewers
find outputs -name "viewer.html"
```

The viewer displays:
- All triplets (anchor/positive/negative) with color coding
- Document text content (scrollable for long documents)
- Images (for vision datasets like ut_zappos50k)
- Quality assessments with ratings and reasoning
- Metadata for each document

The viewer is a standalone HTML file with all data embedded - no server required!

- **Visualize Embeddings**

  For exploring document embeddings and triplet relationships in 2D space, use the interactive visualizer:

  ```bash
  cd visualizer
  npm install
  npm run dev
  ```

  The visualizer shows embeddings with multiple reducers (t-SNE, UMAP, SOM, PCA) and supports triplet highlighting.

## Development

- Primitives
    - `Benchmark: List[Task]`
    - `Task: List[Triplet]`
    - `Triplet: Tuple[String]` composed of `(anchor, positive, negative)`
- Note: All API calls are cached. Re-running evaluations costs $0 after first run.

### View triplets

- Show random triplets from documents
    ```bash
    pytest tests/benchmark/test_triplet_utils.py::test_create_random_triplets"[arxiv_abstract_sentences]" -vs
    pytest tests/benchmark/test_triplet_utils.py::test_create_random_triplets"[arxiv_cs]" -vs
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
