<h1 align="center">Multiview</h1>

<p align="center"><b>Scalable evaluations for conditional semantic similarity</b></p>


Standard embedding models prioritize general notions of semantic similarity. This benchmark evaluates embedding models on **criteria-specific** semantic similarity.

## Installation

```
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start
Update `CACHE_ROOT` in `src/multiview/constants.py` to a valid local path

Create triplets for evaluation:
```bash
python scripts/create_eval.py --config-name benchmark
```

Run evaluation:
```bash
python scripts/run_eval.py --config-name benchmark
```

Analyze corpus:
```bash
python scripts/analyze_corpus.py --config-name benchmark
```

View results:
```bash
cd visualizer
npm install
npm run dev
```

## Example

### Side-by-side of two criteria
<img src="writeup/assets/sidebyside.png" alt="Side by side example" width="600"/>

### Leaderboard for triplet agreement
TODO add leaderboard here

## Commands for replicating experiments
### Triplets
- TODO
    ```bash
    uv run python scripts/create_eval.py --config-name benchmark_zappos_100
    uv run python scripts/run_eval.py --config-name benchmark_zappos_100
    uv run python scripts/analyze_corpus.py --config-name benchmark_zappos_100
    cd visualizer
    npm install
    npm run dev
    ```
- GSM8K problems
    ```bash
    python3 scripts/create_eval.py --config-name benchmark_gsm8k_schema_comparison
    python3 scripts/run_eval.py --config-name benchmark_gsm8k_schema_comparison
    ```
### Corpus
- *New Yorker* covers
    ```bash
    python scripts/analyze_corpus.py --config-name corpus_new_yorker_covers
    ```
- TODO
    ```bash
    python scripts/analyze_corpus.py --config-name benchmark_haiku_100
    ```
- TODO
    ```bash
    python scripts/analyze_corpus.py --config-name benchmark_crossword_100
    ```
- TODO
    ```bash
    python scripts/analyze_corpus.py --config-name benchmark_dickinson_100
    ```
- TODO
    ```bash
    python scripts/analyze_corpus.py --config-name benchmark_hn_100
    ```
- TODO
    ```bash
    python scripts/analyze_corpus.py --config-name benchmark_met_100
    ```
- TODO
    ```bash
    python scripts/analyze_corpus.py --config-name benchmark_gsm8k_100
    ```
- TODO
    ```bash
    # TODO
    ```
