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

TODO add screenshot here

TODO add leaderboard here

```bash
python scripts/analyze_corpus.py --config-name corpus_new_yorker_covers
```
