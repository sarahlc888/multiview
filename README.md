# multiview

<img src="assets/glasses.jpg" width="200">

A benchmark for criteria-specific semantic similarity.

Standard embedding models are trained to prioritize **query-document relevance** and **general semantic similarity**.

This benchmark evaluates performance on **"specific" semantic similarity** tasks.

Example task:
TODO

**Overall, the benchmark emphasizes the ability to represent multiple different "views" of the same data.**

The method is general purpose, scalable, and ??TODO??.

See [WRITEUP.md](WRITEUP.md) for a blog-style summary.


## Usage

Run benchmark:
```bash
python scripts/run_eval.py
```

### Development
```bash
pytest tests -vs
```

## Tasks
A benchmark task is defined by `documents: List[str]` and `criterion: str`.

[TODO discuss more: It creates a set of triplets... Can work for generic documents]
