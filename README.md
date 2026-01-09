# multiview

<img src="assets/glasses.jpg" width="200">


## Evaluation structure
A task is defined by `documents: List[str]` and `criterion: str`.

## Usage

```bash
python scripts/run_eval.py
# dev
pytest tests -vs  
```

Configure tasks in `configs/benchmark.yaml`.