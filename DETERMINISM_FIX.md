# Determinism Fix for Cache Misses

## Problem
Cache misses were occurring for negative judgments despite identical configuration:
```
14:42:08 [INFO] Cache hits: 150/150 prompts @ ...judge_pos...  ✅
14:42:08 [INFO] Running 148 uncached @ ...judge_neg...        ❌
```

## Root Cause
`np.argsort()` was using **unstable sort** (quicksort), which reordered tied elements non-deterministically.

When BM25/similarity scores had ties (very common with 0.0 scores):
- Candidate order would change between runs
- Different candidate order → different prompts → different hashes → cache miss

This affected negatives more than positives because:
- Negative candidates use raw BM25 scores (many zero scores)
- Spurious similarity scores often cluster around 0.0
- More ties = more non-determinism

## Solution
Added `kind='stable'` to all `np.argsort()` calls (18 locations):
- `src/multiview/benchmark/triplets/candidate_selection.py` (3 calls)
- `src/multiview/benchmark/triplets/triplet_utils.py` (14 calls)
- `src/multiview/visualization/reducers.py` (1 call)

Stable sort ensures tied elements maintain their original index order.

## Verification

### 1. Cost Tracker Enhancements
Added functions to track LM calls globally:
```python
from multiview.inference import cost_tracker

cost_tracker.reset()
# ... run benchmark ...
requests = cost_tracker.get_total_requests()
cache_hits = cost_tracker.get_total_cache_hits()
```

### 2. Test Suite
Created `tests/benchmark/test_determinism.py` with 10 comprehensive tests:

**Unit Tests (fast):**
```bash
uv run pytest tests/benchmark/test_determinism.py -v
```
- Stable sort verification
- BM25 candidate selection determinism
- Jaccard candidate selection determinism
- Spurious hard negatives determinism
- RRF merge determinism
- Prompt hash consistency
- Candidate ordering with tied scores

**Integration Tests (with API calls):**
```bash
uv run pytest tests/benchmark/test_determinism.py --run-external -v
```
- `test_cache_hits_on_second_run` - Basic cache verification
- `test_end_to_end_cache_with_reuse_false` - **Critical: Verifies cache works even with `reuse_cached_triplets=false`**
- `test_multiple_datasets_cache_consistency` - Tests cache across multiple datasets

### 3. Test Results
All tests pass ✅:

**Unit tests** (7 tests, 0.14s):
```
test_argsort_stability
test_bm25_candidate_selection_deterministic
test_jaccard_candidate_selection_deterministic
test_spurious_hard_negatives_deterministic
test_rrf_merge_deterministic
test_prompt_hash_consistency
test_candidate_ordering_with_tied_scores
```

**Integration tests** (3 tests):
```
test_cache_hits_on_second_run (102s)
  First run  - Requests: 148, Cache hits: 0
  Second run - New requests: 0, New cache hits: 148  ✅

test_end_to_end_cache_with_reuse_false (132s)
  First run  - Requests: 52, Cache hits: 0
  Second run - New requests: 0, New cache hits: 52   ✅
  Proves cache works even with reuse_cached_triplets=false

test_multiple_datasets_cache_consistency (234s)
  crossword_clues/topic: 37 requests → 0 requests (cached) ✅
  gsm8k/arithmetic: 38 requests → 0 requests (cached) ✅
```

## Other Non-Determinism Sources Checked

All verified safe ✅:
- Set operations (only used for set arithmetic, sorted when displayed)
- Dict iteration (Python 3.7+ maintains insertion order)
- Random operations (all use seeded RNGs)
- File system operations (no glob/listdir in benchmark code)
- Hash operations (not used in benchmark code)
- ThreadPoolExecutor (preserves order by iterating futures list)
- RRF merging (uses stable Python sorted())
- Floating point comparisons (deterministic)
- JSON serialization (uses sort_keys=True)

## Usage

Run your benchmark again - you should now see cache hits for both positives AND negatives:
```bash
uv run python scripts/run_eval.py
```

Expected output:
```
[INFO] Cache hits: 150/150 prompts @ ...judge_pos...  ✅
[INFO] Cache hits: 148/148 prompts @ ...judge_neg...  ✅
```

## Testing in Your Workflow

Add this to your scripts to verify cache behavior:
```python
from multiview.inference import cost_tracker

# At start
cost_tracker.reset()

# Run your pipeline
task = Task(config)
task.load_documents()
task.annotate_documents()
task.create_triplets()

# Verify
requests = cost_tracker.get_total_requests()
cache_hits = cost_tracker.get_total_cache_hits()
print(f"API calls: {requests}, Cache hits: {cache_hits}")

# On second run, requests should be 0 and cache_hits should be > 0
```
