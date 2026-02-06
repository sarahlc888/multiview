# Missing Features: Multiview (Current Audit)

**Audit Date**: 2026-02-06
**Reference Project**: `/Users/sarahchen/code/pproj/project/` (s3 package)
**Current Project**: `/Users/sarahchen/code/pproj/multiview/`

---

## Executive Summary

The codebase now has a real evaluation stack and visualization tooling, but the core learning component is still missing. In short:

- Evaluation exists: multiple methods in `/Users/sarahchen/code/pproj/multiview/src/multiview/eval/`, orchestration in `/Users/sarahchen/code/pproj/multiview/src/multiview/benchmark/benchmark.py`, and reporting in `/Users/sarahchen/code/pproj/multiview/scripts/run_eval.py`.
- Visualization exists in `/Users/sarahchen/code/pproj/multiview/src/multiview/visualization/` and `/Users/sarahchen/code/pproj/multiview/scripts/visualize_corpus.py`.
- The biggest missing piece is a **training/learning module** for a multiview representation model.
- Metrics are **triplet-accuracy centric**; corpus/tuple correlation metrics, NDCG, and CIs are not implemented.

---

## Critical Missing Features

### 1. Training / Multiview Model Learning
**Status**: ❌ Missing entirely
**Priority**: 🔴 CRITICAL

**Missing Components**:
- `models/` defining a multiview encoder
- Training loop and losses (triplet/contrastive)
- Dataloaders from triplets and annotations
- Checkpointing and reproducible training configs

**Impact**: The project can evaluate external models but cannot learn its own multiview representation.

---

### 2. Grading + Metrics Beyond Triplet Accuracy
**Status**: ⚠️ Partial
**Priority**: 🔴 CRITICAL

**Current State**:
- Triplet accuracy metrics exist in `/Users/sarahchen/code/pproj/multiview/src/multiview/benchmark/evaluation_utils.py`.

**Missing Components**:
- Rank correlations (Spearman, Kendall, Pearson)
- NDCG for ranking-style tasks
- Bootstrap confidence intervals
- Cross-task aggregation (per-criterion, per-docset)

**Impact**: Results are hard to compare across methods or tasks beyond simple accuracy.

---

### 3. Corpus/Tuple Scoring Modes + Dataframe Scoring API
**Status**: ❌ Missing
**Priority**: 🔴 CRITICAL

**Current State**:
- Evaluation focuses on triplet outcomes only.

**Missing Components**:
- Corpus-level and tuple-level similarity scoring
- A dataframe-style `score_df()` API for adding similarity scores

**Impact**: No standardized way to compute correlation metrics over similarity matrices or anchor neighborhoods.

---

### 4. Results Aggregation and Export
**Status**: ⚠️ Partial
**Priority**: 🔴 CRITICAL

**Current State**:
- Per-method logs and embeddings can be saved, but aggregation is minimal.

**Missing Components**:
- Per-criterion summaries and unified benchmark reports
- Export utilities for clean results tables
- Consistent result metadata (config hash, model versions)

**Impact**: Hard to run repeatable, comparable benchmark suites.

---

## High Priority Missing Features

### 5. Local Model Cache + Smart Batching
**Status**: ⚠️ Basic
**Priority**: 🟡 HIGH

**Current State**:
- Disk caching exists, but in-memory reuse and batching are limited.

**Missing Components**:
- Persistent model caching with eviction policies
- Dynamic batch sizing and dedup across requests

**Impact**: Local inference is slower and more expensive than necessary.

---

### 6. Local Provider Coverage (vLLM, MLX)
**Status**: ⚠️ Partial
**Priority**: 🟡 HIGH

**Current Providers**:
- OpenAI, Anthropic, Gemini, Voyage
- Hugging Face API
- Local HF inference (`hf_local.py`, `hf_local_colbert.py`)

**Missing vs s3**:
- vLLM completions and embeddings
- MLX completions and embeddings

**Impact**: Local, low-cost, high-throughput inference is still limited.

---

### 7. Visualization Gaps
**Status**: ⚠️ Partial
**Priority**: 🟡 HIGH

**Current State**:
- Visualization utilities exist, but are not tied into standard benchmark reporting.

**Missing Components**:
- Graph/community visualization for similarity networks
- Standardized report artifacts (consistent HTML or notebook outputs)

**Impact**: Debugging embeddings and presenting results is still ad hoc.

---

### 8. Configuration Breadth
**Status**: ⚠️ Basic
**Priority**: 🟡 HIGH

**Current State**:
- Limited configs in `/Users/sarahchen/code/pproj/multiview/configs/`.

**Missing Components**:
- Wider, structured config set for evaluation suites and method variants

**Impact**: Reproducibility and large-scale runs are harder than they should be.

---

## Medium Priority Missing Features

### 9. Data Synthesis and Augmentation
**Status**: ⚠️ Partial
**Priority**: 🟠 MEDIUM

**Current State**:
- `benchmark/synthesis/` exists, but coverage is limited.

**Missing Components**:
- Hard negative remixing
- Counterfactual generation
- Multiple rewrite strategies with caching

---

### 10. Export Formats + Dataset Versioning
**Status**: ⚠️ Basic
**Priority**: 🟠 MEDIUM

**Current State**:
- JSONL and NPY outputs exist via artifacts utilities.

**Missing Components**:
- CSV and Parquet export
- HuggingFace dataset export
- Dataset versioning metadata

---

### 11. Dataset Parity with s3
**Status**: ⚠️ Partial
**Priority**: 🟠 MEDIUM

**Current State**:
- Multiview already includes many docsets (e.g., `gsm8k`, `rocstories`, `d5`, `hackernews`, `moralfables`, `fewrel`, `mmlu`).

**Likely Missing vs s3**:
- `chatbotarena`
- `conceptnet`
- `ifbench`
- `llmjepa`

---

## Low Priority Missing Features

### 12. VLM Support
**Status**: ⚠️ Partial
**Priority**: 🔵 LOW

**Current State**:
- Example image docset exists, but no general VLM evaluation pipeline.

---

### 13. Enhanced Logging and Reporting
**Status**: ⚠️ Basic
**Priority**: 🔵 LOW

**Current State**:
- Python logging and some basic summaries exist.

**Missing Components**:
- Structured experiment logs and richer console outputs

---

## Feature Comparison Matrix (Updated)

| Feature Category | Multiview | s3 Project | Priority |
|-----------------|-----------|------------|----------|
| Core Data Pipeline | ✅ Complete | ✅ Complete | - |
| Annotations | ✅ Complete | ✅ Complete | - |
| Triplet Generation | ✅ Strong | ✅ Strong | - |
| Scoring/Evaluation (triplet) | ✅ Present | ✅ Complete | - |
| Metrics/Grading (rank, NDCG, CI) | ❌ Missing | ✅ Complete | 🔴 CRITICAL |
| Benchmark Runner | ✅ Present | ✅ Complete | - |
| Visualization | ✅ Present | ✅ Complete | 🟡 HIGH |
| Training / Finetune | ❌ Missing | ✅ Complete | 🔴 CRITICAL |
| Local Model Coverage | ⚠️ Partial | ✅ Complete | 🟡 HIGH |
| Model Cache + Batching | ⚠️ Basic | ✅ Advanced | 🟡 HIGH |
| Data Synthesis | ⚠️ Partial | ✅ Complete | 🟠 MEDIUM |
| Export + Versioning | ⚠️ Basic | ✅ Complete | 🟠 MEDIUM |
| Dataset Variety | ✅ Strong | ✅ Strong | 🟠 MEDIUM |
| Config Management | ⚠️ Basic | ✅ Extensive | 🟡 HIGH |
| VLM Support | ⚠️ Partial | ✅ Complete | 🔵 LOW |
| Logging/Reporting | ⚠️ Basic | ✅ Polished | 🔵 LOW |

---

## Recommended Implementation Order

### Phase 1: Core Missing Heart
1. Implement a training module with a multiview model, triplet loss, and checkpoints.
2. Add rank/correlation metrics, NDCG, and confidence intervals.
3. Add corpus/tuple scoring modes and a dataframe scoring API.

### Phase 2: Usability and Scale
1. Build results aggregation and export utilities.
2. Add model cache and smart batching.
3. Expand local providers (vLLM, MLX).

### Phase 3: Quality and Coverage
1. Expand data synthesis strategies.
2. Broaden configs for reproducible benchmark suites.
3. Improve VLM support and richer reporting.
