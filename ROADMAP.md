# Multiview Benchmark Roadmap

## 🎯

- [x] Generate 256 triplets for (GSM8K, arithmetic_operations) criterion and evaluate with 4 methods to compare model performance.

- [x] Scale to 10 different docset-criteria combinations for a comprehensive benchmark suite.

- [ ]
Use triplet data for downstream tasks like fine-tuning better embedding models.

---
# Medium term

## ✅ Schema Comparison Feature (IMPLEMENTED)

**Goal**: Compare in_one_word and pseudologit methods with different schema configurations

**Status**: ✅ Implemented (see `SCHEMA_COMPARISON.md` for usage guide)

**Implementation**:
- Config: `configs/benchmark_schema_comparison.yaml`
- Evaluation methods support 4 schema modes:
  1. **No schema**: Generic categorization prompt
  2. **Oracle schema**: Schema from annotations (lm_all)
  3. **Proposed schema**: Dynamically generated fresh schema with `num_trials` support
  4. **Custom schema**: Manually specified categories/classes file (supports per-task dict format)
- Statistics: `compute_trial_statistics()` utility for mean/std across trials

**Supported Methods**:
- ✅ In one word (no schema)
- ✅ In one word using oracle schema (whatever was actually proposed by lm_tag)
- ✅ In one word proposed schema over N trials (oracle schema is withheld)
- ✅ Pseudologit with oracle schema (whatever was actually proposed by lm_tag)
- ✅ Pseudologit proposed schema over N trials (oracle schema is withheld)

**Running Schema Comparison**:
```bash
# Single run with automatic multi-trial execution
# The config includes num_trials: 10 for proposed schema methods
uv run python -m multiview.benchmark.run configs/benchmark_schema_comparison.yaml

# Results will include separate rows for each trial:
# - inoneword_proposed_trial1, inoneword_proposed_trial2, ..., inoneword_proposed_trial10
# - pseudologit_proposed_trial1, pseudologit_proposed_trial2, ..., pseudologit_proposed_trial10

# Compute statistics across trials using pandas or compute_trial_statistics()
```

**Key Features**:
- `num_trials` parameter: Automatically runs N trials with unique cache keys
- Trial differentiation: Each trial gets unique `_trial_idx` in config hash
- Reproducibility: Trials use same document sample but different LLM generations

See `SCHEMA_COMPARISON.md` for full documentation.


## 📋 LONG TERM: Fine-tuning & Advanced Features

**Goal**: Use triplet data for fine-tuning embedding models
**Status**: Infrastructure not yet built
**ETA**: 5-8 weeks (phased approach)

- [ ] **Visualization tools**
  - [ ] Implement UMAP/t-SNE dimensionality reduction
  - [ ] Create 2D scatter plot visualization
  - [ ] Create 3D interactive Plotly widget
  - [ ] Add per-criterion color coding
  - [ ] Add hover tooltips with document text

- [ ] **Embedding fine-tuning**
  - [ ] Implement contrastive loss training
  - [ ] Add hard negative mining during training
  - [ ] Support multiple backbone models
  - [ ] Add validation metrics
  - [ ] Implement early stopping

- [ ] **Query rewriter optimization (GEPA)**
  - [ ] Integrate DSPy for prompt optimization
  - [ ] Implement evolutionary search
  - [ ] Add triplet accuracy objective
  - [ ] Configure search parameters

- [ ] **Training infrastructure**
  - [ ] Set up training data loaders
  - [ ] Add distributed training support (multi-GPU)
  - [ ] Implement checkpointing
  - [ ] Add WandB/TensorBoard logging
  - [ ] Create evaluation callback

- [ ] **Model evaluation**
  - [ ] Test fine-tuned model on held-out tasks
  - [ ] Compare to base model performance
  - [ ] Measure improvement in triplet accuracy
  - [ ] Check for overfitting/generalization

- [ ] **Generate analysis report**
  - [ ] Identify which methods perform best overall
  - [ ] Identify which criteria are hardest/easiest
  - [ ] Note any surprising results or failure modes
  - [ ] Document method rankings across tasks

### Success Criteria

- [ ] Fine-tuned model outperforms base model by >5% on held-out tasks
- [ ] Training pipeline is reproducible
- [ ] Model can be exported and used in production
- [ ] Documentation covers full training workflow
