# Multiview Roadmap

## Where we are

Infrastructure is solid: inference caching, 10 eval methods (5 production, 5 experimental), 40+ docsets, quality-validated triplet generation, production React visualizer with 7 render modes. 17+ days of benchmark runs.

The gap: all of that plumbing hasn't been turned into findings, a public artifact, or a trained model yet.

## Phase 1: Systematic method comparison

**Goal:** Know which methods work on which criteria types.

**Pick ~6 representative docset/criterion pairs:**

| Criteria type | Docset | Criterion |
|---------------|--------|-----------|
| Factual/extractive | gsm8k | arithmetic |
| Subjective/aesthetic | haiku | meaning_evoked |
| Visual/multimodal | met_museum | subject_matter |
| Structural | crossword_clues | clue_type |
| Narrative | rocstories | seven_basic_plots |
| Humor/subjective | onion_headlines | joke_type |

**Deliverable:** Results matrix — method x criteria type → accuracy. This is the core finding.

**Entry points:**
- Create `configs/benchmark_method_comparison.yaml` based on `benchmark_fuzzy.yaml`
- `scripts/create_eval.py` then `scripts/run_eval.py`

**Parallelizable:** Yes — each docset/criterion/method combo is independent. Spawn one agent per config.

---

## Phase 4: Train a criteria-conditional embedding model

**Goal:** A model that embeds documents differently depending on the criterion.

**Depends on:** Benchmark to evaluate against, findings to know what to beat.

- [ ] **Triplet export** — script to export quality-validated triplets as (anchor, pos, neg, criterion_description) tuples in SentenceTransformers format
  - Source: `src/multiview/benchmark/triplets/quality_assurance.py`
- [ ] **Criterion-conditioned fine-tuning** — prepend criterion description to each doc before embedding
  - Base model: `nomic-embed-text-v1.5` or similar
  - Loss: `MultipleNegativesRankingLoss` or `TripletLoss`
  - Criterion description acts as task instruction that steers the embedding
- [ ] **Evaluate** — train/test split on triplets, show improvement over base model
- [ ] **Zero-shot transfer** — train on criteria A, B, C → evaluate on unseen criterion D
- [ ] **Document rewriter variant** — instead of conditioning the embedder, train an LM to rewrite documents to expose a specific criterion (see `loprio-add_grit.md` notes)

**Deliverable:** A model. Evaluated on the benchmark. Ideally with zero-shot transfer results.

---

## Running in parallel

The four phases are sequential in terms of dependencies, but:
- **Within each phase**, work is highly parallelizable (separate agents per docset/method/feature)
- **Dataset expansion** (`todo/b01-b20`) runs in parallel with everything — more data only helps
- **Triplet export script** (Phase 4) can be built during Phase 1
- **Visualizer work** (Phase 2) is independent of backend eval runs

## Priority tiers for todo/ tasks

**Critical path (feeds the roadmap):**
- `c-comparison_plots.md` → Phase 2
- `f-scale_leaderboard.md` → Phase 3
- `a-taxonomy.md` → Phase 1 (taxonomy detail finding)
- `loprio-add_grit.md` → Phase 4

**High value (expand the benchmark):**
- `b01-dickinson_criteria.md` — new criteria for rich literary corpus
- `b06-met_criteria.md` — multimodal criteria expansion
- `b08-new_yorker_cartoons.md` — humor + visual, compelling demo
- `b20-newyorker_covers.md` — 5K images across a century, strong visual benchmark

**Medium value (fill out coverage):**
- `b05-hackernews_criteria.md`, `b07-goodreads_criteria.md`, `b09-tao_te_ching.md`
- `b14-aidanbench_eval.md`, `b16-mmlu_taxonomy.md`
- `b18-auto_tsne.md` — infrastructure improvement

**Speculative (needs more definition):**
- `b10-bitext_mining.md`, `b11-borges_shakespeare.md`, `b13-abstract_graphs.md`
- `b15-paris_review.md`, `b17-similes_by_vibe.md`
- `b19-overflow_ideas.md` — parking lot
