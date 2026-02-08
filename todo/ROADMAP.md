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

**Run all 10 methods on each:**
- embeddings (OpenAI, Voyage, HF)
- lm_judge (bidirectional)
- bm25
- reranker
- pseudologit
- in_one_word
- document_summary
- multisummary
- query_relevance_vectors
- lm_judge_pair

**Deliverable:** Results matrix — method x criteria type → accuracy. This is the core finding.

**Entry points:**
- Create `configs/benchmark_method_comparison.yaml` based on `benchmark_fuzzy.yaml`
- `scripts/create_eval.py` then `scripts/run_eval.py`

**Parallelizable:** Yes — each docset/criterion/method combo is independent. Spawn one agent per config.

---

## Phase 2: Corpus map / multi-criteria visualizer

**Goal:** Same documents, reorganized by different criteria. Toggle between views.

**Builds on:** Phase 1 results + existing visualizer + `todo/c-comparison_plots.md`

**Features to add:**
- [ ] Criterion selector in visualizer (currently only method selector exists)
- [ ] Side-by-side view: same corpus under 2 criteria simultaneously
- [ ] Pareto frontier plot: accuracy on criterion A vs. accuracy on criterion B, per method
- [ ] Correlation heatmap: how correlated are different criteria's embeddings for a given corpus?
- [ ] "Guided tour" mode: walk through 3-4 compelling toggles (e.g., haiku by imagery vs. by meaning_evoked)

**Key files:**
- `visualizer/src/App.tsx` — main app (1322 lines), add criterion selector
- `visualizer/src/render/ScatterPlot.tsx` — scatter plot, needs dual-criterion support
- `visualizer/src/render/HeatmapView.tsx` — extend for correlation
- `viz/` — already structured per task/method, extend to per-criterion

**Deliverable:** Interactive demo: load a corpus, toggle between "sort by meaning" / "sort by structure" / "sort by imagery."

---

## Phase 3: Ship publicly

**Goal:** A URL you can share.

**Depends on:** Findings (Phase 1) and demo (Phase 2).

- [ ] **Leaderboard site** — static site (GitHub Pages) showing method x criteria matrix
  - See `todo/f-scale_leaderboard.md`
- [ ] **Writeup** — blog post or paper draft:
  - Problem: criteria-specific similarity is underserved by existing embedding models
  - Benchmark: triplet-based evaluation with quality validation
  - Findings: which methods work where (from Phase 1)
  - Demo: link to visualizer (from Phase 2)
- [ ] **Package** — `pip install multiview-bench` for others to evaluate their models
  - `pyproject.toml` already exists
- [ ] **README** — fill in TODO placeholders (screenshots, leaderboard link)

**Deliverable:** Public URL — leaderboard, blog post, or both.

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
