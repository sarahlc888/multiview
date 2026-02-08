- We could pour more compute on and get better triples




- [ ] ==write it all up and get a proper leaderboard==
- [ ] eval on the prelabeled data for all of these extractive properties.... (==prelabeled stuff gives us a bunch of well vetted data for free....==)
- [ ] run aidanbench as an eval benchmark





I mean what is the hold up

One thing is that there is a limit to what it generalizes ... there are instances where the data looks like shit and that's pretty blocker-ey

- [ ] Leaderboard
- [ ] Good Triplets - 200 triplets for each document set
- [ ] Automatic table showing an anchor positive and negative of each generated thing. To make it just really clear what it is that we’re talking about
# Multiview Roadmap

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
