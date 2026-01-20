# Multiview: scalable evaluations for conditional semantic similarity

## Overview
- Semantic similarity is **criteria-dependent**.
    - Consider this example from [Prismatic Synthesis](https://nvlabs.github.io/prismatic-synthesis/). Sample A and B are more similar with respect to arithmetic structure (distance formula), but Sample A and C are more similar with respect to the word problem's narrative setup (Bill walking).
         <div align="center">
           <img src="https://nvlabs.github.io/prismatic-synthesis/assets/motivation.png" width="300">
         </div>

- A performant instruction-tuned embedding model
should be able to produce representations
based on **arbitrary criteria**,
but in practice, this capability is limited.
- In this benchmark, we evaluate a model's ability to represent multiple different 'views' of the same document.
- Tasks include:
    - GSM8K problems by arithmetic structure
    - GSM8K problems by narrative setup
    - Crossword clues by answer topic
    - Crossword clues by clue type
    - Haiku by literal images
    - Haiku by meaning evoked
- See [TASK_TABLE.md](TASK_TABLE.md) for additional examples.

## Related work
- Conditional semantic similarity is a canonical task in computer vision and NLP, but existing benchmarks focus primarily on simple conditions and sentence-length texts.
- We build upon existing evaluations to include more complex documents and criteria, focusing on tasks where criteria *cannot* be trivially extracted into a single word or phrase.
- See [Related Work](sections/0_related_work.md) for references, including a discussion of instructed retrieval benchmarks.

## Technical artifacts
1. **Data generation pipeline**. A general-purpose data generation pipeline that creates criteria-conditional triplets from unlabeled document sets. Using more capable LM judges, it can scale to complex documents and criteria. See [Data Generation](sections/1_data_gen.md).
2. **Benchmark**. A benchmark of different document sets and criteria, which includes
data generated via the above method as well as data adapted from pre-existing sources.
3. **Leaderboard**. Evaluation results for instruction-tuned embedding models, rerankers, embedding models with query expansion, and additional baselines. See [Scoring Methods](sections/2_scoring_methods.md) and [Results](sections/3_results.md).

See [Discussion](sections/4_discussion.md).

## Future work
- Future work could scale this evaluation to even more realistic documents and criteria.

- The triplet generation pipeline is also an effective way to generate **training data**, which could be used to finetune instruction-tuned embedding models via standard contrastive losses.
Most methods were "spikey" in that they were very performant for some tasks and not for others, suggesting that ensembling the best of all methods and validating with an LM could be a way to create a good dataset.
(Note: An ideal embedding model would produce document embeddings that contain all of this information without needing the run the full model for every new instruction.
To that end, it could be interesting to tune a small MLP or linear linear, or a hypernetwork.)

- However, when moving beyond toy criteria, it seems difficult to train performant embedding models without including a generative reasoning component. It could also be interesting to finetune dedicated "document rewriters" for use with bm25 or a frozen embedding model. (Triplet correctness could serve as a reward signal for prompt optimization with GEPA or tuning with RL.
In this view, the criteria description and schema is **privileged information** used by the "verifier".
The resulting models would be akin to "advisor models" for the downstream embedding models ([Asawa 2025](https://arxiv.org/abs/2510.02453)).
