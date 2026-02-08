# Multiview: scalable evaluations for conditional semantic similarity

Standard embedding models prioritize general notions of semantic similarity. **Multiview evaluates how well embedding models can represent similarity according to specific criteria.**

For example, two math word problems might be similar in their arithmetic structure (both use the distance formula) but different in their narrative setup (one about walking, another about driving). A good instruction-tuned embedding model should be able to represent *both* views depending on what you ask for.

**This benchmark measures that capability.** Given a corpus of documents and a criteria (e.g., "arithmetic structure" or "narrative setup"), Multiview generates evaluation triplets and tests whether models can distinguish relevant matches from irrelevant ones.


---


## Overview

- Semantic similarity is **criteria-dependent**.
    - Consider this example from [Prismatic Synthesis](https://nvlabs.github.io/prismatic-synthesis/). Sample A and B are more similar with respect to arithmetic structure (distance formula), but Sample A and C are more similar with respect to the word problem's narrative setup (Bill walking).
         <div align="center">
           <img src="https://nvlabs.github.io/prismatic-synthesis/assets/motivation.png" width="300">
         </div>
- A performant instruction-tuned embedding model
should be able to produce representations
conditioned on **arbitrary criteria**.
- In practice, this capability is limited.
- We evaluate how well representations reflect similarity in the context of a specific criteria/instruction. Specifically, we measure the ability to represent multiple different 'views' of the same document.
- Given a **corpus of unlabeled documents** and **a freetext criteria**, `multiview` will...
    - Annotate each document with how it relates to the criteria
    - Use the annotations to create criteria-specific triplets `(anchor, positive, negative)`
    - Evaluate how well embedding models, rerankers, and LMs can distinguish positives from negatives
- Tasks include:
    - GSM8K problems by arithmetic structure
    - GSM8K problems by narrative setup
    - Crossword clues by answer topic
    - Crossword clues by clue type
    - Haiku by literal images
    - Haiku by meaning evoked
    - ...
    - See [TASKS_TABLE.md](../TASKS_TABLE.md) for additional examples.

## Method
Given a **corpus of unlabeled documents** and **a freetext criteria**, the pipeline:
1. Annotates each document with how it relates to the criteria (using LM judges)
2. Creates criteria-specific triplets `(anchor, positive, negative)` where anchor-positive share the criteria better than anchor-negative
3. Evaluates how well embedding models, rerankers, and LMs can distinguish positives from negatives

**Evaluation metric**: Accuracy rate over triplets.

**Example tasks**:
- GSM8K problems by arithmetic structure vs. narrative setup
- Crossword clues by answer topic vs. clue type
- Haiku by literal imagery vs. philosophical meaning
- Code snippets by algorithm vs. domain
- See [TASKS_TABLE.md](../TASKS_TABLE.md) for the full benchmark.

## Related work
- Conditional semantic similarity is a canonical task in computer vision and NLP. However, existing benchmarks focus primarily on simple conditions and sentence-length texts.
- We build upon existing evaluations to include more complex documents and criteria, and we focus on tasks where criteria *cannot* be trivially extracted into a single word or phrase.
- See [Related Work](sections/0_related_work.md) for references, including a discussion of instructed retrieval benchmarks.

## Technical artifacts

1. **Data generation pipeline**: Scalable method for creating criteria-conditional triplets from unlabeled documents. Can be adapted to new domains and used to generate training data. See [Data Generation](sections/1_data_gen.md).

2. **Benchmark suite**: 15+ document types with 2-4 criteria each, including data generated via our pipeline and data adapted from existing sources.

3. **Baseline results**: Evaluation of instruction-tuned embedding models, rerankers, query expansion methods, and LM-based scorers. Current models show inconsistent performance across criteria. See [Scoring Methods](sections/2_scoring_methods.md) and [Results](sections/3_results.md).


## Future work
- Future work could scale this evaluation to even more realistic documents and criteria.

- The triplet generation pipeline is also an effective way to generate **training data**, which could be used to finetune instruction-tuned embedding models via standard contrastive losses.
Most methods were "spikey" in that they were very performant for some tasks and not for others, suggesting that ensembling the best of all methods and validating with an LM could be a way to create a good dataset.
(Note: An ideal embedding model would produce document embeddings that contain all of this information without needing the run the full model for every new instruction.
To that end, it could be interesting to tune a small MLP or linear linear, or a hypernetwork.)

- However, when moving beyond toy criteria, it seems difficult to train performant embedding models without including a generative reasoning component. It could also be interesting to finetune dedicated "document rewriters" for use with bm25 or a frozen embedding model. (Triplet correctness could serve as a reward signal for prompt optimization with GEPA or tuning with RL.
In this view, the criteria description and schema is **privileged information** used by the "verifier".
The resulting models would be akin to "advisor models" for the downstream embedding models ([Asawa 2025](https://arxiv.org/abs/2510.02453)).
