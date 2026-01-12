# [WIP] multiview

<img src="assets/glasses.jpg" width="200">

[WIP] A benchmark for criteria-specific semantic similarity.

## Motivation
Standard embedding models are trained to prioritize **query-document relevance** and **general semantic similarity**.

This benchmark evaluates performance on **"specific" semantic similarity** tasks.

Examples:
- ROCStories based on their alignment with Christopher Booker's Seven Basic Plots
- ROCStories based on the explicit or implied setting
- Crossword clues based on topics or subject matter
- Crossword clues based on the way that they hint at their answer (regardless of topics or subject matter)
- GSM8K math problems based on the arithmetic structure of their solutions

    - A motivating triplet from [Prismatic Synthesis](https://nvlabs.github.io/prismatic-synthesis/):

        - <img src="https://nvlabs.github.io/prismatic-synthesis/assets/motivation.png" width="300">

**Overall, the benchmark emphasizes the ability to represent multiple different "views" of the same data.**

## Technical contribution
We implement a pipeline to create criteria-conditioned triplets from unlabeled document sets. 

Given a criteria and set of unlabeled documents:
1. Use a strong LM judge to reflect on how the criteria relates to the documents
2. Annotate each document with its relationship to the criteria using (a) a categorical assignment (b) boolean tags and (c) freetext summary
3. Randomly select triplet anchors and create triplets.
    - For each anchor, use jaccard similarity over tags to propose true positive candidates. Use an LM judge to select a true positive from the candidates.
    - Use an LM judge to select a hard negative from lexically similar candidates (based on BM25 scores)
4. Discard low quality triplets with an LM judge

The data-generation method is general-purpose and scaling-friendly. It is designed to generalize to arbitrary documents/criteria by using a stronger LM judge.

For document sets with available annotation data, we also collect and repurpose data from pre-existing labeled datasets to create additional "gold" triplets.

Currently supported document sets and criteria can be viewed in `src/multiview/docsets/criteria_metadata.py` and `src/multiview/docsets`.

## Usage
```bash
# before running, please set up uv env 
# and update CACHE_ROOT in `src/multiview/constants.py` to a valid local path
python scripts/run_eval.py --config-name benchmark
```

## Example triplet for crossword clue type

```
    "anchor": "Clue: Old-time baseballer Maglie\nAnswer: sal",
    "positive": "Clue: Belushi's \"Blues Brothers\" costar\nAnswer: aykroyd",
    "negative": "Clue: Caesar s\nAnswer: cciii",
```
## Example triplet for GSM8K arithmetic structure
```
    "anchor": "Question: Pierre decides to bring his mother out for ice cream. His treat. Each scoop is $2. If he gets 3 scoops and his mom gets 4, what is the total bill?\nAnswer: His ice cream is $6 because 3 x 2 = <<3*2=6>>6\nHis mom's is $8 because 4 x 2 = <<4*2=8>>8\nThe total is $14 because 6 + 8 = <<6+8=14>>14\n#### 14",
    "positive": "Question: Tim gets 6 hours of sleep 2 days in a row.  To make up for it he sleeps 10 hours the next 2 days.  How much sleep did he get?\nAnswer: He got 2*6=<<2*6=12>>12 hours the first 2 days.\nThe next 2 days he gets 2*10=<<2*10=20>>20 hours.\nSo he got 12+20=<<12+20=32>>32 hours.\n#### 32",
    "negative": "Question: Darryl is an inventor who just designed a new machine. He had to pay $3600 for the parts to construct the machine, and $4500 for the patent he applied for once he built it. If the machine sells for $180, how many machines does Darryl need to sell to break even after the costs?\nAnswer: Darryl paid 3600 + 4500 = $<<3600+4500=8100>>8100 in costs for the parts and patent.\nThus, Darryl needs to sell 8100 / 180 = <<8100/180=45>>45 machines to break even.\n#### 45",
```
## Example results for GSM8K task

```
================================================================================
EVALUATION RESULTS SUMMARY
================================================================================

Task: gsm8k__arithmetic
  gemini_flash_triplet_with_annotation: 98.81% (250/253 correct)
  gemini_flash_triplet_no_annotation : 91.27% (230/253 correct)
  bm25_lexical                       : 10.28% (26/253 correct)
  qwen3_embedding_8b_with_instructions         : 12.65% (32/253 correct)
  qwen3_embedding_8b_no_instructions           : 13.44% (34/253 correct)
================================================================================
```

- Standard LM judges are able to trivially distinguish positives from negatives
- Lexical similarity and embedding models struggle, even when instructions include the criteria description

## AI code assistance
Core logic was written by hand with Cursor, then later refactored with Claude code (especially for generating tests). Coding agents were especially useful for adding support for new document sets.
