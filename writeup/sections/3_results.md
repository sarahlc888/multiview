# Results


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

## **Run**: benchmark_fuzzy_debug

---

## crossword_clues__topic__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 4.00% | 2 | 50 | — |
| voyage_rerank_lite | 62.00% | 31 | 50 | — |
| dr_gemini_lite_openai_small | 72.00% | 36 | 50 | — |
| gemini_flash_triplet | 90.00% | 90 | 100 | — |
| qwen3_8b_with_instructions | 54.00% | 27 | 50 | +0.0800 |
| qwen3_8b_no_instructions | 46.00% | 23 | 50 | — |

## crossword_clues__type_of_challenge__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 12.00% | 6 | 50 | — |
| voyage_rerank_lite | 48.00% | 24 | 50 | — |
| dr_gemini_lite_openai_small | 58.00% | 29 | 50 | — |
| gemini_flash_triplet | 84.00% | 84 | 100 | — |
| qwen3_8b_with_instructions | 30.00% | 15 | 50 | -0.0400 |
| qwen3_8b_no_instructions | 34.00% | 17 | 50 | — |

## crossword_clues__clue_type__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 10.00% | 5 | 50 | — |
| voyage_rerank_lite | 60.00% | 30 | 50 | — |
| dr_gemini_lite_openai_small | 60.00% | 30 | 50 | — |
| gemini_flash_triplet | 81.00% | 81 | 100 | — |
| qwen3_8b_with_instructions | 52.00% | 26 | 50 | +0.1800 |
| qwen3_8b_no_instructions | 34.00% | 17 | 50 | — |

## gsm8k__problem_type__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 8.00% | 4 | 50 | — |
| voyage_rerank_lite | 70.00% | 35 | 50 | — |
| dr_gemini_lite_openai_small | 86.00% | 43 | 50 | — |
| gemini_flash_triplet | 97.00% | 97 | 100 | — |
| qwen3_8b_with_instructions | 82.00% | 41 | 50 | +0.4400 |
| qwen3_8b_no_instructions | 38.00% | 19 | 50 | — |

## gsm8k__arithmetic__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 0.00% | 0 | 50 | — |
| voyage_rerank_lite | 48.00% | 24 | 50 | — |
| dr_gemini_lite_openai_small | 58.00% | 29 | 50 | — |
| gemini_flash_triplet | 100.00% | 100 | 100 | — |
| qwen3_8b_with_instructions | 30.00% | 15 | 50 | +0.0600 |
| qwen3_8b_no_instructions | 24.00% | 12 | 50 | — |

## haiku__imagery__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 6.00% | 3 | 50 | — |
| voyage_rerank_lite | 76.00% | 38 | 50 | — |
| dr_gemini_lite_openai_small | 82.00% | 41 | 50 | — |
| gemini_flash_triplet | 96.00% | 96 | 100 | — |
| qwen3_8b_with_instructions | 72.00% | 36 | 50 | -0.1200 |
| qwen3_8b_no_instructions | 84.00% | 42 | 50 | — |

## haiku__meaning_evoked__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 2.00% | 1 | 50 | — |
| voyage_rerank_lite | 30.00% | 15 | 50 | — |
| dr_gemini_lite_openai_small | 78.00% | 39 | 50 | — |
| gemini_flash_triplet | 91.00% | 91 | 100 | — |
| qwen3_8b_with_instructions | 38.00% | 19 | 50 | +0.0600 |
| qwen3_8b_no_instructions | 32.00% | 16 | 50 | — |

## haiku__poem_composition__tag__50

| Method | Accuracy | Correct | Total | Instr. Sensitivity |
|--------|----------|---------|-------|--------------------|
| bm25_lexical | 0.00% | 0 | 50 | — |
| voyage_rerank_lite | 18.00% | 9 | 50 | — |
| dr_gemini_lite_openai_small | 56.00% | 28 | 50 | — |
| gemini_flash_triplet | 88.00% | 88 | 100 | — |
| qwen3_8b_with_instructions | 18.00% | 9 | 50 | -0.0400 |
| qwen3_8b_no_instructions | 22.00% | 11 | 50 | — |
