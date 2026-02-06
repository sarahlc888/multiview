# Schema Comparison Feature

This document explains how to compare `in_one_word` and `pseudologit` evaluation methods under different schema configurations, as outlined in the ROADMAP.md medium-term goal.

## Overview

The schema comparison feature allows you to test how method performance varies with different category schemas:

1. **In one word (no schema)** - Generic categorization without specific categories
2. **In one word (oracle schema)** - Using the schema generated during annotation
3. **In one word (proposed schema)** - Using freshly generated schemas (oracle withheld)
4. **Pseudologit (oracle schema)** - Using the schema from annotations
5. **Pseudologit (proposed schema)** - Using freshly generated schemas

## Key Concepts

### Four Schema Modes

#### 1. Oracle Schema
The **oracle schema** is the schema that was generated and used during the annotation phase. This schema is:
- Generated from a deterministic sample of documents
- Stored in `task.document_annotations` as `category_schema` (from `lm_all` triplet style) or `tag_schema` (from `lm_tags` triplet style)
- Used as the "ground truth" for comparison
- Works with both `lm_all` (category schemas) and `lm_tags` (tag schemas)

#### 2. Proposed Schema
A **proposed schema** is a category schema generated freshly during evaluation, without access to the oracle. This tests:
- Whether evaluation methods are sensitive to the specific schema chosen
- How robust methods are to schema variation
- Whether the oracle schema was particularly good/bad

#### 3. Custom Schema
A **custom schema** is a manually specified schema (e.g., from domain knowledge or prior research). This allows:
- Testing performance with expert-designed schemas
- Comparing against hand-crafted taxonomies
- Ablation studies with controlled schema variations

#### 4. No Schema
The **no schema** mode uses a generic prompt without specifying categories. This tests whether explicit categories improve performance.

## Quick Start

### Running a Schema Comparison

```bash
# Full comparison (100 docs, 50 triplets)
uv run python -m multiview.benchmark.run configs/benchmark_schema_comparison.yaml

# Quick test (fewer docs for faster results)
# Edit the config to set max_docs: 50, max_triplets: 25
uv run python -m multiview.benchmark.run configs/benchmark_schema_comparison.yaml
```

### Expected Output

The benchmark will generate results showing accuracy for each method. When using `num_trials` parameter, each trial gets a separate row:

```
Method                             | Accuracy
-----------------------------------|----------
In one word (no schema)            | 0.XX
In one word (oracle schema)        | 0.XX
In one word (proposed)_trial1      | 0.XX
In one word (proposed)_trial2      | 0.XX
...                                | ...
In one word (proposed)_trial10     | 0.XX
Pseudologit (oracle schema)        | 0.XX
Pseudologit (proposed)_trial1      | 0.XX
Pseudologit (proposed)_trial2      | 0.XX
...                                | ...
Pseudologit (proposed)_trial10     | 0.XX
Baseline: Qwen3 8B embedding       | 0.XX
Baseline: BM25                     | 0.XX
```

You can then compute averages and standard deviations across the trial rows to get the "over ten trials average" mentioned in ROADMAP.md.

### Multi-Task Example with Custom Schemas

When running benchmarks on multiple tasks with custom schemas:

```yaml
tasks:
  task_list:
    - document_set: gsm8k
      criterion: arithmetic_operations
    - document_set: onion_headlines
      criterion: joke_type

methods_to_evaluate:
  in_one_word:
    # Custom schema with per-task contexts
    - name: inoneword_custom
      preset: inoneword_hf_qwen3_8b
      category_context:
        gsm8k_arithmetic_operations: "Categories: addition, subtraction, multiplication, division\nQuestion: Categorize this text in one word."
        onion_headlines_joke_type: "Categories: satire, irony, absurdist, parody\nQuestion: Categorize this text in one word."
```

Results will show `inoneword_custom` for both tasks, using the appropriate schema for each.

## Configuration Options

### In-One-Word Methods

```yaml
in_one_word:
  # No schema mode
  - name: inoneword_no_schema
    preset: inoneword_hf_qwen3_8b
    category_context: "Question: Categorize this text in one word."

  # Oracle schema mode (uses annotations)
  - name: inoneword_oracle_schema
    preset: inoneword_hf_qwen3_8b
    # No category_context - will use task.document_annotations

  # Proposed schema mode (generates fresh schema)
  - name: inoneword_proposed_schema
    preset: inoneword_hf_qwen3_8b
    generate_schema: true
    n_schema_samples: 10  # Number of docs to sample for schema generation
```

### Pseudologit Methods

```yaml
pseudologit:
  # Oracle schema mode (uses annotations)
  - name: pseudologit_oracle_schema
    preset: pseudologit_gemini_n100
    use_oracle_schema: true

  # Proposed schema mode (generates fresh schema)
  - name: pseudologit_proposed_schema
    preset: pseudologit_gemini_n100
    generate_schema: true
    n_schema_samples: 10  # Number of docs to sample for schema generation

  # Custom schema mode (manually specified classes)
  - name: pseudologit_custom_schema
    preset: pseudologit_gemini_n100
    classes_file: prompts/custom/gsm8k_classes.json
```

### Custom Schema Configuration

Custom schemas support both **string** (same for all tasks) and **dict** (per-task) formats.

#### Dict Format (Recommended for Multiple Tasks)

```yaml
in_one_word:
  - name: inoneword_custom_schema
    preset: inoneword_hf_qwen3_8b
    category_context:
      # Key format: {dataset_name}_{criterion_name}
      gsm8k_arithmetic_operations: "Categories: addition, subtraction, multiplication, division\nQuestion: Categorize this text in one word."
      onion_headlines_joke_type: "Categories: satire, irony, absurdist, parody\nQuestion: Categorize this text in one word."
      # Or use just criterion name as key (less specific):
      arithmetic_operations: "Categories: ..."

pseudologit:
  - name: pseudologit_custom_schema
    preset: pseudologit_gemini_n100
    classes_file:
      gsm8k_arithmetic_operations: prompts/custom/gsm8k_arithmetic_classes.json
      onion_headlines_joke_type: prompts/custom/onion_joke_classes.json
      # Or use just criterion name:
      arithmetic_operations: prompts/custom/gsm8k_arithmetic_classes.json
```

**Key Lookup Logic:**
1. First tries: `{dataset_name}_{criterion_name}` (e.g., `gsm8k_arithmetic_operations`)
2. Falls back to: `{criterion_name}` (e.g., `arithmetic_operations`)
3. If neither found: logs a warning and skips the method for that task

#### String Format (Same Schema for All Tasks)

```yaml
in_one_word:
  - name: inoneword_custom_schema
    preset: inoneword_hf_qwen3_8b
    category_context: "Categories: addition, subtraction, multiplication, division\nQuestion: Categorize this text in one word."

pseudologit:
  - name: pseudologit_custom_schema
    preset: pseudologit_gemini_n100
    classes_file: prompts/custom/gsm8k_classes.json
```

#### Classes File Format (for Pseudologit)

Create a JSON file with your classes:

```json
{
  "classes": [
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "percentages",
    "fractions"
  ]
}
```

## Running Multiple Trials (for Averaging)

The ROADMAP mentions "proposed schema over ten trials average". This is built into the benchmark system using the `num_trials` parameter:

### Automatic Multi-Trial Execution

Simply specify `num_trials` in your method config:

```yaml
in_one_word:
  - name: inoneword_proposed
    preset: inoneword_hf_qwen3_8b
    generate_schema: true
    num_trials: 10  # Run 10 times automatically
```

This will create 10 separate rows in the results:
- `inoneword_proposed_trial1`
- `inoneword_proposed_trial2`
- ...
- `inoneword_proposed_trial10`

### Computing Statistics

After the benchmark completes, compute mean and std across trials:

```python
from multiview.benchmark.evaluation_utils import compute_trial_statistics

# Load results
results = {...}  # From benchmark output

# Compute statistics
stats = compute_trial_statistics(results)

# Returns: {task_name: {method_avg: {mean_accuracy, std_accuracy, ...}}}
```

Or manually with pandas:

```python
import pandas as pd
import json

# Load results
with open("outputs/schema_comparison/summary_table.json") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results["results"]["gsm8k_arithmetic"]).T

# Filter trials and compute stats
proposed_trials = df[df.index.str.contains("proposed_trial")]
print(f"Mean: {proposed_trials['accuracy'].mean():.4f}")
print(f"Std:  {proposed_trials['accuracy'].std():.4f}")

## Implementation Details

### Code Changes

The schema comparison feature is implemented in:
- `src/multiview/benchmark/evaluation_utils.py:590-651` - `in_one_word` with schema generation
- `src/multiview/benchmark/evaluation_utils.py:653-738` - `pseudologit` with schema generation
- `configs/benchmark_schema_comparison.yaml` - Example configuration

### How It Works

1. **No schema mode**: Passes a generic `category_context` to `evaluate_with_in_one_word`
2. **Oracle schema mode**: Extracts `category_schema` from `task.document_annotations`
3. **Proposed schema mode**: Calls `generate_category_schema()` during evaluation to create fresh schema

### Caching Behavior and Trial Variance

**How Trials Get Different Schemas:**

When using `num_trials > 1`:
1. Each trial gets a unique cache key (includes `_trial_idx` in config hash)
2. Same document sample is used across trials (for fair comparison)
3. LLM generates different schemas due to:
   - Different cache keys (no cache hits)
   - Model sampling/temperature (if enabled)

**Cache Key Structure:**
- Oracle schemas: Cached from annotation phase with criterion-based seed
- Proposed schemas: Each trial gets unique cache like `{task}_eval_{method}_{hash}_trial{N}_generated_schema`
- The `_trial_idx` in the config ensures cache differentiation

**Reproducibility:**
- With caching enabled: Each trial is reproducible within the same benchmark run
- Across runs: Trials may vary if the underlying LLM changes or cache is cleared
- For exact reproducibility: Use deterministic LLM configs (temperature=0)

## Interpreting Results

### Expected Patterns

- **Oracle vs. Proposed**: Large differences suggest schema sensitivity
- **No schema vs. Oracle**: Shows benefit of explicit categorization
- **Custom vs. Oracle**: Tests if manual schemas outperform LLM-generated ones
- **Variance across trials**: High variance indicates unstable schema generation

### Example Analysis

```
In one word (no schema)            | 0.65
In one word (custom schema)        | 0.75  ← +10% with expert schema
In one word (oracle schema)        | 0.78  ← +13% with oracle
In one word (proposed schema avg)  | 0.72  ← +7% with fresh schemas
```

This pattern suggests:
- Explicit schemas help significantly (+7-13% vs no schema)
- Oracle schema is best (+6% vs proposed, +3% vs custom)
- Method is moderately sensitive to schema choice (78% - 72% = 6% variance)
- Custom expert schemas underperform oracle by 3% (interesting finding!)

### Use Cases for Schema Comparison

1. **Schema Sensitivity Analysis**: How much does performance vary with schema?
2. **Oracle Quality Assessment**: Is the oracle schema particularly good/bad?
3. **Expert vs. LLM Schemas**: Do hand-crafted schemas beat generated ones?
4. **Ablation Studies**: Test specific schema design choices (granularity, coverage, etc.)
5. **Robustness Testing**: Does the method work across different reasonable schemas?

## Troubleshooting

### "Missing category_context" Error

If you see this error for `in_one_word`:
- Make sure task has annotations with `category_schema`, OR
- Specify `category_context` in config, OR
- Set `generate_schema: true`

### "Missing category schema in annotations" Error

If you see this error for `pseudologit` with `use_oracle_schema: true`:
- Make sure task used `lm_all` triplet style (not `prelabeled`)
- Check that annotations contain `category_schema` field

### Proposed Schemas Are Identical

If proposed schemas don't vary across runs:
- Disable caching: `use_cache: false` in config
- Use different `cache_alias` values for each trial
- Use different `seed` values

## Advanced: Schema Ablation Studies

You can use custom schemas to test specific design choices:

```yaml
# Test schema granularity
in_one_word:
  # Coarse-grained (4 categories)
  - name: inoneword_coarse
    preset: inoneword_hf_qwen3_8b
    category_context: "Categories: arithmetic, word_problems, percentages, other\nQuestion: Categorize this text in one word."

  # Fine-grained (8 categories)
  - name: inoneword_fine
    preset: inoneword_hf_qwen3_8b
    category_context: "Categories: addition, subtraction, multiplication, division, fractions, decimals, percentages, mixed\nQuestion: Categorize this text in one word."

  # Very fine-grained (12 categories)
  - name: inoneword_very_fine
    preset: inoneword_hf_qwen3_8b
    category_context: "Categories: simple_addition, multi_step_addition, subtraction, multiplication, division, fractions, decimals, percentages, ratios, mixed_operations, word_problems, geometry\nQuestion: Categorize this text in one word."
```

Compare results to find optimal granularity for your domain.

## Future Enhancements

Potential improvements to this feature:

1. **Schema quality metrics**: Measure schema consistency, coverage, distinctiveness
2. **Schema visualization**: Compare oracle vs. proposed schemas visually
3. **Cross-dataset schema transfer**: Test schemas generated on one dataset, used on another
4. **Automatic granularity search**: Find optimal number of categories automatically

## Related Files

- `ROADMAP.md` - Original feature specification (medium-term goal)
- `configs/benchmark_schema_comparison.yaml` - Example configuration
- `src/multiview/benchmark/evaluation_utils.py` - Implementation
- `src/multiview/eval/in_one_word.py` - In-one-word evaluation method
- `src/multiview/eval/pseudologit.py` - Pseudologit evaluation method
- `src/multiview/benchmark/annotations/class_schema.py` - Schema generation
