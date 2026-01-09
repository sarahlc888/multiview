# Plan: Add Synthetic Document Augmentation Phase

**STATUS: ✅ COMPLETED** (2026-01-09)

## Implementation Summary

Successfully implemented LM-based synthetic document augmentation using the reference prompts:

- **Files modified**: 5 files
- **Lines added**: ~200 lines total
- **Synthesis approach**: LM-based using reference prompts (not dataset-specific)
  - `synthesize_hard_positive`: Preserve criteria relationship, change everything else
  - `synthesize_hard_negative`: Change criteria relationship, keep other aspects
- **Model**: Gemini 2.5 Flash Lite (cheap and fast)
- **Architecture**:
  - Presets in `src/multiview/inference/presets.py` (following repo conventions)
  - Synthesis logic in `src/multiview/benchmark/synthesis_utils.py`
  - Task orchestration in `Task.augment_with_synthetic_documents()`
  - New delimiter parser for extracting synthesis results
  - Configuration-driven via `add_synthetic_docs` and `num_synthetic_per_doc` flags

## Context

We need to add a phase that generates synthetic documents to AUGMENT the pool of documents. This should happen **after task loads documents** and **before document annotation starts**.

## Reference Implementation Analysis

Found a comprehensive reference implementation in `/Users/sarahchen/code/pproj/project/` with the following architecture:

### Reference Annotators

The config references two synthesis annotators:
- **`synthesize_add_superficial_diff`** (aka "hard positive"): Rewrites text while PRESERVING relationship to criteria, but changing surface details
- **`synthesize_make_fundamental_change`** (aka "hard negative"): Rewrites text to COMPLETELY CHANGE relationship to criteria

### Reference Architecture

**Configuration-Driven Design**:
- Annotators defined via YAML configs (`configs/annotate/annotators/synthesize/*/configs.yaml`)
- Each annotator maps to a prompt template file
- Uses local LM (Qwen3-0.6B) for text generation
- Synthesis config in main benchmark config (`synthesis.synthesis_annotators`)

**Integration Point** (`make_tuples.py:560-570`):
```python
if synthesis_config.add_synthesized:
    tuple_dataset = augment_tuples(
        tuple_dataset=tuple_dataset,
        synthesis_config=synthesis_config,
        force_refresh=force_refresh
    )
```

**Core Logic** (`synthesis_utils.py`):
- `augment_tuples()`: Main function with 2-pass batch processing
  1. **First pass**: Batch query all annotators to build cache
  2. **Second pass**: Process individual tuples using cached results
- `_synthesize_for_tuple_cached()`: Per-tuple processing applying each synthesis annotator
- Input format: `{"instruction": request, "text": output, "criteria": similarity_criteria}`
- Validation: Non-identical, length < 2x original, completeness checks
- Metadata tracking: synthesis indices, provenance

## Current Multiview Flow

```
Task.__init__(config)
  └─> Task.load_documents()
      └─> DocumentSet.load_documents()
  └─> if triplet_style != "random":
      └─> Task.annotate_documents()
          └─> annotation_utils.annotate_with_*()
  └─> Task.create_triplets()
      └─> triplet_utils.create_*_triplets()
  └─> Task.save_triplets()
```

**Key Files**:
- `scripts/run_eval.py`: Entry point
- `src/multiview/benchmark/task.py`: Task orchestration (~10 line methods)
- `src/multiview/benchmark/annotation_utils.py`: Annotation logic
- `src/multiview/benchmark/triplets/triplet_utils.py`: Triplet creation
- `src/multiview/inference/inference.py`: LM inference with caching

**Insertion Point**: After `load_documents()`, before `annotate_documents()`

**Existing Infrastructure**:
- Config flag exists: `add_synthetic_docs: false` in `benchmark.yaml`
- Inference module with caching: `src/multiview/inference/`
- Thin wrapper pattern: delegate to utility modules

## User Requirements

1. **Architecture**: Config-driven (YAML configs + prompt templates)
2. **Synthesis strategies**: Hybrid approach
   - **LM-based annotators**: Port both `synthesize_add_superficial_diff` and `synthesize_make_fundamental_change`
   - **Dataset-specific synthesis**: For datasets with structure (e.g., GSM8K)
     - Example: Sample problems X and Y, create (X, B, C) where:
       - B = math of X + spurious features of Y
       - C = math of Y + spurious features of X
3. **LM provider**: Reuse existing inference module (OpenAI/Anthropic)
4. **Synthesis timing**: Document-level, criterion-aware (after loading, before annotation)

## Proposed Architecture (Simplified)

### Start Simple: Dataset-Specific Synthesis Only

**Phase 1 (this implementation)**: Dataset-specific synthesis for GSM8K
- Implement GSM8K cross-pollination (swap math vs spurious features)
- Clean interface: just override one method in DocumentSet
- No external config needed beyond the `add_synthetic_docs` flag

**Phase 2 (future)**: Add LM-based synthesis if needed
- Can add prompt templates + inference module later
- Would work as fallback for datasets without custom synthesis
- Keep it optional and pluggable

**Why start simple**:
- GSM8K is the immediate use case
- Dataset-specific synthesis is more effective anyway
- Can add LM-based synthesis later without changing interfaces
- Reduces complexity significantly (no prompt templates, no InferenceConfig, no global synthesis config)

### Config Structure (Simplified)

**Benchmark Config** - Minimal changes to `configs/benchmark.yaml`:
```yaml
# Under tasks.defaults (task-specific settings)
tasks:
  defaults:
    add_synthetic_docs: false  # EXISTING FLAG - just use it
    num_synthetic_per_doc: 2   # NEW - controls how many synthetic docs to generate

  task_list:
    - document_set: gsm8k
      criterion: arithmetic
      add_synthetic_docs: true  # Enable for GSM8K
```

**That's it!** No global synthesis config, no prompt templates, no inference presets needed.

## Implementation Plan (Simplified)

### Phase 1: Core Infrastructure (Minimal)

**1.1 Add synthesis method to Task** (`src/multiview/benchmark/task.py`)
- Extract flags in `__init__`:
  - `self.add_synthetic_docs = config.get("add_synthetic_docs", False)`
  - `self.num_synthetic_per_doc = config.get("num_synthetic_per_doc", 2)`

```python
def augment_with_synthetic_documents(self):
    """Generate synthetic documents to augment the pool."""
    if not self.add_synthetic_docs:
        return

    # Try dataset-specific synthesis
    synthetic_docs = self.document_set.synthesize_documents(
        self.documents,
        self.criterion_name,
        self.num_synthetic_per_doc
    )

    if synthetic_docs:
        logger.info(f"Added {len(synthetic_docs)} synthetic documents")
        self.documents.extend(synthetic_docs)
```

**Clean interface**: No parameters needed, no utility module needed, just calls the DocumentSet!

**1.2 Update run_eval.py integration**
```python
cur_task.load_documents()
if cur_task.add_synthetic_docs:  # NEW - use existing flag
    cur_task.augment_with_synthetic_documents()  # NEW - no parameters!
if cur_task.triplet_style != "random":
    cur_task.annotate_documents()
```

### Phase 2: Dataset-Specific Synthesis (GSM8K)

**2.1 Extend BaseDocSet interface** (`src/multiview/benchmark/document_sets/base.py`)
```python
class BaseDocSet:
    def synthesize_documents(
        self,
        documents: List[str],
        criterion_name: str,
        num_synthetic_per_doc: int = 2
    ) -> List[str]:
        """Optional: Generate synthetic documents using dataset-specific logic.

        This method can be overridden by subclasses to provide custom synthesis.
        If not overridden, returns empty list (no dataset-specific synthesis).

        Args:
            documents: List of original documents
            criterion_name: Criterion being used for triplet creation
            num_synthetic_per_doc: How many synthetic docs to generate per original

        Returns:
            List of synthetic documents, or empty list if not implemented.
        """
        return []  # Default: no dataset-specific synthesis
```

**Why this design is general and extensible**:
- **Zero configuration needed**: Any DocumentSet can add synthesis by simply overriding this method
- **No registration required**: Just implement the method in your DocumentSet subclass
- **Graceful fallback**: If not overridden, returns empty list and falls back to LM-based synthesis
- **Clean interface**: Clear contract with type hints and documentation
- **Easy to add new datasets**: To add synthesis for a new dataset:
  1. Open the dataset's file (e.g., `rocstories.py`)
  2. Override `synthesize_documents()` method
  3. Implement your dataset-specific logic
  4. Done! No need to modify any other files

**Example for adding a new dataset**:
```python
# In src/multiview/benchmark/document_sets/rocstories.py
class RocStoriesDocSet(BaseDocSet):
    # ... existing methods ...

    def synthesize_documents(self, documents, criterion_name, num_synthetic_per_doc=2):
        """Generate synthetic stories by mixing characters/settings."""
        # Your custom logic here for ROC stories
        # e.g., swap characters between stories, change settings, etc.
        return synthetic_stories
```

That's it! The framework handles everything else.

**2.2 Implement GSM8K-specific synthesis** (`src/multiview/benchmark/document_sets/gsm8k.py`)
- Override `synthesize_documents()` method
- Parse documents to extract "Question" and "Answer" components
- Implement cross-pollination strategy:
  - Sample pairs (X, Y)
  - Create B: question from X, answer from Y (hard negative)
  - Create C: question from Y, answer from X (hard negative)
- Handle spurious features (number values, context, etc.)
- Return list of synthetic documents in same format as originals

### Phase 3: Configuration & Testing

**3.1 Update benchmark.yaml**
- Keep existing `add_synthetic_docs: false` flag in defaults
- Add `num_synthetic_per_doc: 2` to defaults
- For GSM8K task specifically, set `add_synthetic_docs: true`

**3.2 Test with GSM8K task**
- Set `add_synthetic_docs: true` for GSM8K task
- Verify synthetic documents are created
- Verify they're annotated alongside real documents
- Verify they appear in triplets


## Critical Files to Modify (Simplified)

1. **`src/multiview/benchmark/task.py`** (3 lines added to `__init__`, ~8 line method)
   - Extract `add_synthetic_docs` and `num_synthetic_per_doc` from config
   - Add `augment_with_synthetic_documents()` method (no parameters!)

2. **`scripts/run_eval.py`** (2 lines added)
   - Insert synthesis call after document loading

3. **`src/multiview/benchmark/document_sets/base.py`** (~10 lines)
   - Add `synthesize_documents()` interface method with default implementation

4. **`src/multiview/benchmark/document_sets/gsm8k.py`** (~30-50 lines)
   - Implement GSM8K-specific synthesis

5. **`configs/benchmark.yaml`** (2 lines changed)
   - Add `num_synthetic_per_doc: 2` to defaults
   - Set `add_synthetic_docs: true` for GSM8K task

**That's it! 5 files, minimal changes, clean interfaces.**

## Verification Plan

**End-to-end test**:
1. Set `add_synthetic_docs: true` for GSM8K task in config
2. Run benchmark on GSM8K task: `python scripts/run_eval.py`
3. Verify outputs:
   - Check that synthetic documents are in document list (print length before/after)
   - Check that annotations include synthetic documents
   - Check that triplets include synthetic documents

**Unit tests** (in `tests/benchmark/`):
- Test GSM8K `synthesize_documents()` method with sample documents
- Test that non-implemented DocumentSets return empty list
