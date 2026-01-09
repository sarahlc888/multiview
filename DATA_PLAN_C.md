# Data Plan C: Analogies DocumentSet

## Overview
Implement Analogies DocumentSet for loading word analogy pairs from HuggingFace.

**Priority**: High
**Complexity**: Medium (two docs per record + error handling)
**Pattern**: HuggingFace JSONL with custom extraction

---

## Source Dataset

**HuggingFace**: `relbert/analogy_questions`
**Configs**: `bats`, `sat`, `google`, etc.
**Default Split**: `test` (not `train`)
**Fields**: `stem`, `choice`, `answer`, `prefix`

**Legacy Implementation**:
- Config: `/Users/sarahchen/code/pproj/project/data/benchmark/tasks/analogies/config.yaml`
- Custom task: `/Users/sarahchen/code/pproj/project/src/s3/benchmark/utils/factory_utils/custom_tasks/analogies.py`

---

## Document Format

**Output**: Word pairs formatted as "word1 : word2"

**Two documents per analogy record**:
1. Stem pair: `" : ".join(stem)` → "word1 : word2"
2. Answer pair: `" : ".join(choice[answer])` → "word3 : word4"

**Example**:
```
Record: {
  "stem": ["france", "paris"],
  "choice": [["england", "london"], ["germany", "berlin"]],
  "answer": 0,
  "prefix": "country-capital"
}

Documents produced:
- "france : paris"
- "england : london"
```

---

## Key Features

### 1. Two Documents Per Record
- Extract both stem and answer pairs
- Each becomes a separate document
- max_docs applies to pairs, not questions

### 2. Multiple Dataset Configs
- BATS analogies (default)
- SAT analogies
- Google analogies
- Configurable via `dataset_config`

### 3. Analogy Type as Known Criterion
- The `prefix` field (e.g., "country-capital", "verb-past") is a **known criterion**
- Can be extracted deterministically without LM
- Store with each document for later use

### 4. Robust Error Handling
- Skip malformed records
- Validate array bounds
- Log warnings for bad data

### 5. Classmap Grouping Support
- Build classmap: `prefix → list of word pairs`
- Store in document metadata
- Enables balanced sampling in triplet creation

---

## Implementation Steps

### Step 1: Get Working Documents (PRIORITY)

**Goal**: Load analogy pairs and verify they work with random triplet test.

**File**: `src/multiview/benchmark/document_sets/analogies.py`

**Minimal implementation**:

```python
"""Analogies document_set loader."""

import logging
from typing import Any

from datasets import load_dataset

from multiview.benchmark.document_sets.base import BaseDocSet

logger = logging.getLogger(__name__)


class AnalogiesDocSet(BaseDocSet):
    """Word analogy pairs document_set."""

    DATASET_PATH = "relbert/analogy_questions"
    DESCRIPTION = "Word analogy pairs (stem and answer)"
    KNOWN_CRITERIA = ["analogy_type"]  # prefix field (e.g., "country-capital")

    def load_documents(self) -> list[Any]:
        """Load analogy pairs from HuggingFace.

        For each analogy, extracts two word pairs:
        1. The stem pair: "word1 : word2"
        2. The answer pair: "word3 : word4"

        Documents are dicts with 'text' and 'analogy_type' keys to support
        analogy_type as a known criterion.

        Returns:
            List of document dicts: {"text": "word : word", "analogy_type": "country-capital"}
        """
        logger.info(f"Loading Analogies from HuggingFace: {self.DATASET_PATH}")

        # Get config params
        max_docs = self.config.get("max_docs")
        split = self.config.get("split", "test")  # Default to test
        dataset_config = self.config.get("dataset_config", "bats")

        # max_docs applies to PAIRS (each question generates 2 pairs)
        max_questions = (max_docs // 2) if max_docs else None
        use_streaming = max_questions is not None and max_questions < 100

        if use_streaming:
            logger.debug(f"Using streaming mode (max_questions={max_questions} < 100)")
            dataset = load_dataset(
                self.DATASET_PATH, dataset_config, split=split, streaming=True
            )
            dataset = dataset.shuffle(seed=42)
        else:
            dataset = load_dataset(
                self.DATASET_PATH, dataset_config, split=split
            )
            if max_questions is not None:
                dataset = dataset.shuffle(seed=42)

        # Extract word pairs with analogy type
        documents = []
        for i, example in enumerate(dataset):
            try:
                # Get analogy type (prefix)
                analogy_type = example.get("prefix", "")

                # Extract stem pair
                stem = example.get("stem", [])
                if len(stem) >= 2:
                    stem_text = " : ".join(stem)
                    documents.append({
                        "text": stem_text,
                        "analogy_type": analogy_type
                    })

                # Extract answer pair
                choices = example.get("choice", [])
                answer_idx = example.get("answer")
                if answer_idx is not None and 0 <= int(answer_idx) < len(choices):
                    answer_choice = choices[int(answer_idx)]
                    if len(answer_choice) >= 2:
                        answer_text = " : ".join(answer_choice)
                        documents.append({
                            "text": answer_text,
                            "analogy_type": analogy_type
                        })

            except (KeyError, IndexError, ValueError, TypeError) as e:
                logger.warning(f"Skipping malformed analogy at index {i}: {e}")
                continue

            # Check if we've loaded enough questions
            if max_questions is not None and i + 1 >= max_questions:
                break

        # Final max_docs enforcement
        if max_docs is not None and len(documents) > max_docs:
            documents = documents[:max_docs]

        logger.debug(f"Loaded {len(documents)} word pair documents from Analogies")
        return documents

    def get_document_text(self, document: Any) -> str:
        """Extract text from a document."""
        if isinstance(document, dict):
            return document.get("text", "")
        return document if isinstance(document, str) else ""

    def get_known_criterion_value(self, document: Any, criterion: str):
        """Extract known criterion values.

        Supports:
        - word_count: from base class
        - analogy_type: the prefix field (e.g., "country-capital")
        """
        if criterion == "analogy_type":
            if isinstance(document, dict):
                return document.get("analogy_type")
            return None

        # Fall back to base class for word_count
        return super().get_known_criterion_value(document, criterion)
```

**Verification**:
```python
# Quick test script
from multiview.benchmark.document_sets.analogies import AnalogiesDocSet

docset = AnalogiesDocSet(config={"max_docs": 10, "dataset_config": "bats"})
docs = docset.load_documents()
print(f"Loaded {len(docs)} documents")
print(f"Sample: {docs[0]}")
```

---

### Step 2: Registry Integration

Update `src/multiview/benchmark/document_sets/__init__.py`:

```python
from multiview.benchmark.document_sets.analogies import AnalogiesDocSet

DOCSETS = {
    # ... existing ...
    "analogies": AnalogiesDocSet,
}
```

---

### Step 3: Task Integration Test

Verify it works with Task class and random triplets:

```python
from multiview.benchmark.task import Task

task = Task({
    "document_set": "analogies",
    "criterion": "word_count",
    "max_docs": 10,
    "triplet_style": "random",
    "dataset_config": "bats"
})
task.load_documents()
task.create_triplets()

assert len(task.documents) > 0
assert len(task.triplets) > 0
print(f"✅ Analogies works with random triplets!")
```

---

## Configuration

### Config Parameters

```python
{
    "document_set": "analogies",
    "max_docs": 200,              # Total pairs (~100 questions)
    "split": "test",               # Default: test (not train)
    "dataset_config": "bats",      # or "sat", "google"
    "criterion": "word_count",
    "triplet_style": "random"
}
```

### Different Analogy Sets

```python
# BATS analogies (default)
{"document_set": "analogies", "dataset_config": "bats"}

# SAT analogies
{"document_set": "analogies", "dataset_config": "sat"}

# Google analogies
{"document_set": "analogies", "dataset_config": "google"}
```

---

## Implementation Notes

### Challenge 1: Two Documents Per Example

**Problem**: Each analogy question produces 2 word pairs.

**Solution**:
- Calculate `max_questions = max_docs // 2`
- Use streaming if `max_questions < 100`
- Do final enforcement: `documents[:max_docs]`

**Edge case**: Some questions may be malformed and produce 0-2 documents, so final count may vary.

### Challenge 2: Array Indexing Safety

**Validate everything**:
- Check `len(stem) >= 2` before joining
- Check `answer_idx` is within bounds
- Check `len(answer_choice) >= 2` before joining
- Wrap in try/except to catch malformed records

**Error handling**:
```python
try:
    # extraction logic
except (KeyError, IndexError, ValueError, TypeError) as e:
    logger.warning(f"Skipping malformed analogy at index {i}: {e}")
    continue
```

### Challenge 3: Dataset Config

**Multiple configs available**:
- `bats` - Bigger Analogy Test Set (default)
- `sat` - SAT-style analogies
- `google` - Google analogies

**Solution**: Make configurable via `config["dataset_config"]`

### Challenge 4: Split Default

**Override default to "test"**:
- Most datasets use `split="train"`
- Analogies better suited for evaluation, use `split="test"`
- Still configurable if needed

---

## What We're Skipping (For Now)

### Classmap Grouping Method
- Documents INCLUDE analogy_type as a known criterion
- But NO `build_classmap()` method in DocumentSet class
- Any grouping logic can be done externally if needed (e.g., in triplet_utils)
- Keeps DocumentSet thin per AGENTS.md philosophy

### BM25-based Selection
- Legacy system uses BM25 to select hard negatives
- Complex logic with `rank_bm25` library
- Belongs in `triplet_utils`, not DocumentSet
- Can add later if needed

### Strong Scores Computation
- Legacy system computes scores based on grouping
- NOT part of document loading
- Belongs in triplet creation phase

---

## Success Criteria

✅ Load documents from HuggingFace
✅ Extract both stem and answer pairs
✅ Format as "word : word"
✅ Support different dataset configs (bats, sat, etc.)
✅ Handle malformed records gracefully
✅ Support streaming mode
✅ Respect max_docs limit
✅ Integrate with Task class
✅ Works with random triplets

---

## Reference Files

**Pattern Template**: `src/multiview/benchmark/document_sets/gsm8k.py`
**Base Class**: `src/multiview/benchmark/document_sets/base.py`
**Legacy Config**: `/Users/sarahchen/code/pproj/project/data/benchmark/tasks/analogies/config.yaml`
**Legacy Processor**: `/Users/sarahchen/code/pproj/project/src/s3/benchmark/utils/factory_utils/custom_tasks/analogies.py`
