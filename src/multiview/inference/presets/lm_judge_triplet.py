"""Triplet similarity judge presets.

This module defines presets for LM judges that compare three documents and determine
which of two documents is more similar to an anchor document.
"""

from ._base import InferenceConfig

# ============================================================================
# TRIPLET COMPARISON JUDGE (a vs b vs c)
# ============================================================================

LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_PROMPT = """You are a helpful assistant that evaluates the similarity of texts based on specific criteria.

Given three texts, (a), (b), and (c), please judge whether text (a) is more similar to text (b) or text (c) based the similarity criteria.

### Similarity criteria

{similarity_criteria}

### Text (a)

{document_a}

### Text (b)

{document_b}

### Text (c)

{document_c}

### Task

Is output (a) more similar to output (b) or output (c)?

Answer with "(b)", "(c)", or "(d)" to indicate a draw/tie.

Follow this format exactly:
```
<reasoning about the decision>
Final answer: (b) or (c) or (d)
```

IMPORTANT: Make sure to consider only the specified criteria."""

LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI_PROMPT,
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"Final\s+[Aa]nswer:\s*\(b\)": 1,
            r"Final\s+[Aa]nswer:\s*\(c\)": -1,
            r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# TRIPLET COMPARISON JUDGE WITH ANNOTATIONS (a vs b vs c)
# ============================================================================

LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_PROMPT = """You are a helpful assistant that evaluates the similarity of texts based on specific criteria.

Given three texts, (a), (b), and (c), please judge whether text (a) is more similar to text (b) or text (c) based the similarity criteria.

Each text is accompanied by a summary that highlights relevant aspects based on the similarity criteria.

### Similarity criteria

{similarity_criteria}

### Text (a)

{document_a}

**Summary for (a):**
{annotation_a}

### Text (b)

{document_b}

**Summary for (b):**
{annotation_b}

### Text (c)

{document_c}

**Summary for (c):**
{annotation_c}

### Task

Is output (a) more similar to output (b) or output (c)?

Answer with "(b)", "(c)", or "(d)" to indicate a draw/tie.

Follow this format exactly:
```
<reasoning about the decision>
Final answer: (b) or (c) or (d)
```

IMPORTANT: Make sure to consider only the specified criteria. Use the summaries to help you understand the relevant aspects of each text."""

LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=LMJUDGE_TRIPLET_WITH_ANNOTATION_GEMINI_PROMPT,
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"Final\s+[Aa]nswer:\s*\(b\)": 1,
            r"Final\s+[Aa]nswer:\s*\(c\)": -1,
            r"Final\s+[Aa]nswer:\s*\(d\)": 0.0,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)
