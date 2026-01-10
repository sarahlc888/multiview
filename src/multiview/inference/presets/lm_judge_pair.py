"""Pairwise similarity judge presets.

This module defines presets for LM judges that compare two documents and output
similarity scores.
"""

from ._base import InferenceConfig

# ============================================================================
# LIKERT SCALE JUDGE (1-5)
# ============================================================================

LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI_PROMPT = """Given two texts -- (A) and (B) -- please judge how similar text (A) is to text (B).

Please make this judgement based on the similarity criteria given below.

### Similarity criteria

{similarity_criteria}

### Text (A)

{document_a}

### Text (B)

{document_b}

### Task

How similar are text (a) and (b) based on the similarity criteria?

1: not at all similar
2: somewhat similar
3: moderately similar
4: very similar
5: extremely close match

Consider only the provided criteria. Ignore irrelevant/extraneous aspects of the texts.

Think out loud, then, provide a judgement in the format "Final judgement: [digit]"."""

LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI_PROMPT,
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# BINARY JUDGE (0=same, 1=different)
# ============================================================================

LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI_PROMPT = """Given two texts -- (A) and (B) -- please judge how similar text (A) is to text (B).

Please make this judgement based on the similarity criteria given below.

### Similarity criteria

{similarity_criteria}

### Text (A)

{document_a}

### Text (B)

{document_b}

### Task

How similar are text (A) and text (B) based on the similarity criteria?

0: generally the same (matching)
1: generally different (not matching)

Consider only the provided criteria. Ignore irrelevant/extraneous aspects of the texts.

Provide a judgement in the format "Final judgement: [digit]" exactly."""

LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI_PROMPT,
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            # Model outputs 0 for "same", return 0; outputs 1 for "different", return 1
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
        }
    },
    temperature=0.0,
    max_tokens=2048,
)

# ============================================================================
# LIKERT SCALE JUDGE WITH ANNOTATIONS (1-5)
# ============================================================================

LMJUDGE_PAIR_WITH_ANNOTATION_LIKERTHARD_GEMINI_PROMPT = """Given two texts -- (A) and (B) -- please judge how similar text (A) is to text (B).

Please make this judgement based on the similarity criteria given below.

Each text is accompanied by a summary that highlights relevant aspects based on the similarity criteria.

### Similarity criteria

{similarity_criteria}

### Text (A)

{document_a}

**Summary for (A):**
{annotation_a}

### Text (B)

{document_b}

**Summary for (B):**
{annotation_b}

### Task

How similar are text (a) and (b) based on the similarity criteria?

1: not at all similar
2: somewhat similar
3: moderately similar
4: very similar
5: extremely close match

Consider only the provided criteria. Ignore irrelevant/extraneous aspects of the texts. Use the summaries to help you understand the relevant aspects of each text.

Think out loud, then, provide a judgement in the format "Final judgement: [digit]"."""

LMJUDGE_PAIR_WITH_ANNOTATION_LIKERTHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=LMJUDGE_PAIR_WITH_ANNOTATION_LIKERTHARD_GEMINI_PROMPT,
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?2(?:\]|\*\*)?": 2,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?3(?:\]|\*\*)?": 3,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?4(?:\]|\*\*)?": 4,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?5(?:\]|\*\*)?": 5,
        }
    },
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# BINARY JUDGE WITH ANNOTATIONS (0=same, 1=different)
# ============================================================================

LMJUDGE_PAIR_WITH_ANNOTATION_BINARYHARD_GEMINI_PROMPT = """Given two texts -- (A) and (B) -- please judge how similar text (A) is to text (B).

Please make this judgement based on the similarity criteria given below.

Each text is accompanied by a summary that highlights relevant aspects based on the similarity criteria.

### Similarity criteria

{similarity_criteria}

### Text (A)

{document_a}

**Summary for (A):**
{annotation_a}

### Text (B)

{document_b}

**Summary for (B):**
{annotation_b}

### Task

How similar are text (A) and text (B) based on the similarity criteria?

0: generally the same (matching)
1: generally different (not matching)

Consider only the provided criteria. Ignore irrelevant/extraneous aspects of the texts. Use the summaries to help you understand the relevant aspects of each text.

Provide a judgement in the format "Final judgement: [digit]" exactly."""

LMJUDGE_PAIR_WITH_ANNOTATION_BINARYHARD_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=LMJUDGE_PAIR_WITH_ANNOTATION_BINARYHARD_GEMINI_PROMPT,
    parser="regex",
    parser_kwargs={
        "outputs_to_match": {
            # Model outputs 0 for "same", return 0; outputs 1 for "different", return 1
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?0(?:\]|\*\*)?": 0,
            r"[Jj]udge?ment\*?\*?:?\s*(?:\[|\*\*)?1(?:\]|\*\*)?": 1,
        }
    },
    temperature=0.0,
    max_tokens=2048,
)
