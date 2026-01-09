"""Inference configuration and presets.

This module defines InferenceConfig and provides preset configurations
for common models/providers. All presets are defined in one place for easy
browsing and maintenance.
"""

from dataclasses import dataclass, field, replace


@dataclass
class InferenceConfig:
    """Configuration for inference with LMs or embedding models.

    This replaces the old YAML-based annotator configs with a type-safe
    Python dataclass that supports the same functionality.
    """

    # Provider and model
    provider: str  # "openai", "anthropic", "hf_api"
    model_name: str

    # Prompt templates
    # Can be inline strings or paths to template files
    prompt_template: str
    embed_query_instr_template: str | None = None
    embed_doc_instr_template: str | None = None
    force_prefill_template: str | None = None

    # Model parameters
    temperature: float = 0.0
    max_tokens: int | None = None

    # Model type
    is_embedding: bool = False

    # Output parsing
    parser: str = "text"  # "json", "vector", "text"
    parser_kwargs: dict | None = None

    # Additional kwargs to pass to completion function
    extra_kwargs: dict = field(default_factory=dict)

    def with_overrides(self, **kwargs) -> "InferenceConfig":
        """Return a new config with overrides applied.

        Args:
            **kwargs: Fields to override

        Returns:
            New InferenceConfig with overrides applied
        """
        return replace(self, **kwargs)

    def to_completion_kwargs(self) -> dict:
        """Convert config to kwargs for completion functions."""
        kwargs = {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        # Merge in any extra kwargs
        kwargs.update(self.extra_kwargs)

        return kwargs


# ============================================================================
# EMBEDDING MODEL PRESETS
# ============================================================================

OPENAI_EMBEDDING_LARGE = InferenceConfig(
    provider="openai",
    model_name="text-embedding-3-large",
    prompt_template="{document}",
    is_embedding=True,
    parser="vector",
)

OPENAI_EMBEDDING_SMALL = InferenceConfig(
    provider="openai",
    model_name="text-embedding-3-small",
    prompt_template="{document}",
    is_embedding=True,
    parser="vector",
)

HF_QWEN3_EMBEDDING_8B = InferenceConfig(
    provider="hf_api",
    model_name="Qwen/Qwen3-Embedding-8B",
    prompt_template="{document}",
    embed_query_instr_template="Represent this query for retrieval: ",
    is_embedding=True,
    parser="vector",
)

HF_QWEN3_EMBEDDING_4B = InferenceConfig(
    provider="hf_api",
    model_name="Qwen/Qwen3-Embedding-4B",
    prompt_template="{document}",
    embed_query_instr_template="Represent this query for retrieval: ",
    is_embedding=True,
    parser="vector",
)

# ============================================================================
# LM JUDGE PRESETS
# ============================================================================

CLAUDE_SONNET = InferenceConfig(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

CLAUDE_HAIKU = InferenceConfig(
    provider="anthropic",
    model_name="claude-3-5-haiku-20241022",
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

GPT41 = InferenceConfig(
    provider="openai",
    model_name="gpt-4.1",  # Updated to GPT-4.1 (outperforms gpt-4o)
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

GPT41_MINI = InferenceConfig(
    provider="openai",
    model_name="gpt-4.1-mini",  # Updated to GPT-4.1-mini (outperforms gpt-4o-mini)
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

GEMINI_FLASH = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",  # Lite model with higher free tier limits
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

GEMINI_PRO = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template="{document}",
    parser="json",
    temperature=0.0,
    max_tokens=4096,
)

# ============================================================================
# SPECIALIZED TASK PRESETS
# ============================================================================
# These presets are configured for specific multiview similarity tasks

# Note: Qwen3-Embedding-4B is not available via HF Inference API, using 8B instead
EMBED_PLAINTEXT_HFAPI = InferenceConfig(
    provider="hf_api",
    model_name="Qwen/Qwen3-Embedding-8B",
    prompt_template="{document}",
    embed_query_instr_template="Represent this query for retrieval: ",
    is_embedding=True,
    parser="vector",
)

# Prompt for document summarization based on criteria
REWRITE_PLAINTEXT_FREEFORM_GEMINI_PROMPT = """Please summarize the given document based on specific criteria.

## Document

{document}

## Criteria

{similarity_criteria}

## Your task

How does the document relate to the criteria? Provide an extractive/abstractive summary.

IMPORTANT: The summary should be standalone, and it should not excessively refer to the full document. There is no need to use complete sentences unless helpful. It will be used as a query string to search for other documents that are similar under the criteria.

Think out loud before providing your final answer in JSON format, with the key "summary"."""

REWRITE_PLAINTEXT_FREEFORM_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=REWRITE_PLAINTEXT_FREEFORM_GEMINI_PROMPT,
    parser="json",
    parser_kwargs={"annotation_key": "summary"},
    temperature=0.0,
    max_tokens=2048,
)

# Pairwise similarity judge with Likert scale (1-5)
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

# Pairwise similarity judge with binary output (0=same, 1=different)
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

# Triplet similarity judge (a vs b vs c)
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
# PRESET REGISTRY
# ============================================================================

PRESET_REGISTRY = {
    # Basic embeddings
    "openai_embedding_large": OPENAI_EMBEDDING_LARGE,
    "openai_embedding_small": OPENAI_EMBEDDING_SMALL,
    "hf_qwen3_embedding_8b": HF_QWEN3_EMBEDDING_8B,
    "hf_qwen3_embedding_4b": HF_QWEN3_EMBEDDING_4B,
    # Basic LM models
    "claude_sonnet": CLAUDE_SONNET,
    "claude_haiku": CLAUDE_HAIKU,
    "gpt41": GPT41,
    "gpt41_mini": GPT41_MINI,
    "gemini_flash": GEMINI_FLASH,
    "gemini_pro": GEMINI_PRO,
    # Specialized task presets
    "embed_plaintext_hfapi": EMBED_PLAINTEXT_HFAPI,
    "rewrite_plaintext_freeform_gemini": REWRITE_PLAINTEXT_FREEFORM_GEMINI,
    "lmjudge_pair_plaintext_likerthard_gemini": LMJUDGE_PAIR_PLAINTEXT_LIKERTHARD_GEMINI,
    "lmjudge_triplet_plaintext_binaryhard_gemini": LMJUDGE_TRIPLET_PLAINTEXT_BINARYHARD_GEMINI,
    "lmjudge_pair_norewrite_binaryhard_gemini": LMJUDGE_PAIR_NOREWRITE_BINARYHARD_GEMINI,
}


def get_preset(name: str) -> InferenceConfig:
    """Get a preset configuration by name.

    Args:
        name: Preset name (e.g., "openai_embedding_large", "lmjudge_pair_plaintext_likerthard_gemini")

    Returns:
        InferenceConfig for the preset

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESET_REGISTRY:
        raise ValueError(
            f"Unknown preset: {name}. "
            f"Available presets: {sorted(PRESET_REGISTRY.keys())}"
        )
    return PRESET_REGISTRY[name]


def list_presets() -> list[str]:
    """List all available preset names.

    Returns:
        Sorted list of preset names
    """
    return sorted(PRESET_REGISTRY.keys())
