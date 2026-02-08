"""Pseudologit provider for taxonomy-based classification with distribution vectors.

Samples a language model N times (default: 100) with temperature > 0 to create soft,
probabilistic embeddings over taxonomy classes instead of single hard classifications.

## Concept

Instead of: "This problem is type A" (single label)
You get: A: 65%, B: 20%, C: 10%, D: 5%, others: 0%
Embedding: `[65, 20, 10, 5, 0, 0, 0, 0]`

**Benefits**: Captures uncertainty, richer signal than hard labels, interpretable dimensions,
better for visualization.

## Usage

1. **Define taxonomy**: Create JSON with classes, descriptions, taxonomy_context, and instruction
   (e.g., `prompts/custom/gsm8k_classes.json`)

2. **Use preset**: `--embedding-preset pseudologit_gemini_n100`

## Available Presets

Naming: `pseudologit_<model>_n<samples>`

- `pseudologit_gemini_n10/n50/n100/n200`: Gemini 2.5 Flash Lite with 10/50/100/200 samples
- `pseudologit_openai_n100`: GPT-4o-mini with 100 samples

Recommended: n100 for standard use, n50 for testing, n200 for high precision.

## Custom Presets

Add to `src/multiview/inference/presets/__init__.py`:

```python
"pseudologit_<model>_n<N>": InferenceConfig(
    provider="pseudologit",
    model_name="...",
    parser="vector",
    extra_kwargs={
        "classes_file": "prompts/custom/taxonomy.json",
        "n_samples": N,
        "temperature": 0.8,
        "provider": "openai",  # or "gemini"
        "max_tokens": 16,
        "max_workers": 10,
    },
)
```

## Tips

- Start with n50 for prototyping, n100+ for final results
- Higher temp (0.8-1.0) = more diversity; lower (0.5-0.7) = more consistency
- Embedding dimension = number of taxonomy classes
- Specify "output ONLY the letter" in taxonomy instruction for reliable parsing
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from multiview.inference.parsers import get_parser
from multiview.utils.prompt_utils import read_or_return

logger = logging.getLogger(__name__)


def _load_classes_file(classes_file: str) -> dict:
    """Load the taxonomy classes JSON file.

    Args:
        classes_file: Path to JSON file with classes, descriptions, taxonomy_context, and instruction

    Returns:
        Dict with keys: classes, descriptions, taxonomy_context, instruction
    """
    classes_path = Path(classes_file)

    # Try relative to prompts/custom/ first
    if not classes_path.is_absolute():
        # Try relative to src/multiview/
        base_path = Path(__file__).parent.parent.parent
        classes_path = base_path / classes_file

    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found: {classes_file}")

    with open(classes_path) as f:
        classes_data = json.load(f)

    required_keys = ["classes", "descriptions", "instruction"]
    for key in required_keys:
        if key not in classes_data:
            raise ValueError(f"Classes file missing required key: {key}")

    # Auto-generate taxonomy_context from classes and descriptions if not provided
    if "taxonomy_context" not in classes_data:
        logger.info("Auto-generating taxonomy_context from classes and descriptions")
        classes_list = classes_data["classes"]
        descriptions = classes_data["descriptions"]

        # Build formatted taxonomy context
        taxonomy_lines = ["Here is the taxonomy:"]
        for cls in classes_list:
            if cls in descriptions:
                taxonomy_lines.append(f"\n{cls}. {descriptions[cls]}")

        classes_data["taxonomy_context"] = "\n".join(taxonomy_lines)

    return classes_data


def _build_regex_parser_config(valid_classes: list[str]) -> dict:
    """Build regex parser config for extracting class labels.

    Creates patterns for "$\\boxed{<label>}$" format.

    Args:
        valid_classes: List of valid class labels (e.g., ["A", "B", "C", ...])

    Returns:
        Dict with outputs_to_match for regex parser
    """
    import re

    outputs_to_match = {}
    for class_label in valid_classes:
        # Pattern matches "$\boxed{A}$" (LaTeX boxed format)
        # Escape the label in case it contains regex metacharacters
        escaped_label = re.escape(class_label)
        pattern = rf"\$\\boxed\{{{escaped_label}\}}\$"
        outputs_to_match[pattern] = class_label
    return {"outputs_to_match": outputs_to_match}


def pseudologit_completions(
    prompts: list[str],
    model_name: str,
    classes_file: str,
    prompt_template: str = "prompts/custom/pseudologit_classify.txt",
    n_samples: int = 100,
    temperature: float = 0.7,
    provider: str = "gemini",
    max_tokens: int = 16,
    max_workers: int = 5,
    normalize: str | bool = "sum",
    **kwargs,
) -> dict:
    """Generate pseudologit embeddings by sampling a model N times over taxonomy classes.

    For each prompt:
    1. Load taxonomy from classes_file
    2. Format prompt using prompt_template with taxonomy context + instruction
    3. Sample model N times with temperature > 0
    4. Extract class label from each response
    5. Count occurrences of each class
    6. Return distribution vector (counts or frequencies)

    Args:
        prompts: List of documents to classify
        model_name: Model name to use for sampling
        classes_file: Path to JSON file defining the taxonomy
        prompt_template: Path to prompt template file or inline template string
                        (default: "prompts/custom/pseudologit_classify.txt")
                        Template should accept {document}, {taxonomy_context}, {instruction}
        n_samples: Number of times to sample for each prompt (default: 100)
        temperature: Sampling temperature (default: 0.7)
        provider: Which provider to use ("gemini", "openai", etc.)
        max_tokens: Max tokens for each sample (default: 16, just need the class label)
        max_workers: Concurrent workers for API calls
        normalize: Normalization mode: "l2" (unit vector, default, recommended for cosine similarity),
                   "sum" (probability vector, sum=1), or False/None (keep as raw counts)
        **kwargs: Additional provider-specific parameters

    Returns:
        Dict with "completions" key containing list of dicts with "vector" key
        Each vector has length = num_classes, with counts or normalized values
    """
    # Load taxonomy
    classes_data = _load_classes_file(classes_file)
    valid_classes = classes_data["classes"]
    taxonomy_context = classes_data["taxonomy_context"]
    instruction = classes_data["instruction"]

    logger.info(f"Loaded taxonomy with {len(valid_classes)} classes: {valid_classes}")
    logger.info(
        f"Generating pseudologit embeddings with {n_samples} samples per document"
    )

    # Load prompt template
    template_text = read_or_return(prompt_template)

    # Build regex parser config for extracting class labels
    regex_parser = get_parser("regex")
    parser_kwargs = _build_regex_parser_config(valid_classes)

    # Import the appropriate provider
    if provider == "gemini":
        from multiview.inference.providers.gemini import gemini_completions

        completion_fn = gemini_completions
    elif provider == "openai":
        from multiview.inference.providers.openai import openai_completions

        completion_fn = openai_completions
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Format ALL prompts for ALL documents upfront (batch them together)
    all_sample_prompts = []
    doc_prompt_indices = []  # Track which prompts belong to which document

    for doc_idx, document in enumerate(prompts):
        # Format the classification prompt using template
        full_prompt = template_text.format(
            document=document,
            taxonomy_context=taxonomy_context,
            instruction=instruction,
        )

        # Track start index for this document's samples
        start_idx = len(all_sample_prompts)

        # Create N copies of this prompt (one per sample)
        all_sample_prompts.extend([full_prompt] * n_samples)

        # Track end index for this document's samples
        end_idx = len(all_sample_prompts)
        doc_prompt_indices.append((start_idx, end_idx))

        if doc_idx == 0:
            logger.debug(f"Sample prompt: {full_prompt}")

    logger.info(
        f"Processing {len(prompts)} documents x {n_samples} samples = "
        f"{len(all_sample_prompts)} total prompts"
    )

    # Process ALL prompts in one batch
    result = completion_fn(
        prompts=all_sample_prompts,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
        **kwargs,
    )

    all_completions = result["completions"]

    # Process results for each document
    all_vectors = []
    for doc_idx, (start_idx, end_idx) in enumerate(doc_prompt_indices):
        # Get completions for this document
        completions = all_completions[start_idx:end_idx]

        # Extract class labels from each sample using regex parser
        extracted_labels = []
        for completion in completions:
            label = regex_parser(completion, **parser_kwargs)
            if label:
                extracted_labels.append(label)

        # Count occurrences
        label_counts = Counter(extracted_labels)

        # Create distribution vector (in order of classes)
        vector = [label_counts.get(cls, 0) for cls in valid_classes]

        # Log statistics
        total_valid = sum(vector)
        logger.info(
            f"  Document {doc_idx + 1}: {total_valid}/{n_samples} valid labels extracted"
        )
        if total_valid > 0:
            top_class = valid_classes[vector.index(max(vector))]
            top_count = max(vector)
            logger.info(
                f"  Top class: {top_class} ({top_count}/{total_valid} = {top_count/total_valid*100:.1f}%)"
            )

        # Normalize vector if requested
        if normalize:
            if normalize == "sum" or normalize is True:
                # Sum normalization (probability vector, default)
                total = sum(vector)
                if total > 0:
                    vector = [count / total for count in vector]
                # If total is 0, keep zeros (no valid labels extracted)
            elif normalize == "l2":
                # L2 normalization (unit vector)
                vec_array = np.array(vector, dtype=float)
                norm = np.linalg.norm(vec_array)
                if norm > 0:
                    vector = (vec_array / norm).tolist()
                # If norm is 0, keep zeros (no valid labels extracted)
            else:
                raise ValueError(
                    f"Invalid normalize value: {normalize}. Must be 'sum', 'l2', or False"
                )

        all_vectors.append(
            {
                "vector": vector,
                "raw_samples": completions,  # Cache N raw LM samples for re-aggregation
            }
        )

    return {"completions": all_vectors}
