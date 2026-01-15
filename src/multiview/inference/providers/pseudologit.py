"""Pseudologit provider for taxonomy-based classification with distribution vectors.

Samples a model N times to get a distribution over taxonomy classes instead of a single hard classification.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

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

    required_keys = ["classes", "descriptions", "taxonomy_context", "instruction"]
    for key in required_keys:
        if key not in classes_data:
            raise ValueError(f"Classes file missing required key: {key}")

    return classes_data


def _build_regex_parser_config(valid_classes: list[str]) -> dict:
    """Build regex parser config for extracting class labels.

    Creates patterns for "$\\boxed{<letter>}$" format.

    Args:
        valid_classes: List of valid class labels (e.g., ["A", "B", "C", ...])

    Returns:
        Dict with outputs_to_match for regex parser
    """
    outputs_to_match = {}
    for class_label in valid_classes:
        # Pattern matches "$\boxed{A}$" (LaTeX boxed format)
        # Need to escape: $ \ { }
        pattern = rf"\$\\boxed\{{{class_label}\}}\$"
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
        **kwargs: Additional provider-specific parameters

    Returns:
        Dict with "completions" key containing list of dicts with "vector" key
        Each vector has length = num_classes, with counts of each class
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

    # For each document, format prompts with taxonomy and sample N times
    all_vectors = []

    for doc_idx, document in enumerate(prompts):
        # Format the classification prompt using template
        full_prompt = template_text.format(
            document=document,
            taxonomy_context=taxonomy_context,
            instruction=instruction,
        )

        # Create N copies of this prompt (one per sample)
        sample_prompts = [full_prompt] * n_samples

        logger.info(
            f"Sampling document {doc_idx + 1}/{len(prompts)} ({n_samples} samples)..."
        )
        logger.debug(f"Sample prompt: {sample_prompts[0]}")
        # Sample N times
        result = completion_fn(
            prompts=sample_prompts,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_workers=max_workers,
            **kwargs,
        )
        # print(f"{sample_prompts[0]=}")
        # print(f"{result['completions'][0]=}")
        # print(f"{result['completions'][1]=}")

        # Extract class labels from each sample using regex parser
        completions = result["completions"]
        extracted_labels = []

        for completion in completions:
            label = regex_parser(completion, **parser_kwargs)
            if label:
                extracted_labels.append(label)
        print(f"{extracted_labels=}")
        # breakpoint()

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

        # Normalize to frequencies (optional - can also keep as counts)
        # For now, keep as counts for interpretability
        all_vectors.append({"vector": vector})

    return {"completions": all_vectors}
