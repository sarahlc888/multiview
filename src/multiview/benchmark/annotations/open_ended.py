"""Summary guidance generation and open-ended annotation.

This module handles open-ended summary annotation (similar to lm_open_ended.py):
- Generate enhanced criteria descriptions from sample documents
- Generate summary guidance from sample documents
- Create structured summaries based on guidance
"""

from __future__ import annotations

import logging

from multiview.inference.inference import run_inference
from multiview.utils.sampling_utils import deterministic_sample

logger = logging.getLogger(__name__)


def generate_criteria_description(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    n_samples: int = 10,
    cache_alias: str | None = None,
) -> dict:
    """Generate an enhanced criteria description from sample documents.

    Takes a brief criterion name and description, samples documents, and generates
    a more detailed description that helps compare documents for similarity.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Brief description of what the criterion means
        n_samples: Number of documents to sample
        cache_alias: Optional cache alias for inference calls

    Returns:
        Dict with structure:
        {
            "description": "Enhanced criteria description (str)",
        }
    """
    # Sample documents deterministically based on criterion
    sample_docs = deterministic_sample(documents, n_samples, criterion)
    sample_docs_str = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(sample_docs))

    # Prepare inputs
    inputs = {
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "sample_documents": [sample_docs_str],
    }

    # Generate enhanced description using inference
    results = run_inference(
        inputs=inputs,
        config="criteria_description_generation_gemini",
        cache_alias=cache_alias,
        verbose=True,
    )

    description = results[0]
    if description is None:
        description = criterion_description or ""
    else:
        description = description.strip()

    logger.info("Generated enhanced criteria description")
    return {"description": description}


def generate_summary_guidance(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    n_samples: int = 10,
    guidance_hint: str | None = None,
    format_hint: str | None = None,
    cache_alias: str | None = None,
) -> dict:
    """Generate summary guidance from sample documents.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name
        criterion_description: Description of what the criterion means
        n_samples: Number of documents to sample
        guidance_hint: Optional hint about what to include in summaries
        format_hint: Optional hint about summary format/structure
        cache_alias: Optional cache alias for inference calls

    Returns:
        Summary guidance dict with structure:
        {
            "summary_guidance": "..."
        }
    """
    # Sample documents deterministically based on criterion
    sample_docs = deterministic_sample(documents, n_samples, criterion)
    sample_docs_str = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(sample_docs))

    # Prepare inputs with template variables
    # Pass empty strings for optional fields - presets handle this gracefully
    inputs = {
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "guidance_hint": [guidance_hint or ""],
        "format_hint": [format_hint or ""],
        "sample_documents": [sample_docs_str],
    }

    # Generate guidance using inference with retry on failure
    for attempt in range(2):
        results = run_inference(
            inputs=inputs,
            config="summary_guidance_generation_gemini",
            cache_alias=cache_alias,
            force_refresh=(attempt > 0),  # Skip cache on retry
            verbose=True,
        )

        guidance = results[0]
        if guidance is not None:
            logger.info(f"Generated summary guidance on attempt {attempt + 1}")
            return guidance

        if attempt == 0:
            logger.warning("First attempt failed to parse, retrying once...")

    raise ValueError("Failed to generate summary guidance after 2 attempts")


def generate_summary(
    document: str,
    criterion: str,
    criterion_description: str,
    summary_guidance: dict,
    cache_alias: str | None = None,
) -> dict:
    """Generate a structured summary for a single document.

    Args:
        document: Document string
        criterion: Criterion name
        criterion_description: Criterion description
        summary_guidance: Summary guidance dict
        cache_alias: Optional cache alias for inference calls

    Returns:
        Summary dict with structure:
        {
            "annotation_trace": "...",
            "final_summary": "..."
        }
    """
    # Extract the summary_guidance string from the dict
    guidance_str = (
        summary_guidance.get("summary_guidance", "")
        if isinstance(summary_guidance, dict)
        else str(summary_guidance)
    )

    # Prepare inputs
    inputs = {
        "document": [document],
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "summary_guidance": [guidance_str],
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config="summary_generate_gemini",
        cache_alias=cache_alias,
        verbose=False,
    )

    # Extract summary
    result = results[0]
    if result is None:
        return {"annotation_trace": "", "final_summary": ""}

    return (
        result
        if isinstance(result, dict)
        else {"annotation_trace": "", "final_summary": str(result)}
    )


def generate_summaries_batch(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    summary_guidance: dict,
    cache_alias: str | None = None,
) -> list[dict]:
    """Generate structured summaries for multiple documents.

    Args:
        documents: List of document strings
        criterion: Criterion name
        criterion_description: Criterion description
        summary_guidance: Summary guidance dict
        cache_alias: Optional cache alias for inference calls

    Returns:
        List of annotation dicts:
        [
            {
                "summary": {
                    "annotation_trace": "...",
                    "final_summary": "..."
                }
            },
            ...
        ]
    """
    # Extract the summary_guidance string from the dict
    guidance_str = (
        summary_guidance.get("summary_guidance", "")
        if isinstance(summary_guidance, dict)
        else str(summary_guidance)
    )

    # Prepare inputs
    inputs = {
        "document": documents,
        "criterion": [criterion] * len(documents),
        "criterion_description": [criterion_description or ""] * len(documents),
        "summary_guidance": [guidance_str] * len(documents),
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config="summary_generate_gemini",
        cache_alias=cache_alias,
        verbose=False,
    )

    # Extract annotations
    annotations = []
    for result in results:
        if result is None:
            annotations.append(
                {"summary": {"annotation_trace": "", "final_summary": ""}}
            )
        else:
            # Result should be dict with annotation_trace and final_summary
            summary_dict = (
                result
                if isinstance(result, dict)
                else {"annotation_trace": "", "final_summary": str(result)}
            )
            annotations.append({"summary": summary_dict})

    logger.info(f"Generated summaries for {len(documents)} documents")
    return annotations
