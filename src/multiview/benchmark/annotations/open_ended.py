"""Summary guidance generation and open-ended annotation.

This module handles open-ended summary annotation (similar to lm_open_ended.py):
- Generate pairwise similarity hints from sample documents
- Generate summary guidance from sample documents
- Create structured summaries based on guidance
"""

from __future__ import annotations

import logging

from multiview.inference.inference import run_inference
from multiview.utils.sampling_utils import deterministic_sample

logger = logging.getLogger(__name__)


def generate_pairwise_sim_hint(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    document_type: str | None = None,
    n_samples: int = 10,
    cache_alias: str | None = None,
    run_name: str | None = None,
) -> dict:
    """Generate a summary hint from sample documents.

    Takes a brief criterion name and description, samples documents, and generates
    a richer description that helps create structured summaries.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name (e.g., "arithmetic_operations")
        criterion_description: Brief description of what the criterion means
        document_type: Type of documents (e.g., "haiku", "math problem")
        n_samples: Number of documents to sample
        cache_alias: Optional cache alias for inference calls
        run_name: Optional experiment/run name for cache organization

    Returns:
        Dict with structure:
        {
            "summary_hint": "Summary hint (str)",
        }
    """
    # Sample documents deterministically based on criterion
    sample_docs = deterministic_sample(documents, n_samples, criterion)
    sample_docs_str = "\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(sample_docs)
    )

    # Prepare inputs
    inputs = {
        "document_type": [document_type or "document"],
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "sample_documents": [sample_docs_str],
    }

    # Generate hint using inference
    results = run_inference(
        inputs=inputs,
        config="summary_hint_generation_gemini",
        cache_alias=cache_alias,
        run_name=run_name,
        verbose=True,
    )

    hint = results[0]
    if hint is None:
        hint = criterion_description or ""
    else:
        # Parsing (e.g., delimiter extraction) is handled by the inference preset.
        hint = str(hint).strip()

    logger.info("Generated summary hint")
    return {"summary_hint": hint}


def generate_summary_guidance(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    document_type: str | None = None,
    n_samples: int = 10,
    summary_hint: str | None = None,
    cache_alias: str | None = None,
    run_name: str | None = None,
    guidance_preset: str = "summary_guidance_generation_gemini",
) -> dict:
    """Generate summary guidance from sample documents.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name
        criterion_description: Description of what the criterion means
        document_type: Type of documents (e.g., "haiku", "math problem")
        n_samples: Number of documents to sample
        summary_hint: Optional combined hint (may include desired format)
        cache_alias: Optional cache alias for inference calls
        run_name: Optional experiment/run name for cache organization
        guidance_preset: Inference preset to use for guidance generation
            (default: "summary_guidance_generation_gemini")

    Returns:
        Summary guidance dict with structure:
        {
            "summary_guidance": "..."
        }
    """
    # Sample documents deterministically based on criterion
    sample_docs = deterministic_sample(documents, n_samples, criterion)
    sample_docs_str = "\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(sample_docs)
    )

    criterion_description = (criterion_description or "").strip()

    # Format summary_hint with heading if provided
    summary_hint_formatted = (
        f"\nSUMMARY HINT:\n{summary_hint}\n" if summary_hint else ""
    )

    # Prepare inputs with template variables
    inputs = {
        "document_type": [document_type or "document"],
        "criterion": [criterion],
        "criterion_description": [criterion_description],
        "summary_hint": [summary_hint_formatted],
        "sample_documents": [sample_docs_str],
    }

    # Generate guidance using inference with retry on failure
    for attempt in range(2):
        results = run_inference(
            inputs=inputs,
            config=guidance_preset,
            cache_alias=cache_alias,
            run_name=run_name,
            force_refresh=(attempt > 0),  # Skip cache on retry
            verbose=True,
        )
        logger.info(f"Summary guidance: {results[0]['summary_guidance']}")
        guidance = results[0]
        if guidance is not None:
            logger.info(f"Generated summary guidance on attempt {attempt + 1}")
            return guidance

        if attempt == 0:
            logger.warning("First attempt failed to parse, retrying once...")
    raise ValueError("Failed to generate summary guidance after 2 attempts")


def generate_summaries_batch(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    summary_guidance: dict,
    cache_alias: str | None = None,
    run_name: str | None = None,
    generate_preset: str = "summary_generate_gemini",
) -> list[dict]:
    """Generate structured summaries for multiple documents.

    Args:
        documents: List of document strings
        criterion: Criterion name
        criterion_description: Criterion description
        summary_guidance: Summary guidance dict
        cache_alias: Optional cache alias for inference calls
        run_name: Optional experiment/run name for cache organization
        generate_preset: Inference preset to use for summary generation
            (default: "summary_generate_gemini")

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

    criterion_description = (criterion_description or "").strip()

    # Prepare inputs
    inputs = {
        "document": documents,
        "criterion": [criterion] * len(documents),
        "criterion_description": [criterion_description] * len(documents),
        "summary_guidance": [guidance_str] * len(documents),
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config=generate_preset,
        cache_alias=cache_alias,
        run_name=run_name,
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
            # Result should be dict with annotation_trace and final_summary.
            # Be tolerant to minor schema variants (e.g., list-of-one).
            if (
                isinstance(result, list)
                and len(result) == 1
                and isinstance(result[0], dict)
            ):
                result = result[0]

            summary_dict = (
                result
                if isinstance(result, dict)
                else {"annotation_trace": "", "final_summary": str(result)}
            )
            annotations.append({"summary": summary_dict})

    logger.info(f"Generated summaries for {len(documents)} documents")
    return annotations
