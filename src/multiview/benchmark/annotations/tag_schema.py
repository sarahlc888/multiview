"""Tag schema generation and application.

This module handles multi-label binary tag annotation (similar to lm_tags.py):
- Generate tag schemas (criterion-relevant and spurious)
- Apply tags to documents
"""

from __future__ import annotations

import logging
import random

from multiview.inference.inference import run_inference

logger = logging.getLogger(__name__)


def generate_tag_schema(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    n_samples: int = 10,
    schema_hint: str | None = None,
    is_spurious: bool = False,
    cache_alias: str | None = None,
) -> dict:
    """Generate a tag schema from sample documents.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name
        criterion_description: Description of what the criterion means
        n_samples: Number of documents to sample for schema generation
        schema_hint: Optional hint about what tags to create
        is_spurious: If True, generate spurious tags (surface-level properties)
        cache_alias: Optional cache alias for inference calls

    Returns:
        Tag schema dict with structure:
        {
            "tags": [
                {"name": "...", "description": "..."},
                ...
            ]
        }
    """
    # Sample documents
    sample_docs = random.sample(documents, min(n_samples, len(documents)))
    sample_docs_str = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(sample_docs))

    # Build prompt conditionally based on tag type
    from multiview.inference.presets import InferenceConfig

    if is_spurious:
        # Spurious tags prompt
        prompt_parts = [
            "You are designing a tag schema to identify SPURIOUS (surface-level) similarities between documents.",
            "",
            f"CRITERION OF INTEREST: {criterion}",
        ]

        if criterion_description:
            prompt_parts.extend(
                [
                    "",
                    "CRITERION DESCRIPTION:",
                    criterion_description,
                ]
            )

        prompt_parts.extend(
            [
                "",
                "SAMPLE DOCUMENTS:",
                sample_docs_str,
                "",
                "Your task is to create a tag schema that captures dimensions of variation that:",
                f'1. Are INDEPENDENT of the criterion "{criterion}" (not relevant to it)',
                "2. Capture superficial or surface-level properties of documents",
                "3. Could cause two documents to APPEAR similar even though they differ on the criterion",
                "4. Could be used to identify confounders or spurious correlations",
                "",
                "IMPORTANT:",
                "- Tags should be BINARY (yes/no, present/absent)",
                "- Tags should be INDEPENDENT of the criterion",
                "- Tags should capture SUPERFICIAL similarities",
                "- Aim for 5-10 tags that cover different aspects of spurious similarity",
                "- Each tag should be clearly defined",
                "",
                "Return valid JSON with reasoning:",
                "{",
                '  "reasoning": "Explain what spurious/superficial properties you identified and why these tags capture surface-level similarities independent of the criterion",',
                '  "tags": [{"name": "tag_name", "description": "when to apply this tag"}, ...]',
                "}",
            ]
        )
    else:
        # Regular criterion-relevant tags prompt
        prompt_parts = [
            "You are designing a tagging schema to annotate documents based on a specific lens/criteria.",
            "",
            f"CRITERIA: {criterion}",
        ]

        if criterion_description:
            prompt_parts.extend(
                [
                    "",
                    "CRITERIA DESCRIPTION:",
                    criterion_description,
                ]
            )

        if schema_hint:
            prompt_parts.extend(
                [
                    "",
                    "TAG SCHEMA HINT:",
                    schema_hint,
                ]
            )

        prompt_parts.extend(
            [
                "",
                f"Here are {len(sample_docs)} randomly sampled documents from the corpus (showing all fields):",
                "",
                sample_docs_str,
                "",
                f'Your task: Create a tagging schema with tags that capture different aspects of how documents relate to the criteria "{criterion}".',
                "",
                "IMPORTANT: Tags are NOT mutually exclusive. A document can have multiple tags or no tags. Think of tags as binary attributes.",
                "",
                "GUIDELINES:",
                "- Tags should be relevant to the criteria",
                "- Each tag should represent a single, clear attribute that either does or does not apply to a given document",
                "- There is no limit on the number of tags. Use as many as seems reasonable. A little bit of redundancy is OK, but use your judgement. The priority is to capture the range of variation across documents as much as possible.",
                "- If there is a finite, enumerable set of options or prototypes, enumerate them. For example, color -> [red, yellow, green, ...]",
                "- If the decision space can be factorized into independent attributes, do so.",
                "",
                "Think through multiple candidate tag schemas before choosing the final one.",
                "",
                "Output valid JSON with reasoning:",
                "{",
                '  "reasoning": "Explain what tag options you considered and why you chose these specific tags to capture variation along the criteria",',
                '  "tags": [{"name": "tag_name", "description": "when this tag applies"}, ...]',
                "}",
            ]
        )

    prompt = "\n".join(prompt_parts)

    # Prepare inputs with pre-built prompt
    inputs = {
        "prompt": [prompt],
    }

    custom_config = InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="{prompt}",
        parser="json",
        temperature=0.0,
        max_tokens=8192,
    )

    # Generate schema using inference
    results = run_inference(
        inputs=inputs,
        config=custom_config,
        cache_alias=cache_alias,
        verbose=True,
    )

    schema = results[0]

    # Handle case where json parser wraps result in list
    if isinstance(schema, list) and len(schema) > 0:
        schema = schema[0]

    if schema is None:
        raise ValueError(
            f"Failed to generate {'spurious ' if is_spurious else ''}tag schema"
        )

    logger.info(
        f"Generated {'spurious ' if is_spurious else ''}tag schema with {len(schema.get('tags', []))} tags"
    )
    return schema


def generate_spurious_tag_schema(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    n_samples: int = 10,
    cache_alias: str | None = None,
) -> dict:
    """Generate spurious tag schema (surface-level properties).

    Convenience wrapper around generate_tag_schema with is_spurious=True.

    Args:
        documents: List of document strings to sample from
        criterion: Criterion name
        criterion_description: Description of what the criterion means
        n_samples: Number of documents to sample for schema generation
        cache_alias: Optional cache alias for inference calls

    Returns:
        Spurious tag schema dict
    """
    return generate_tag_schema(
        documents=documents,
        criterion=criterion,
        criterion_description=criterion_description,
        n_samples=n_samples,
        schema_hint="Focus on surface-level properties independent of the criterion",
        is_spurious=True,
        cache_alias=cache_alias,
    )


def apply_tags(
    document: str,
    criterion: str,
    criterion_description: str,
    tag_schema: dict,
    cache_alias: str | None = None,
) -> dict:
    """Apply binary tags to a single document.

    Args:
        document: Document string
        criterion: Criterion name
        criterion_description: Criterion description
        tag_schema: Tag schema dict
        cache_alias: Optional cache alias for inference calls

    Returns:
        Dict with tags: {"tag1": true, "tag2": false, ...}
    """
    # Format schema for prompt
    tags = tag_schema.get("tags", [])
    tags_text = "\n".join([f"- {tag['name']}: {tag['description']}" for tag in tags])

    # Prepare inputs
    inputs = {
        "document": [document],
        "criterion": [criterion],
        "criterion_description": [criterion_description or ""],
        "tag_schema": [tags_text],
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config="tag_apply_gemini",
        cache_alias=cache_alias,
        verbose=False,
    )

    # Extract tags (convert 0/1 to boolean)
    result = results[0]
    if result is None:
        return {}

    tags_dict = result if isinstance(result, dict) else {}
    return {k: bool(v) for k, v in tags_dict.items()}


def apply_tags_batch(
    documents: list[str],
    criterion: str,
    criterion_description: str,
    tag_schema: dict,
    cache_alias: str | None = None,
) -> list[dict]:
    """Apply binary tags to multiple documents.

    Args:
        documents: List of document strings
        criterion: Criterion name
        criterion_description: Criterion description
        tag_schema: Tag schema dict
        cache_alias: Optional cache alias for inference calls

    Returns:
        List of annotation dicts:
        [
            {"tags": {"tag1": true, "tag2": false, ...}},
            ...
        ]
    """
    # Format schema for prompt
    tags = tag_schema.get("tags", [])
    tags_text = "\n".join([f"- {tag['name']}: {tag['description']}" for tag in tags])

    # Prepare inputs
    inputs = {
        "document": documents,
        "criterion": [criterion] * len(documents),
        "criterion_description": [criterion_description or ""] * len(documents),
        "tag_schema": [tags_text] * len(documents),
    }

    # Run inference
    results = run_inference(
        inputs=inputs,
        config="tag_apply_gemini",
        cache_alias=cache_alias,
        verbose=False,
    )

    # Extract annotations (convert 0/1 to boolean)
    annotations = []
    for result in results:
        if result is None:
            annotations.append({"tags": {}, "tag_reasoning": None})
        else:
            # Result now has reasoning and tags fields
            if isinstance(result, dict) and "tags" in result:
                tags_dict = result.get("tags", {})
                reasoning = result.get("reasoning", "")
            else:
                # Fallback for old format (plain dict of tags)
                tags_dict = result if isinstance(result, dict) else {}
                reasoning = ""

            tags_bool = {k: bool(v) for k, v in tags_dict.items()}
            annotations.append({"tags": tags_bool, "tag_reasoning": reasoning})

    logger.info(f"Applied tags to {len(documents)} documents")
    return annotations
