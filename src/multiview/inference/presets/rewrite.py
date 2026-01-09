"""Document rewriting presets.

This module defines presets for rewriting/summarizing documents based on criteria.
"""

from ._base import InferenceConfig

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
