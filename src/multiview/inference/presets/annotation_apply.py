"""Annotation application presets.

This module defines presets for applying annotation schemas (categories, tags, summaries)
to individual documents.
"""

from ._base import InferenceConfig

# ============================================================================
# CATEGORY CLASSIFICATION
# ============================================================================

CATEGORY_CLASSIFY_GEMINI_PROMPT = """Classify this document into one of the given categories based on the criteria.

CRITERIA: {criterion}

CRITERIA DESCRIPTION:
{criterion_description}

CATEGORIES:
{category_schema}

DOCUMENT (all fields):
{document}

Use the category definitions strictly. If numeric fields are present, classify based primarily on those values.

Output valid JSON with reasoning:
{{
  "reasoning": "Briefly explain why this document fits this category based on the criteria",
  "category": "exact_category_name"
}}"""

CATEGORY_CLASSIFY_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=CATEGORY_CLASSIFY_GEMINI_PROMPT,
    parser="json",
    temperature=0.0,
    max_tokens=2048,
)

# ============================================================================
# TAG APPLICATION
# ============================================================================

TAG_APPLY_GEMINI_PROMPT = """Determine which tags apply to this document based on the criteria.

CRITERIA: {criterion}

CRITERIA DESCRIPTION:
{criterion_description}

TAGS:
{tag_schema}

DOCUMENT (all fields):
{document}

For each tag, output 1 if it applies to this document, 0 if it does not.
Output as a JSON object with reasoning and tags:
{{
  "reasoning": "Briefly explain your tag decisions, especially for tags that apply",
  "tags": {{"tag_name": 0 or 1, ...}}
}}"""

TAG_APPLY_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=TAG_APPLY_GEMINI_PROMPT,
    parser="json",
    temperature=0.0,
    max_tokens=2048,
)

# ============================================================================
# SUMMARY GENERATION
# ============================================================================

SUMMARY_GENERATE_GEMINI_PROMPT = """We are analyzing documents through this criteria: "{criterion}"

Criteria description: {criterion_description}

Document:
{document}

Summarize how this document relates to the criteria in 1-3 sentences. Focus only on aspects relevant to the criteria. Be specific and concrete.

Return JSON with two keys: "annotation_trace" and "final_summary".

ADDITIONAL SUMMARY GUIDANCE:
{summary_guidance}

IMPORTANT: The final_summary MUST follow the exact format specified in the guidance above. Do not write prose descriptions.

ANNOTATION:"""

SUMMARY_GENERATE_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=SUMMARY_GENERATE_GEMINI_PROMPT,
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)
