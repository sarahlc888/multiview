"""Annotation schema generation presets.

This module defines presets for generating annotation schemas (categories, tags, summaries)
from sample documents. These schemas are used for multi-faceted document annotation.
"""

from ._base import InferenceConfig

# ============================================================================
# CATEGORY SCHEMA GENERATION
# ============================================================================

CATEGORY_SCHEMA_GENERATION_GEMINI_PROMPT = """You have a set of unlabeled documents and a criterion of interest. \
Your task is to design a category schema that captures how the documents vary with respect to the criterion.

CRITERION: {criterion}

CRITERION DESCRIPTION:
{criterion_description}

SCHEMA HINT (optional):
{schema_hint}

SAMPLE DOCUMENTS:
{sample_documents}

Think about what kind of candidate schemas are possible.
- The ideal candidate schema partitions the output space into discrete categories.
- If there is a way to enumerate a closed set of categories, that would be best, but if not, it's OK to include an 'other' category.
- Try to choose a number of categories that reflects the range of variation within the sample documents.

Choose the single best schema strategy and return it in valid JSON with reasoning:
{{
  "reasoning": "Explain your schema choice: what alternatives you considered and why this schema best captures variation along the criteria",
  "categories": [{{"name": "...", "description": "..."}}, ...]
}}"""

CATEGORY_SCHEMA_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template=CATEGORY_SCHEMA_GENERATION_GEMINI_PROMPT,
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)

# ============================================================================
# TAG SCHEMA GENERATION
# ============================================================================

TAG_SCHEMA_GENERATION_GEMINI_PROMPT = """You are designing a tagging schema to annotate documents based on a specific lens/criteria.

CRITERIA: {criterion}

CRITERIA DESCRIPTION:
{criterion_description}

TAG SCHEMA HINT (optional):
{schema_hint}

Here are {n_samples} randomly sampled documents from the corpus (showing all fields):

{sample_documents}

Your task: Create a tagging schema with tags that capture different aspects of how documents relate to the criteria "{criterion}".

IMPORTANT: Tags are NOT mutually exclusive. A document can have multiple tags or no tags. Think of tags as binary attributes.

GUIDELINES:
- Tags should be relevant to the criteria
- Each tag should represent a single, clear attribute that either does or does not apply to a given document
- There is no limit on the number of tags. Use as many as seems reasonable. A little bit of redundancy is OK, but use your judgement. The priority is to capture the range of variation across documents as much as possible.
- If there is a finite, enumerable set of options or prototypes, enumerate them. For example, color -> [red, yellow, green, ...]
- If the decision space can be factorized into independent attributes, do so. For example, weather -> [temperature_hot, temperature_mild, temperature_cold, precipitation_rain, precipitation_none, precipitation_snow, ...]. Another example: sports team records for a 10-game season -> [game1_win, game1_loss, game2_win, game2_loss, ...]. Another example: action_sequence with actions {{A,B,C,None}} → [step1_A, step1_B, step1_C, step1_None, step2_A, ...]

Think through multiple candidate tag schemas before choosing the final one.

Output valid JSON with reasoning:
{{
  "reasoning": "Explain what tag options you considered and why you chose these specific tags to capture variation along the criteria",
  "tags": [
    {{"name": "tag_name", "description": "when this tag applies"}},
    ...
  ]
}}"""

TAG_SCHEMA_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template=TAG_SCHEMA_GENERATION_GEMINI_PROMPT,
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)

# ============================================================================
# SPURIOUS TAG SCHEMA GENERATION
# ============================================================================

SPURIOUS_TAG_SCHEMA_GENERATION_GEMINI_PROMPT = """You are designing a tag schema to identify SPURIOUS (surface-level) similarities between documents.

CRITERION OF INTEREST: {criterion}

CRITERION DESCRIPTION:
{criterion_description}

SAMPLE DOCUMENTS:
{sample_documents}

Your task is to create a tag schema that captures dimensions of variation that:
1. Are INDEPENDENT of the criterion "{criterion}" (not relevant to it)
2. Capture superficial or surface-level properties of documents
3. Could cause two documents to APPEAR similar even though they differ on the criterion
4. Could be used to identify confounders or spurious correlations

For example, if the criterion is "mathematical operations used", spurious tags might include:
- Word problem context (shopping, time, school, etc.)
- Presence of specific entities (money, people, objects)
- Problem length (short, medium, long)
- Numeric ranges used (small numbers, large numbers)

These are properties that don't affect what operations are used, but might cause superficial similarity.

IMPORTANT:
- Tags should be BINARY (yes/no, present/absent)
- Tags should be INDEPENDENT of the criterion
- Tags should capture SUPERFICIAL similarities
- Aim for 5-10 tags that cover different aspects of spurious similarity
- Each tag should be clearly defined

Return valid JSON with reasoning:
{{
  "reasoning": "Explain what spurious/superficial properties you identified and why these tags capture surface-level similarities independent of the criterion",
  "tags": [
    {{"name": "tag_name", "description": "when to apply this tag"}},
    ...
  ]
}}"""

SPURIOUS_TAG_SCHEMA_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template=SPURIOUS_TAG_SCHEMA_GENERATION_GEMINI_PROMPT,
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)

# ============================================================================
# SUMMARY GUIDANCE GENERATION
# ============================================================================

SUMMARY_GUIDANCE_GENERATION_GEMINI_PROMPT = """You are designing annotation guidance for free-text summaries.

CRITERION: {criterion}

CRITERION DESCRIPTION:
{criterion_description}

SUMMARY HINT (optional):
{guidance_hint}

DESIRED FORMAT (optional):
{format_hint}

SAMPLE DOCUMENTS:
{sample_documents}

The summary should include two sections:
1) Annotation trace: freeform text with justification and references to the document.
2) Final summary: concise, structured, high lexical overlap for similar documents; invariant to spurious factors; should not reference the specifics of the document unless there's no other way to illustrate the point.

If a desired format is specified above, the final summary should follow that format.

Include a short demonstration showing how to produce both fields for one example document.

Think through multiple candidate guidance options, then choose the single best option.
Return valid JSON with reasoning:
{{
  "reasoning": "Explain what guidance options you considered and why this approach best helps annotators create useful summaries",
  "summary_guidance": "..."
}}"""

SUMMARY_GUIDANCE_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template=SUMMARY_GUIDANCE_GENERATION_GEMINI_PROMPT,
    parser="json",
    temperature=0.0,
    max_tokens=8192,
)
