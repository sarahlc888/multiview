"""Criteria description generation presets.

This module defines presets for generating detailed descriptions of similarity criteria
from sample documents.
"""

from ._base import InferenceConfig

CRITERIA_DESCRIPTION_GENERATION_GEMINI_PROMPT = """Below are a few randomly sampled documents from an unlabeled corpus.

We are going to analyze these documents using this criteria: "{criterion}"

CRITERIA CONTEXT:
{criterion_description}

Our downstream goal is to create a set of similarity scores using comparisons like:
"given document A and document B, how similar are they based on this criteria: {criterion}".

However, the criteria is just a short free text string, which could be somewhat ambiguous.
In order to make it easier to rate criteria-specific similarity, we need to provide a more
detailed description.

Your task is to create this description.

For example, for a straightforward criteria like "color" for a dataset of images, we could
create a criteria description such as "What are the dominant color or colors in the image,
e.g. red, orange, yellow, green, blue, purple, grey, white, black, brown". Since color
is a property that has discrete categories, the description lists out salient categories
almost like class labels.

For a more complex criteria like "literary themes" for a dataset of short stories, we could
create a criteria description such as: "literary themes: what are some of the biggest thematic
preoccupations of the story? Does the story relate to any universal themes like love,
mortality, greed? Does the story make reference to any well-known prior works?" Even
though there are not clear "class labels" to use here, it is useful to provide examples
of some archetypes that could appear.

SAMPLE DOCUMENTS:
{sample_documents}

Create a criteria description for "{criterion}" that will help compare documents for similarity.
Be specific and provide examples where helpful. Keep it under 100 words.

CRITERIA DESCRIPTION:"""

CRITERIA_DESCRIPTION_GENERATION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-pro",
    prompt_template=CRITERIA_DESCRIPTION_GENERATION_GEMINI_PROMPT,
    parser="text",
    temperature=0.0,
    max_tokens=8192,
)
