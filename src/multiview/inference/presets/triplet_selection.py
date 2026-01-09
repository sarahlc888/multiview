"""Triplet selection presets.

This module defines presets for LM judges that select positive and negative examples
from a pool of candidates during triplet creation. This is used to build high-quality
training data for similarity models.
"""

from ._base import InferenceConfig

TRIPLET_SELECTION_GEMINI_PROMPT = """We are finding similar documents based on this criteria: "{criterion}"

Criteria description: {criterion_description}

The gold standard would be a candidate that has exactly the same set of tags and a very similar summary.

UNDERSTANDING THE SIMILARITY SCORES:
- "True Tag Similarity": Jaccard similarity based on criterion-relevant tags (HIGH = similar on the criterion)
- "Spurious Tag Similarity": Jaccard similarity based on superficial properties (HIGH = looks similar but may differ on criterion)
- A good POSITIVE has HIGH true tag similarity
- A good HARD NEGATIVE has LOW true tag similarity but HIGH spurious tag similarity (looks similar, actually different)

{triplet_example_section}

ANCHOR DOCUMENT:
{anchor_doc}
ANCHOR ANNOTATION:
{anchor_annotation}

CANDIDATES (pick one as positive, one as negative relative to anchor):

{candidates}

Based on the criteria "{criterion}", which candidate should be the positive example?
Which candidate should be the negative example?

IMPORTANT:
- Pick a HARD POSITIVE: similar to the anchor on the criterion (high true tag similarity), but not trivially identical (avoid exact duplicates).
- Pick a HARD NEGATIVE: Choose a candidate that differs on the criterion but is NOT obviously different. For example:
  * PREFER candidates with LOW true tag similarity but HIGH spurious tag similarity
  * If candidates have category labels, PREFER negatives from the SAME category as the anchor
  * The negative should be subtly different on the criterion, not maximally different
  * Avoid selecting negatives that are from completely different categories or have completely unrelated content
  * A good hard negative would fool a simple model but fail under careful criterion-specific analysis

Explain your reasoning briefly, then answer on separate lines:
CHOSEN POSITIVE: [number]
CHOSEN NEGATIVE: [number]"""

TRIPLET_SELECTION_GEMINI = InferenceConfig(
    provider="gemini",
    model_name="gemini-2.5-flash-lite",
    prompt_template=TRIPLET_SELECTION_GEMINI_PROMPT,
    parser="text",
    temperature=0.0,
    max_tokens=4096,
)
