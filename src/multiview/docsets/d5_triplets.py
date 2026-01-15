"""Custom triplet creation for D5 Applicability dataset.

D5_applicability creates property-text matching triplets in a joint embedding space:
- Anchor can be a property OR a text (headline/description)
- If anchor is a property: positive is applicable text, negative is non-applicable text
- If anchor is a text: positive is applicable property, negative is non-applicable property

TRIPLET EXAMPLES:
-----------------

Property-anchored (anchor is property, pos/neg are texts):
  Anchor: property: highlight struggles of certain industries, such as aviation
  Positive: headline: crew of stranded coal ship drinking tainted water
    → Property IS applicable to this text
  Negative: description: sports scores today
    → Property is NOT applicable to this text

Text-anchored (anchor is text, pos/neg are properties):
  Anchor: headline: north korea fires projectiles south korea military says
  Positive: property: discuss politics and government responses
    → Property IS applicable to this text
  Negative: property: discuss criminal cases, such as trials
    → Property is NOT applicable to this text

The triplet creation function generates a mix of both types, creating a joint
embedding space where properties and applicable texts are close together.

ALGORITHM:
----------
1. For each property: Find applicable/non-applicable texts, create triplet
2. For each text: Find applicable/non-applicable properties, create triplet
3. Shuffle and limit to max_triplets
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiview.docsets.d5_applic import D5ApplicabilityDocSet

logger = logging.getLogger(__name__)


def create_d5_applicability_triplets(
    documents: list[Any],
    docset: D5ApplicabilityDocSet,
    max_triplets: int | None = None,
    selection_strategy: str = "hard_negatives",
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Create property-text matching triplets for D5 applicability.

    Creates triplets in a joint embedding space:
    - Anchor is property → pos is applicable text, neg is non-applicable text
    - Anchor is text → pos is applicable property, neg is non-applicable property

    Args:
        documents: List of all documents (properties + texts)
        docset: D5ApplicabilityDocSet instance with applicability matrix
        max_triplets: Maximum number of triplets to create
        selection_strategy: Strategy for selecting negatives (currently ignored)
        seed: Random seed for reproducibility

    Returns:
        List of (anchor_idx, positive_idx, negative_idx) tuples
    """
    random.seed(seed)

    # Separate properties and texts
    property_indices = list(docset.doc_idx_to_property_idx.keys())
    text_indices = list(docset.doc_idx_to_text_idx.keys())

    logger.info(
        f"Document distribution:\n"
        f"  Properties: {len(property_indices)}\n"
        f"  Texts: {len(text_indices)}"
    )

    # Create triplets
    triplets = []

    # Strategy 1: Anchor is property, pos/neg are texts
    for anchor_doc_idx in property_indices:
        property_idx = docset.doc_idx_to_property_idx[anchor_doc_idx]

        # Find applicable and non-applicable texts for this property
        applicable_texts = []
        not_applicable_texts = []

        for text_doc_idx in text_indices:
            text_idx = docset.doc_idx_to_text_idx[text_doc_idx]
            applicability = docset.get_applicability(property_idx, text_idx)

            if applicability == "applicable":
                applicable_texts.append(text_doc_idx)
            else:
                not_applicable_texts.append(text_doc_idx)

        # Create triplet if we have both applicable and non-applicable texts
        if applicable_texts and not_applicable_texts:
            pos_idx = random.choice(applicable_texts)
            neg_idx = random.choice(not_applicable_texts)
            triplets.append((anchor_doc_idx, pos_idx, neg_idx))

    # Strategy 2: Anchor is text, pos/neg are properties
    for anchor_doc_idx in text_indices:
        text_idx = docset.doc_idx_to_text_idx[anchor_doc_idx]

        # Find applicable and non-applicable properties for this text
        applicable_properties = []
        not_applicable_properties = []

        for property_doc_idx in property_indices:
            property_idx = docset.doc_idx_to_property_idx[property_doc_idx]
            applicability = docset.get_applicability(property_idx, text_idx)

            if applicability == "applicable":
                applicable_properties.append(property_doc_idx)
            else:
                not_applicable_properties.append(property_doc_idx)

        # Create triplet if we have both applicable and non-applicable properties
        if applicable_properties and not_applicable_properties:
            pos_idx = random.choice(applicable_properties)
            neg_idx = random.choice(not_applicable_properties)
            triplets.append((anchor_doc_idx, pos_idx, neg_idx))

    # Shuffle and limit
    random.shuffle(triplets)
    if max_triplets is not None and len(triplets) > max_triplets:
        triplets = triplets[:max_triplets]

    logger.info(
        f"Created {len(triplets)} property-text matching triplets\n"
        f"  (mixture of property-anchored and text-anchored triplets)"
    )

    return triplets
