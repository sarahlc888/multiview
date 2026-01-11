"""Document annotation utilities.

This package provides functions for annotating documents with:
- Category classification (class_schema.py)
- Multi-label binary tags (tag_schema.py)
- Open-ended summaries (open_ended.py)
- All annotation types combined (union_all.py)

Module Organization:
    class_schema.py: Single-category classification
        - generate_category_schema(): Generate category schema from samples
        - classify_documents_batch(): Classify multiple documents

    tag_schema.py: Multi-label binary tag annotation
        - generate_tag_schema(): Generate tag schema from samples
        - generate_spurious_tag_schema(): Generate spurious tag schema
        - apply_tags_batch(): Apply tags to multiple documents

    open_ended.py: Open-ended summary generation
        - generate_pairwise_sim_hint(): Generate pairwise similarity hint
        - generate_summary_guidance(): Generate summary guidance from samples
        - generate_summaries_batch(): Generate summaries for multiple documents

    union_all.py: Unified multi-faceted annotation (⭐ Main Entry Point)
        - annotate_with_lm_all(): Orchestrates all annotation types together
          * Generates schemas from sample documents
          * Applies all annotation types (categories, tags, summaries)
          * Returns rich annotations combining all information
"""

import logging

# Schema generation
# Batch annotation
from multiview.benchmark.annotations.class_schema import (
    classify_documents_batch,
    generate_category_schema,
)
from multiview.benchmark.annotations.open_ended import (
    generate_pairwise_sim_hint,
    generate_summaries_batch,
    generate_summary_guidance,
)
from multiview.benchmark.annotations.tag_schema import (
    apply_tags_batch,
    generate_spurious_tag_schema,
    generate_tag_schema,
)

# Main "all" annotation (from union_all.py, mirrors lm_all.py from clean_slate)
from multiview.benchmark.annotations.union_all import annotate_with_lm_all

logger = logging.getLogger(__name__)


def annotate_with_known_criterion(
    documents: list[str],
    document_set,
    criterion: str,
) -> list[dict]:
    """Annotate documents using known criterion extraction.

    Args:
        documents: List of document strings
        document_set: DocumentSet instance with extraction method
        criterion: Criterion name to extract

    Returns:
        List of annotation dicts with criterion_value
    """
    annotations = []
    for doc in documents:
        value = document_set.get_known_criterion_value(doc, criterion)
        annotations.append({"criterion_value": value})

    logger.debug(f"Extracted {criterion} for {len(documents)} documents")
    return annotations


__all__ = [
    # Main entry points
    "annotate_with_known_criterion",
    "annotate_with_lm_all",
    # Schema generation
    "generate_category_schema",
    "generate_tag_schema",
    "generate_spurious_tag_schema",
    "generate_pairwise_sim_hint",
    "generate_summary_guidance",
    # Batch annotation
    "classify_documents_batch",
    "apply_tags_batch",
    "generate_summaries_batch",
]
