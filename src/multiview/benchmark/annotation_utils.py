"""Utilities for annotating documents with criterion values."""

import logging

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


def annotate_with_lm(
    documents: list[str],
    criterion: str,
) -> list[dict]:
    """Annotate documents using language model.

    Args:
        documents: List of document strings
        criterion: Criterion to annotate

    Returns:
        List of annotation dicts with criterion_value
    """
    # TODO: Implement LM-based annotation
    logger.warning(f"{criterion} is not a known criterion.")
    logger.warning("LM-based annotation not yet implemented.")

    # For now, return None values
    return [{"criterion_value": None} for _ in documents]
