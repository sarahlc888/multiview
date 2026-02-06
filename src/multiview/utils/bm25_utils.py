"""BM25 utilities with advanced tokenization.

This module provides reusable BM25 scoring functions with sophisticated
tokenization for use in candidate selection and triplet evaluation.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

import numpy as np
from rank_bm25 import BM25Okapi


def _extract_text(doc: str | dict) -> str:
    """Extract text from document (handles both str and dict)."""
    if isinstance(doc, str):
        return doc
    return doc.get("text", "")


# Lucene's default English stopwords
LUCENE_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    ]
)


def tokenize_lucene_like(
    text: str,
    remove_stopwords: bool = True,
    min_token_length: int = 2,
) -> list[str]:
    """Tokenize text using Lucene-like rules."""
    if not text or not isinstance(text, str):
        return []

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    tokens = re.findall(r"\b[\w']+\b", text)

    processed_tokens = []
    for token in tokens:
        if token.endswith("'s"):
            token = token[:-2]
        elif token.endswith("s'"):
            token = token[:-1]

        token = token.strip("'")

        if len(token) < min_token_length:
            continue
        if remove_stopwords and token in LUCENE_STOPWORDS:
            continue

        processed_tokens.append(token)

    return processed_tokens


def compute_bm25_scores(
    documents: list[str | dict],
    anchor_idx: int,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> np.ndarray:
    """Compute BM25 similarity scores from anchor to all documents.

    Args:
        documents: List of documents (strings or dicts with 'text' field)
        anchor_idx: Index of the anchor document (query)
        tokenizer: Optional tokenizer function

    Returns:
        Array of BM25 scores (one per document), or zeros if no text
    """
    if tokenizer is None:
        tokenizer = tokenize_lucene_like

    if not documents:
        raise ValueError("documents cannot be empty")

    if anchor_idx < 0:
        anchor_idx = len(documents) + anchor_idx

    if anchor_idx < 0 or anchor_idx >= len(documents):
        raise ValueError(
            f"anchor_idx {anchor_idx} out of range for {len(documents)} documents"
        )

    # Extract text from documents (handles both str and dict)
    texts = [_extract_text(doc) for doc in documents]

    # Tokenize
    tokenized_corpus = [tokenizer(text) for text in texts]

    # Check if documents have sufficient tokens
    if all(not tokens for tokens in tokenized_corpus):
        # Return zeros - no text similarity
        return np.zeros(len(documents))

    bm25 = BM25Okapi(tokenized_corpus)
    anchor_query = tokenized_corpus[anchor_idx]
    scores = bm25.get_scores(anchor_query)
    return scores


def compute_bm25_matrix(
    documents: list[str | dict],
    tokenizer: Callable[[str], list[str]] | None = None,
) -> np.ndarray:
    """Compute full NxN BM25 similarity matrix.

    Args:
        documents: List of documents (strings or dicts with 'text' field)
        tokenizer: Optional tokenizer function

    Returns:
        NxN matrix of BM25 scores
    """
    if tokenizer is None:
        tokenizer = tokenize_lucene_like

    if not documents:
        raise ValueError("documents cannot be empty")

    # Extract text from documents (handles both str and dict)
    texts = [_extract_text(doc) for doc in documents]

    # Tokenize
    tokenized_corpus = [tokenizer(text) for text in texts]

    # Check if documents have sufficient tokens
    if all(not tokens for tokens in tokenized_corpus):
        # Return zeros matrix - no text similarity
        return np.zeros((len(documents), len(documents)))

    bm25 = BM25Okapi(tokenized_corpus)

    n_docs = len(documents)
    matrix = np.zeros((n_docs, n_docs))

    for i in range(n_docs):
        anchor_query = tokenized_corpus[i]
        scores = bm25.get_scores(anchor_query)
        matrix[i] = scores

    return matrix
