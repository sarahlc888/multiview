"""BM25 utilities with advanced tokenization.

This module provides reusable BM25 scoring functions with sophisticated
tokenization for use in candidate selection and triplet evaluation.
"""

import re
import unicodedata
from collections.abc import Callable

import numpy as np
from rank_bm25 import BM25Okapi

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
    documents: list[str],
    anchor_idx: int,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> np.ndarray:
    """Compute BM25 similarity scores from anchor to all documents."""
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

    tokenized_corpus = [tokenizer(doc) for doc in documents]
    if all(not tokens for tokens in tokenized_corpus):
        raise ValueError("All documents tokenized to empty")

    bm25 = BM25Okapi(tokenized_corpus)
    anchor_query = tokenized_corpus[anchor_idx]
    scores = bm25.get_scores(anchor_query)
    return scores


def compute_bm25_matrix(
    documents: list[str],
    tokenizer: Callable[[str], list[str]] | None = None,
) -> np.ndarray:
    """Compute full NxN BM25 similarity matrix."""
    if tokenizer is None:
        tokenizer = tokenize_lucene_like

    if not documents:
        raise ValueError("documents cannot be empty")

    tokenized_corpus = [tokenizer(doc) for doc in documents]
    if all(not tokens for tokens in tokenized_corpus):
        raise ValueError("All documents tokenized to empty")

    bm25 = BM25Okapi(tokenized_corpus)

    n_docs = len(documents)
    matrix = np.zeros((n_docs, n_docs))

    for i in range(n_docs):
        anchor_query = tokenized_corpus[i]
        scores = bm25.get_scores(anchor_query)
        matrix[i] = scores

    return matrix
