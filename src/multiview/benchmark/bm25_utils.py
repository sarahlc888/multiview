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
    """Tokenize text using Lucene-like rules.

    Features:
    - Unicode NFKC normalization (full-width chars, ligatures)
    - Lowercase conversion
    - Possessive removal ('s, s')
    - Stopword filtering (33 Lucene English stopwords)
    - Min length filtering

    Args:
        text: Input text
        remove_stopwords: Whether to filter stopwords
        min_token_length: Minimum token length (default: 2)

    Returns:
        List of tokens

    Example:
        >>> tokenize_lucene_like("The company's data")
        ['company', 'data']
        >>> tokenize_lucene_like("It's the students' work")
        ['student', 'work']
    """
    if not text or not isinstance(text, str):
        return []

    # Unicode normalization (handles full-width chars, ligatures)
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Tokenize on word boundaries, keep apostrophes
    # This keeps contractions like "we're" together
    tokens = re.findall(r"\b[\w']+\b", text)

    processed_tokens = []
    for token in tokens:
        # Remove possessives: "company's" → "company", "students'" → "students"
        if token.endswith("'s"):
            token = token[:-2]
        elif token.endswith("s'"):
            token = token[:-1]

        # Strip remaining leading/trailing apostrophes
        token = token.strip("'")

        # Filter by minimum length
        if len(token) < min_token_length:
            continue

        # Filter stopwords
        if remove_stopwords and token in LUCENE_STOPWORDS:
            continue

        processed_tokens.append(token)

    return processed_tokens


def compute_bm25_scores(
    documents: list[str],
    anchor_idx: int,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> np.ndarray:
    """Compute BM25 similarity scores from anchor to all documents.

    Args:
        documents: List of document texts
        anchor_idx: Index of anchor document (or -1 for last)
        tokenizer: Tokenization function (default: tokenize_lucene_like)

    Returns:
        Numpy array of BM25 scores (shape: [n_documents])

    Raises:
        ValueError: If documents is empty or all tokenize to empty

    Example:
        >>> docs = ["hello world", "world peace", "hello peace"]
        >>> scores = compute_bm25_scores(docs, anchor_idx=0)
        >>> scores[1] > scores[2]  # "world peace" more similar to "hello world"
        True
    """
    if tokenizer is None:
        tokenizer = tokenize_lucene_like

    if not documents:
        raise ValueError("documents cannot be empty")

    # Handle negative index
    if anchor_idx < 0:
        anchor_idx = len(documents) + anchor_idx

    if anchor_idx < 0 or anchor_idx >= len(documents):
        raise ValueError(
            f"anchor_idx {anchor_idx} out of range for {len(documents)} documents"
        )

    # Tokenize corpus
    tokenized_corpus = [tokenizer(doc) for doc in documents]

    # Validate
    if all(not tokens for tokens in tokenized_corpus):
        raise ValueError("All documents tokenized to empty")

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Get anchor query
    anchor_query = tokenized_corpus[anchor_idx]

    # Score all documents
    scores = bm25.get_scores(anchor_query)

    return scores


def compute_bm25_matrix(
    documents: list[str],
    tokenizer: Callable[[str], list[str]] | None = None,
) -> np.ndarray:
    """Compute full NxN BM25 similarity matrix.

    Precomputes all pairwise BM25 similarities. More efficient than
    calling compute_bm25_scores() multiple times when scoring many
    anchor-document pairs.

    Args:
        documents: List of document texts
        tokenizer: Tokenization function (default: tokenize_lucene_like)

    Returns:
        Numpy array of shape [n_documents, n_documents] where
        matrix[i][j] is BM25 score from document i to document j

    Raises:
        ValueError: If documents is empty or all tokenize to empty

    Example:
        >>> docs = ["hello world", "world peace", "hello peace"]
        >>> matrix = compute_bm25_matrix(docs)
        >>> matrix.shape
        (3, 3)
        >>> matrix[0][0] > matrix[0][1]  # Self-similarity is highest
        True
    """
    if tokenizer is None:
        tokenizer = tokenize_lucene_like

    if not documents:
        raise ValueError("documents cannot be empty")

    # Tokenize corpus once
    tokenized_corpus = [tokenizer(doc) for doc in documents]

    # Validate
    if all(not tokens for tokens in tokenized_corpus):
        raise ValueError("All documents tokenized to empty")

    # Build BM25 index once
    bm25 = BM25Okapi(tokenized_corpus)

    # Compute matrix
    n_docs = len(documents)
    matrix = np.zeros((n_docs, n_docs))

    for i in range(n_docs):
        anchor_query = tokenized_corpus[i]
        scores = bm25.get_scores(anchor_query)
        matrix[i] = scores

    return matrix
