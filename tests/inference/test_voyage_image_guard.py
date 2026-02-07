"""Guards for Voyage providers when image payloads are present."""

from __future__ import annotations

import pytest

from multiview.inference.providers.voyage import (
    voyage_embedding_completions,
    voyage_reranker_completions,
)


def test_voyage_reranker_rejects_image_payloads():
    with pytest.raises(ValueError, match="does not support image inputs"):
        voyage_reranker_completions(
            prompts=["doc"],
            model_name="rerank-2.5-lite",
            queries=["query"],
            images=["/tmp/example.png"],
        )


def test_voyage_embedding_rejects_image_payloads():
    with pytest.raises(ValueError, match="text-only"):
        voyage_embedding_completions(
            prompts=["doc"],
            model_name="voyage-4-lite",
            images=["/tmp/example.png"],
        )


def test_voyage_reranker_allows_empty_image_payloads(monkeypatch):
    def _boom():
        raise RuntimeError("client creation reached")

    monkeypatch.setattr("multiview.inference.providers.voyage._get_voyage_client", _boom)

    with pytest.raises(RuntimeError, match="client creation reached"):
        voyage_reranker_completions(
            prompts=["doc"],
            model_name="rerank-2.5-lite",
            queries=["query"],
            images=[None, []],
        )
