"""Guards for Gemini embedding provider when image payloads are present."""

from __future__ import annotations

import pytest

from multiview.inference.providers.gemini import gemini_embedding_completions


def test_gemini_embedding_rejects_image_payloads():
    with pytest.raises(ValueError, match="text-only"):
        gemini_embedding_completions(
            prompts=["doc"],
            model_name="gemini-embedding-001",
            images=["/tmp/example.png"],
        )


def test_gemini_embedding_allows_empty_image_payloads(monkeypatch):
    # Ensure we pass the guard and reach provider initialization path.
    monkeypatch.setattr("multiview.inference.providers.gemini.GEMINI_API_KEY", None)

    with pytest.raises(
        ValueError,
        match="GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set",
    ):
        gemini_embedding_completions(
            prompts=["doc"],
            model_name="gemini-embedding-001",
            images=[None, []],
        )
