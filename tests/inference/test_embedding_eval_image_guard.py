"""Guards for embedding evaluation with text-only embedding providers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from multiview.eval.embeddings import evaluate_with_embeddings


def test_text_only_embedding_rejects_image_only_documents():
    documents = [
        {"text": "", "image_path": "/tmp/a.jpg"},
        {"text": "<image>", "image_path": "/tmp/b.jpg"},
        {"text": "has text", "image_path": "/tmp/c.jpg"},
    ]

    with pytest.raises(ValueError, match="text-only provider"):
        evaluate_with_embeddings(
            documents=documents,
            triplet_ids=[(0, 2, 2)],
            embedding_preset="hf_qwen3_embedding_8b",
        )


def test_text_only_embedding_rejects_placeholder_only_strings():
    documents = ["<image>", "<image>", "text"]

    with pytest.raises(ValueError, match="text-only provider"):
        evaluate_with_embeddings(
            documents=documents,
            triplet_ids=[(0, 2, 2)],
            embedding_preset="hf_qwen3_embedding_8b",
        )


@patch("multiview.eval.embeddings.run_inference")
def test_text_only_embedding_ignores_images_when_text_exists(mock_run_inference):
    mock_run_inference.return_value = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]
    documents = [
        {"text": "dress shoe", "image_path": "/tmp/a.jpg"},
        {"text": "formal shoe", "image_path": "/tmp/b.jpg"},
        {"text": "sneaker", "image_path": "/tmp/c.jpg"},
    ]

    evaluate_with_embeddings(
        documents=documents,
        triplet_ids=[(0, 1, 2)],
        embedding_preset="hf_qwen3_embedding_8b",
    )

    call_kwargs = mock_run_inference.call_args.kwargs
    assert call_kwargs["inputs"]["document"] == [
        "dress shoe",
        "formal shoe",
        "sneaker",
    ]
    assert "images" not in call_kwargs["inputs"]
