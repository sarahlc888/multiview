"""BM25 lexical retrieval presets."""

from __future__ import annotations

from ._base import InferenceConfig

BM25_PRESETS = {
    "bm25_lexical": InferenceConfig(
        provider="bm25",
        model_name="bm25",
        prompt_template="{document}",
        parser="score",
    ),
    "bm25_raw": InferenceConfig(
        provider="bm25",
        model_name="bm25",
        prompt_template="{document}",
        parser="score",
    ),
}
