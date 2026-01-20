"""Evaluation presets for document summarization, query relevance, and triplet selection."""

from __future__ import annotations

from ._base import InferenceConfig

EVALUATION_PRESETS = {
    # ========================================================================
    # DOCUMENT REWRITING / SUMMARIZATION
    # ========================================================================
    "rewrite_plaintext_freeform_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/rewrite/rewrite_plaintext_freeform.txt",
        parser="json",
        parser_kwargs={"annotation_key": "summary"},
        temperature=0.7,
        max_tokens=4096,
    ),
    # ========================================================================
    # DOCUMENT SUMMARIZATION (EVALUATION)
    # ========================================================================
    # Document summarization: generates criterion-focused summaries of documents
    "document_summary_gemini_flash_lite": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/eval/document_rewrite_summary.txt",
        parser="json",
        temperature=0.7,
        max_tokens=4096,
    ),
    "document_summary_gemini_flash": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/eval/document_rewrite_summary.txt",
        parser="json",
        temperature=0.7,
        max_tokens=4096,
    ),
    # Document-to-summaries: generates multiple summaries from documents for evaluation
    "document_to_summaries_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash-lite",
        prompt_template="prompts/eval/document_to_summaries.txt",
        parser="json",
        parser_kwargs={"annotation_key": "summaries"},
        temperature=0.7,
        max_tokens=4096,
    ),
    # ========================================================================
    # QUERY RELEVANCE SCORING (EVALUATION)
    # ========================================================================
    # Query relevance scoring: expands query into variations for scoring documents
    "query_relevance_scores_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/eval/query_expansion_variations.txt",
        parser="json",
        parser_kwargs={"annotation_key": "queries"},
        temperature=0.7,
        max_tokens=4096,
    ),
    # ========================================================================
    # SUMMARY HINT GENERATION
    # ========================================================================
    "summary_hint_generation_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-pro",
        prompt_template="prompts/schema/summary_hint_generation.txt",
        parser="delimiter",
        parser_kwargs={"delimiter": "---FINAL OUTPUT---"},
        temperature=0.7,
        max_tokens=8192,
    ),
    # ========================================================================
    # TRIPLET SELECTION
    # ========================================================================
    "triplet_select_positive_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/triplet/triplet_select_positive.txt",
        parser="json",
        parser_kwargs={"annotation_key": "chosen_positive"},
        temperature=0.5,
        max_tokens=8192,
        extra_kwargs={"response_mime_type": "application/json"},
    ),
    "triplet_select_negative_gemini": InferenceConfig(
        provider="gemini",
        model_name="gemini-2.5-flash",
        prompt_template="prompts/triplet/triplet_select_negative.txt",
        parser="json",
        parser_kwargs={"annotation_key": "chosen_negative"},
        temperature=0.5,
        max_tokens=8192,
        extra_kwargs={"response_mime_type": "application/json"},
    ),
}
