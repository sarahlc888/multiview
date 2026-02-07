import pytest

from multiview.eval.document_summary import _generate_summaries
from multiview.eval.embeddings import evaluate_with_embeddings
from multiview.eval.generation_utils import generate_text_variations_from_documents


def test_generate_text_variations_requires_non_empty_criterion_description():
    with pytest.raises(ValueError, match="Missing criterion_description"):
        generate_text_variations_from_documents(
            documents=["doc"],
            criterion="functional_type",
            criterion_description="   ",
            num_variations=2,
            generation_preset="document_to_summaries_gemini",
            cache_alias=None,
            run_name=None,
        )


def test_generate_summaries_requires_non_empty_criterion_description():
    with pytest.raises(ValueError, match="Missing criterion_description"):
        _generate_summaries(
            documents=["doc"],
            criterion="functional_type",
            criterion_description="",
            summary_preset="document_summary_gemini_flash_lite",
            cache_alias=None,
            run_name=None,
        )


def test_embeddings_require_non_empty_criterion_description_when_criterion_given():
    with pytest.raises(ValueError, match="Missing criterion_description"):
        evaluate_with_embeddings(
            documents=["a", "b", "c"],
            triplet_ids=[(0, 1, 2)],
            embedding_preset="openai_embedding_small",
            criterion="functional_type",
            criterion_description="",
        )
