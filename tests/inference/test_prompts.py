"""Tests for prompt formatting and PromptCollection.

Tests cover:
- PromptCollection attributes and methods
- Prompt deduplication
- Input broadcasting
- Aliasing (documents -> document, criteria -> criterion)
"""

import pytest

from multiview.inference.presets import InferenceConfig
from multiview.inference.prompts import PromptCollection, format_prompts


class TestPromptCollection:
    """Test PromptCollection class."""

    def test_basic_attributes(self):
        """Test that PromptCollection has expected attributes."""
        pc = PromptCollection(
            packed_prompts=["packed1", "packed2"],
            prompts=["prompt1", "prompt2"],
        )

        assert hasattr(pc, "packed_prompts")
        assert hasattr(pc, "prompts")
        assert pc.packed_prompts == ["packed1", "packed2"]
        assert pc.prompts == ["prompt1", "prompt2"]
        # This was the bug - accessing non_packed_prompts as attribute
        assert not hasattr(pc, "non_packed_prompts")

    def test_to_dict_method(self):
        """Test PromptCollection.to_dict() returns correct structure."""
        pc = PromptCollection(
            packed_prompts=["packed1", "packed2"],
            prompts=["prompt1", "prompt2"],
            embed_query_instr=["instr1", "instr2"],
        )

        result = pc.to_dict()

        assert isinstance(result, dict)
        assert "packed_prompts" in result
        assert "non_packed_prompts" in result  # This is the dict key
        assert result["packed_prompts"] == ["packed1", "packed2"]
        assert result["non_packed_prompts"] == ["prompt1", "prompt2"]
        assert result["embed_query_instrs"] == ["instr1", "instr2"]

    def test_deduplication(self):
        """Test prompt deduplication."""
        pc = PromptCollection(
            packed_prompts=["prompt1", "prompt2", "prompt1", "prompt3"],
            prompts=["p1", "p2", "p1", "p3"],
        )

        remap_idxs, deduped = pc.dedup()

        # Should have 3 unique prompts (prompt1, prompt2, prompt3)
        assert len(deduped.packed_prompts) == 3
        assert len(deduped.prompts) == 3

        # Remap should map back to original order
        assert len(remap_idxs) == 4
        assert remap_idxs[0] == remap_idxs[2]  # Both are "prompt1"

    def test_deduplication_preserves_order(self):
        """Test that deduplication preserves first occurrence order."""
        pc = PromptCollection(
            packed_prompts=["a", "b", "a", "c", "b"],
            prompts=["pa", "pb", "pa", "pc", "pb"],
        )

        remap_idxs, deduped = pc.dedup()

        # Should keep first occurrences: a, b, c
        assert deduped.packed_prompts == ["a", "b", "c"]
        assert deduped.prompts == ["pa", "pb", "pc"]

        # Remap should correctly map duplicates
        assert remap_idxs == [0, 1, 0, 2, 1]


class TestFormatPrompts:
    """Test format_prompts function."""

    def test_basic_formatting(self):
        """Test basic prompt formatting."""
        inputs = {"text": ["hello", "world"]}
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",
            prompt_template="Say: {text}",
            parser="text",
        )

        pc = format_prompts(inputs, config)

        assert len(pc.prompts) == 2
        assert pc.prompts[0] == "Say: hello"
        assert pc.prompts[1] == "Say: world"

    def test_formatting_allows_literal_json_braces(self):
        """Prompt templates often contain JSON examples with literal braces."""
        inputs = {"criterion": ["arithmetic"], "sample_documents": ["[1] a\n\n[2] b"]}
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",
            prompt_template=(
                "CRITERION: {criterion}\n\n"
                "SAMPLE DOCUMENTS:\n{sample_documents}\n\n"
                "Return valid JSON:\n"
                "{\n"
                '  "reasoning": "Explain why",\n'
                '  "categories": [{"name": "...", "description": "..."}]\n'
                "}\n"
            ),
            parser="text",
        )

        pc = format_prompts(inputs, config)
        assert len(pc.prompts) == 1
        assert "CRITERION: arithmetic" in pc.prompts[0]
        assert '"reasoning"' in pc.prompts[0]
        assert pc.prompts[0].count("{") >= 2
        assert pc.prompts[0].count("}") >= 2

    def test_singleton_broadcasting(self):
        """Test that singleton inputs are broadcasted."""
        inputs = {
            "documents": ["doc1", "doc2", "doc3"],
            "criterion": ["quality"],  # Singleton, should be broadcasted
        }
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",
            prompt_template="Doc: {document}\nCriterion: {criterion}",
            parser="text",
        )

        pc = format_prompts(inputs, config)

        assert len(pc.prompts) == 3
        # All should have the same criterion
        assert "Criterion: quality" in pc.prompts[0]
        assert "Criterion: quality" in pc.prompts[1]
        assert "Criterion: quality" in pc.prompts[2]

    def test_input_key_aliasing(self):
        """Test that 'documents' is aliased to 'document'."""
        inputs = {"documents": ["text1", "text2"]}
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",
            prompt_template="Analyze: {document}",  # Uses singular
            parser="text",
        )

        pc = format_prompts(inputs, config)

        assert len(pc.prompts) == 2
        assert pc.prompts[0] == "Analyze: text1"
        assert pc.prompts[1] == "Analyze: text2"

    def test_criteria_aliasing(self):
        """Test that 'criteria' is aliased to 'criterion'."""
        inputs = {
            "documents": ["doc1"],
            "criteria": ["quality"],  # Plural
        }
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",
            prompt_template="{document} | {criterion}",  # Singular
            parser="text",
        )

        pc = format_prompts(inputs, config)

        assert len(pc.prompts) == 1
        assert pc.prompts[0] == "doc1 | quality"

    def test_packed_prompts_include_prefill(self):
        """Test that packed prompts include prefill."""
        inputs = {"text": ["hello"]}
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",
            prompt_template="Translate: {text}",
            force_prefill_template="Translation:",
            parser="text",
        )

        pc = format_prompts(inputs, config)

        # Packed prompt should include prefill
        assert "Translation:" in pc.packed_prompts[0]
        # Regular prompt should not
        assert "Translation:" not in pc.prompts[0]

    def test_embed_instructions(self):
        """Test embedding instructions are formatted."""
        inputs = {"query": ["search term"]}
        config = InferenceConfig(
            provider="openai",
            model_name="text-embedding-3-large",
            prompt_template="{query}",
            embed_query_instr_template="Represent this query: ",
            is_embedding=True,
            parser="vector",
        )

        pc = format_prompts(inputs, config)

        assert pc.embed_query_instr is not None
        assert len(pc.embed_query_instr) == 1
        assert pc.embed_query_instr[0] == "Represent this query: "

        # Packed prompt should include instruction
        assert "Represent this query:" in pc.packed_prompts[0]


class TestPromptDeduplicationIntegration:
    """Test deduplication in realistic scenarios."""

    def test_duplicate_documents_deduplicated(self):
        """Test that identical documents are deduplicated."""
        inputs = {
            "documents": ["same text", "different text", "same text"],
            "criterion": ["quality"],
        }
        config = InferenceConfig(
            provider="gemini",
            model_name="gemini-2.5-flash-lite",
            prompt_template="Doc: {document}\nCrit: {criterion}",
            parser="text",
        )

        pc = format_prompts(inputs, config)
        remap_idxs, deduped = pc.dedup()

        # Should deduplicate to 2 unique prompts
        assert len(deduped.packed_prompts) == 2
        # Original length preserved in remap
        assert len(remap_idxs) == 3
        # First and last should map to same deduped index
        assert remap_idxs[0] == remap_idxs[2]
