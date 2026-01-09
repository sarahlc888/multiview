"""Tests for output parsers.

Tests cover:
- JSON parser wrapping behavior
- JSON parser with annotation_key
- Text parser
- Parser error handling
"""

import pytest

from multiview.inference.parsers import (
    get_parser,
    json_parser,
    text_parser,
    vector_parser,
)


class TestJSONParser:
    """Test JSON parser."""

    def test_json_parser_wraps_dict_in_list(self):
        """Test that JSON parser wraps dict in list when no annotation_key."""
        completion = '{"key": "value", "number": 42}'

        result = json_parser(completion)

        # Should wrap in list for consistency
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {"key": "value", "number": 42}

    def test_json_parser_preserves_list(self):
        """Test that JSON parser preserves lists."""
        completion = '[{"a": 1}, {"a": 2}]'

        result = json_parser(completion)

        # Should not double-wrap
        assert isinstance(result, list)
        assert len(result) == 2
        assert result == [{"a": 1}, {"a": 2}]

    def test_json_parser_with_annotation_key_dict(self):
        """Test extracting key from dict."""
        completion = '{"score": 5, "reason": "good"}'

        result = json_parser(completion, annotation_key="score")

        # Should extract value, not wrap in list
        assert result == 5

    def test_json_parser_with_annotation_key_list(self):
        """Test extracting key from list of dicts."""
        completion = '[{"score": 5}, {"score": 3}]'

        result = json_parser(completion, annotation_key="score")

        # Should extract list of values
        assert result == [5, 3]

    def test_json_parser_with_markdown_wrapper(self):
        """Test parsing JSON wrapped in markdown code block."""
        completion = """```json
{
  "categories": [
    {"name": "cat1", "description": "desc1"}
  ]
}
```"""

        result = json_parser(completion)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "categories" in result[0]
        assert len(result[0]["categories"]) == 1

    def test_json_parser_error_on_empty(self):
        """Test that empty completion raises error."""
        with pytest.raises(ValueError, match="empty"):
            json_parser("")

    def test_json_parser_error_on_invalid_json(self):
        """Test that invalid JSON raises error."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            json_parser("not valid json {{{")


class TestTextParser:
    """Test text parser."""

    def test_text_parser_with_string(self):
        """Test text parser with string input."""
        result = text_parser("Hello world")
        assert result == "Hello world"

    def test_text_parser_with_dict(self):
        """Test text parser with dict input (text key)."""
        result = text_parser({"text": "Hello world"})
        assert result == "Hello world"

    def test_text_parser_with_dict_content_key(self):
        """Test text parser with dict input (content key)."""
        result = text_parser({"content": "Hello world"})
        assert result == "Hello world"

    def test_text_parser_error_on_dict_missing_keys(self):
        """Test that dict without text/content raises error."""
        with pytest.raises(ValueError, match="No text/content"):
            text_parser({"other_key": "value"})


class TestVectorParser:
    """Test vector parser."""

    def test_vector_parser_with_dict(self):
        """Test vector parser extracts vector from dict."""
        result = vector_parser({"vector": [0.1, 0.2, 0.3]})
        assert result == [0.1, 0.2, 0.3]

    def test_vector_parser_with_list(self):
        """Test vector parser returns list as-is."""
        result = vector_parser([0.1, 0.2, 0.3])
        assert result == [0.1, 0.2, 0.3]

    def test_vector_parser_error_on_missing_vector(self):
        """Test that dict without vector key raises error."""
        with pytest.raises(ValueError, match="No vector found"):
            vector_parser({"text": "not a vector"})


class TestParserRegistry:
    """Test parser registry."""

    def test_get_parser_json(self):
        """Test getting JSON parser."""
        parser = get_parser("json")
        assert parser == json_parser

    def test_get_parser_text(self):
        """Test getting text parser."""
        parser = get_parser("text")
        assert parser == text_parser

    def test_get_parser_vector(self):
        """Test getting vector parser."""
        parser = get_parser("vector")
        assert parser == vector_parser

    def test_get_parser_invalid(self):
        """Test that invalid parser name raises error."""
        with pytest.raises(ValueError, match="Unknown parser"):
            get_parser("nonexistent")


class TestJSONParserEdgeCases:
    """Test edge cases for JSON parser to prevent regressions."""

    def test_schema_generation_response(self):
        """Test parsing response from category schema generation.

        This was the actual bug - the response is a dict but gets wrapped in a list.
        """
        completion = """{
  "reasoning": "I chose these categories because...",
  "categories": [
    {"name": "category1", "description": "desc1"},
    {"name": "category2", "description": "desc2"}
  ]
}"""

        result = json_parser(completion)

        # Should be wrapped in list
        assert isinstance(result, list)
        assert len(result) == 1

        # Unwrap to get actual schema
        schema = result[0]
        assert isinstance(schema, dict)
        assert "categories" in schema
        assert "reasoning" in schema
        assert len(schema["categories"]) == 2

    def test_classification_response_with_annotation_key(self):
        """Test parsing classification response with annotation_key."""
        completion = '{"category": "category1", "reasoning": "because..."}'

        result = json_parser(completion, annotation_key="category")

        # With annotation_key, should extract value directly
        assert result == "category1"

    def test_batch_classification_response(self):
        """Test parsing batch classification response."""
        completion = """[
  {"category": "cat1", "reasoning": "..."},
  {"category": "cat2", "reasoning": "..."}
]"""

        result = json_parser(completion)

        # List should not be wrapped
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["category"] == "cat1"
        assert result[1]["category"] == "cat2"

    def test_json_with_nested_backticks(self):
        """Test parsing JSON that contains backticks within string values.

        Regression test for issue where JSON strings containing triple backticks
        (like code blocks in markdown) would cause parsing to fail because the
        regex would match the first ``` instead of the last.
        """
        # Simulate a real LLM response with code blocks in the JSON string values
        completion = '```json\n{\n  "reasoning": "Consider code like `calculate()` and blocks.",\n  "summary": "Step 1: Do this.\\nStep 2: Example output:\\n```\\n[\\"a\\", \\"b\\"]\\n```\\nDone."\n}\n```'

        result = json_parser(completion)

        assert isinstance(result, list)
        assert len(result) == 1

        data = result[0]
        assert "reasoning" in data
        assert "summary" in data
        # Verify the content with backticks is preserved correctly
        assert "`calculate()`" in data["reasoning"]
        # The summary should contain the code block markers
        assert "```" in data["summary"]
