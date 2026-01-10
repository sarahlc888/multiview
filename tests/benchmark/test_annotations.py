"""Tests for schema generation and annotation functions.

Most tests are skipped by default as they require API keys and make real API calls.
Tests cover: category schemas, tag schemas, summary guidance, and full lm_all pipeline.

Run with API: export GEMINI_API_KEY=... && pytest tests/benchmark/test_annotations.py -v
Run locally: pytest tests/benchmark/test_annotations.py -v -k "Helper"
"""

import pytest

from multiview.benchmark.annotations import (
    annotate_with_lm_all,
    classify_documents_batch,
    generate_category_schema,
    generate_spurious_tag_schema,
    generate_summary_guidance,
    generate_summaries_batch,
    generate_tag_schema,
    apply_tags_batch,
)

pytestmark = pytest.mark.dev

# Sample test documents
SAMPLE_DOCS = [
    "There are 5 apples. If I buy 3 more, how many do I have?",
    "What is 12 divided by 4?",
    "John has 10 dollars. He spends 7 dollars. How much is left?",
]


@pytest.fixture
def mock_category_schema():
    """Mock category schema for testing."""
    return {
        "categories": [
            {"name": "addition", "description": "Problems involving addition"},
            {"name": "subtraction", "description": "Problems involving subtraction"},
            {"name": "multiplication", "description": "Problems involving multiplication"},
            {"name": "division", "description": "Problems involving division"},
        ]
    }


@pytest.fixture
def mock_tag_schema():
    """Mock tag schema for testing."""
    return {
        "tags": [
            {"name": "small_numbers", "description": "Uses numbers less than 20"},
            {"name": "money", "description": "Involves monetary amounts"},
            {"name": "multiple_operations", "description": "Uses more than one operation"},
        ]
    }


@pytest.fixture
def mock_summary_guidance():
    """Mock summary guidance for testing."""
    return {
        "summary_guidance": (
            "Provide a structured summary with two parts:\n"
            "1. Annotation trace: Explain what operations are used and why\n"
            "2. Final summary: List the operations in order (e.g., 'addition then division')"
        )
    }


class TestSchemaGeneration:
    """Tests for schema generation functions."""

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_generate_category_schema(self):
        """Test category schema generation with real API."""
        schema = generate_category_schema(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            n_samples=3,
            schema_hint="Focus on basic operations: addition, subtraction, multiplication, division",
        )

        # Verify schema structure
        assert "categories" in schema
        assert isinstance(schema["categories"], list)
        assert len(schema["categories"]) > 0

        # Verify each category has required fields
        for category in schema["categories"]:
            assert "name" in category
            assert "description" in category
            assert isinstance(category["name"], str)
            assert isinstance(category["description"], str)

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_generate_tag_schema(self):
        """Test tag schema generation with real API."""
        schema = generate_tag_schema(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            n_samples=3,
            is_spurious=False,
        )

        # Verify schema structure
        assert "tags" in schema
        assert isinstance(schema["tags"], list)
        assert len(schema["tags"]) > 0

        # Verify each tag has required fields
        for tag in schema["tags"]:
            assert "name" in tag
            assert "description" in tag

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_generate_spurious_tag_schema(self):
        """Test spurious tag schema generation with real API."""
        schema = generate_spurious_tag_schema(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            n_samples=3,
        )

        # Verify schema structure
        assert "tags" in schema
        assert isinstance(schema["tags"], list)

        # Spurious tags should capture surface properties
        # (we can't verify content without knowing LM response)

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_generate_summary_guidance(self):
        """Test summary guidance generation with real API."""
        guidance = generate_summary_guidance(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            n_samples=3,
            guidance_hint="Focus on listing operations in order",
        )

        # Verify guidance structure
        assert "summary_guidance" in guidance
        assert isinstance(guidance["summary_guidance"], str)
        assert len(guidance["summary_guidance"]) > 0


class TestAnnotationApplication:
    """Tests for annotation application functions."""

    def test_classify_documents_batch_structure(self, mock_category_schema):
        """Test that classify_documents_batch returns correct structure (mocked)."""
        # This test would need mocking of run_inference
        # For now, just verify the schema fixture is valid
        assert "categories" in mock_category_schema
        assert len(mock_category_schema["categories"]) > 0

    def test_apply_tags_batch_structure(self, mock_tag_schema):
        """Test that apply_tags_batch returns correct structure (mocked)."""
        # This test would need mocking of run_inference
        # For now, just verify the schema fixture is valid
        assert "tags" in mock_tag_schema
        assert len(mock_tag_schema["tags"]) > 0

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_classify_documents_batch_real(self, mock_category_schema):
        """Test document classification with real API."""
        annotations = classify_documents_batch(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            category_schema=mock_category_schema,
        )

        # Verify structure
        assert len(annotations) == len(SAMPLE_DOCS)

        for ann in annotations:
            assert "category" in ann
            assert ann["category"] is not None
            # Category should be one of the schema categories
            category_names = [c["name"] for c in mock_category_schema["categories"]]
            assert ann["category"] in category_names

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_apply_tags_batch_real(self, mock_tag_schema):
        """Test tag application with real API."""
        annotations = apply_tags_batch(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            tag_schema=mock_tag_schema,
        )

        # Verify structure
        assert len(annotations) == len(SAMPLE_DOCS)

        for ann in annotations:
            assert "tags" in ann
            assert isinstance(ann["tags"], dict)

            # All tags from schema should be present
            tag_names = [t["name"] for t in mock_tag_schema["tags"]]
            for tag_name in tag_names:
                assert tag_name in ann["tags"]
                assert isinstance(ann["tags"][tag_name], bool)

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_generate_summaries_batch_real(self, mock_summary_guidance):
        """Test summary generation with real API."""
        annotations = generate_summaries_batch(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            summary_guidance=mock_summary_guidance,
        )

        # Verify structure
        assert len(annotations) == len(SAMPLE_DOCS)

        for ann in annotations:
            assert "summary" in ann
            assert isinstance(ann["summary"], dict)
            assert "annotation_trace" in ann["summary"]
            assert "final_summary" in ann["summary"]


class TestAllAnnotation:
    """Tests for the complete 'all' annotation pipeline (lm_all)."""

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_annotate_with_lm_all_real(self):
        """Test complete 'all' annotation with real API."""
        annotations = annotate_with_lm_all(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used in the problem",
            n_schema_samples=3,
            category_schema_hint="Focus on addition, subtraction, multiplication, division",
            tag_schema_hint="Include tags for number size, context, complexity",
        )

        # Verify we got annotations for all documents
        assert len(annotations) == len(SAMPLE_DOCS)

        # Verify structure of each annotation
        for ann in annotations:
            # Check all required fields present
            assert "criterion_value" in ann  # Backward compatibility
            assert "category" in ann
            assert "tags" in ann
            assert "spurious_tags" in ann
            assert "summary" in ann

            # Check schemas are included (for reproducibility)
            assert "category_schema" in ann
            assert "tag_schema" in ann
            assert "spurious_tag_schema" in ann
            assert "summary_guidance" in ann

            # Check types
            assert isinstance(ann["category"], (str, type(None)))
            assert isinstance(ann["tags"], dict)
            assert isinstance(ann["spurious_tags"], dict)
            assert isinstance(ann["summary"], dict)

            # Check summary structure
            assert "annotation_trace" in ann["summary"]
            assert "final_summary" in ann["summary"]

            # Check that tags are boolean
            for tag_value in ann["tags"].values():
                assert isinstance(tag_value, bool)
            for tag_value in ann["spurious_tags"].values():
                assert isinstance(tag_value, bool)

    def test_annotation_backward_compatibility(self):
        """Test that annotations maintain backward compatibility."""
        # Mock annotation result
        mock_annotation = {
            "criterion_value": None,
            "category": "addition",
            "tags": {"small_numbers": True, "money": False},
            "spurious_tags": {"short_text": True},
            "summary": {
                "annotation_trace": "This problem uses addition...",
                "final_summary": "Addition: 5 + 3",
            },
            "category_schema": {},
            "tag_schema": {},
            "spurious_tag_schema": {},
            "summary_guidance": {},
        }

        # Verify backward-compatible field exists
        assert "criterion_value" in mock_annotation

        # Verify new fields exist
        assert "category" in mock_annotation
        assert "tags" in mock_annotation
        assert "spurious_tags" in mock_annotation
        assert "summary" in mock_annotation


class TestAnnotationEdgeCases:
    """Tests for edge cases in annotation system."""

    def test_empty_documents_list(self):
        """Test handling of empty document list."""
        # This should not crash
        # (Would need mocking to test without API calls)
        pass

    def test_single_document(self):
        """Test annotation with single document."""
        # Should work with n_schema_samples=1
        pass

    def test_very_long_document(self):
        """Test annotation with very long document."""
        # Should handle token limits gracefully
        pass

    def test_special_characters_in_document(self):
        """Test annotation with special characters."""
        special_docs = [
            "Problem with $10.50 and €20.30",
            "Use the formula: x = (a + b) / 2",
            "Text with emoji: 🎯 Target is 100%",
        ]
        # Should handle without crashing
        pass


class TestAnnotationHelpers:
    """Tests for annotation helper utilities."""

    def test_extract_active_tags(self):
        """Test tag extraction helper."""
        from multiview.benchmark.triplets.utils import extract_active_tags

        annotation = {
            "tags": {
                "tag1": True,
                "tag2": False,
                "tag3": True,
                "tag4": False,
            }
        }

        active = extract_active_tags(annotation, "tags")
        assert active == {"tag1", "tag3"}

    def test_extract_active_tags_spurious(self):
        """Test spurious tag extraction."""
        from multiview.benchmark.triplets.utils import extract_active_tags

        annotation = {
            "spurious_tags": {
                "spur1": True,
                "spur2": False,
                "spur3": True,
            }
        }

        active = extract_active_tags(annotation, "spurious_tags")
        assert active == {"spur1", "spur3"}

    def test_extract_active_tags_empty(self):
        """Test tag extraction with no tags."""
        from multiview.benchmark.triplets.utils import extract_active_tags

        annotation = {}
        active = extract_active_tags(annotation, "tags")
        assert active == set()

    def test_format_annotation_for_display(self):
        """Test annotation formatting for LM judge."""
        from multiview.benchmark.triplets.utils import format_annotation_for_display

        annotation = {
            "category": "addition",
            "tags": {"small_numbers": True, "money": False, "multiple_ops": True},
            "summary": {
                "annotation_trace": "Reasoning...",
                "final_summary": "Addition: 5 + 3",
            },
        }

        formatted = format_annotation_for_display(annotation)

        # Should include category
        assert "Category: addition" in formatted

        # Should include active tags only
        assert "small_numbers" in formatted
        assert "multiple_ops" in formatted
        assert "money" not in formatted  # False tag should not appear

        # Should include summary
        assert "Addition: 5 + 3" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
