"""Tests for category schema generation and classification.

Tests cover:
- Schema unwrapping from JSON parser
- Category schema structure validation
- Classification logic
"""

import pytest

from multiview.benchmark.annotations.class_schema import (
    classify_documents_batch,
    generate_category_schema,
)

pytestmark = pytest.mark.dev
class TestSchemaUnwrapping:
    """Test that schema results are properly unwrapped."""

    def test_schema_unwrapping_mock(self, monkeypatch):
        """Test that wrapped schema from JSON parser is unwrapped correctly."""
        # Mock run_inference to return wrapped schema (as JSON parser does)
        def mock_run_inference(*args, **kwargs):
            # JSON parser wraps dict in list
            return [
                [
                    {
                        "reasoning": "Test reasoning",
                        "categories": [
                            {"name": "cat1", "description": "desc1"},
                            {"name": "cat2", "description": "desc2"},
                        ],
                    }
                ]
            ]

        monkeypatch.setattr(
            "multiview.benchmark.annotations.class_schema.run_inference",
            mock_run_inference,
        )

        # This should not raise an error
        schema = generate_category_schema(
            documents=["doc1", "doc2"],
            criterion="test_criterion",
            criterion_description="test description",
            n_samples=2,
        )

        # Should be unwrapped to a dict
        assert isinstance(schema, dict)
        assert "categories" in schema
        assert "reasoning" in schema
        assert len(schema["categories"]) == 2

    def test_schema_validation_rejects_list(self, monkeypatch):
        """Test that schema validation rejects lists."""

        def mock_run_inference(*args, **kwargs):
            # Return a list that can't be unwrapped (wrong format)
            return [["cat1", "cat2"]]

        monkeypatch.setattr(
            "multiview.benchmark.annotations.class_schema.run_inference",
            mock_run_inference,
        )

        # Should raise ValueError because schema is not a dict
        with pytest.raises(ValueError, match="Expected schema to be a dict"):
            generate_category_schema(
                documents=["doc1", "doc2"],
                criterion="test_criterion",
                criterion_description="test description",
                n_samples=2,
            )

    def test_schema_validation_rejects_none(self, monkeypatch):
        """Test that None schema raises error."""

        def mock_run_inference(*args, **kwargs):
            return [None]

        monkeypatch.setattr(
            "multiview.benchmark.annotations.class_schema.run_inference",
            mock_run_inference,
        )

        with pytest.raises(ValueError, match="Failed to generate category schema"):
            generate_category_schema(
                documents=["doc1"],
                criterion="test",
                criterion_description="test",
            )


class TestCategoryClassification:
    """Test category classification logic."""

    def test_batch_classification_result_structure(self, monkeypatch):
        """Test that batch classification returns correct structure."""

        def mock_run_inference(*args, **kwargs):
            # Simulate classification results
            return [
                {"category": "cat1", "reasoning": "reason1"},
                {"category": "cat2", "reasoning": "reason2"},
            ]

        monkeypatch.setattr(
            "multiview.benchmark.annotations.class_schema.run_inference",
            mock_run_inference,
        )

        schema = {
            "categories": [
                {"name": "cat1", "description": "desc1"},
                {"name": "cat2", "description": "desc2"},
            ]
        }

        results = classify_documents_batch(
            documents=["doc1", "doc2"],
            criterion="test",
            criterion_description="test desc",
            category_schema=schema,
        )

        assert len(results) == 2
        assert all("category" in r for r in results)
        assert all("category_reasoning" in r for r in results)
        assert results[0]["category"] == "cat1"
        assert results[1]["category"] == "cat2"

    def test_classification_matches_category_names(self, monkeypatch):
        """Test that classification matches to valid category names."""

        def mock_run_inference(*args, **kwargs):
            # Model returns slightly different case
            return [
                {"category": "Category_One", "reasoning": "reason"},
            ]

        monkeypatch.setattr(
            "multiview.benchmark.annotations.class_schema.run_inference",
            mock_run_inference,
        )

        schema = {
            "categories": [
                {"name": "category_one", "description": "desc1"},
                {"name": "category_two", "description": "desc2"},
            ]
        }

        results = classify_documents_batch(
            documents=["doc1"],
            criterion="test",
            criterion_description="test desc",
            category_schema=schema,
        )

        # Should match despite case difference
        assert results[0]["category"] == "category_one"

    def test_classification_fallback_on_no_match(self, monkeypatch):
        """Test fallback to first category when no match."""

        def mock_run_inference(*args, **kwargs):
            return [
                {"category": "invalid_category", "reasoning": "reason"},
            ]

        monkeypatch.setattr(
            "multiview.benchmark.annotations.class_schema.run_inference",
            mock_run_inference,
        )

        schema = {
            "categories": [
                {"name": "cat1", "description": "desc1"},
                {"name": "cat2", "description": "desc2"},
            ]
        }

        results = classify_documents_batch(
            documents=["doc1"],
            criterion="test",
            criterion_description="test desc",
            category_schema=schema,
        )

        # Should fall back to first category
        assert results[0]["category"] == "cat1"

    def test_classification_handles_none_result(self, monkeypatch):
        """Test handling of None classification result."""

        def mock_run_inference(*args, **kwargs):
            return [None]

        monkeypatch.setattr(
            "multiview.benchmark.annotations.class_schema.run_inference",
            mock_run_inference,
        )

        schema = {
            "categories": [
                {"name": "cat1", "description": "desc1"},
            ]
        }

        results = classify_documents_batch(
            documents=["doc1"],
            criterion="test",
            criterion_description="test desc",
            category_schema=schema,
        )

        assert results[0]["category"] is None
        assert results[0]["category_reasoning"] is None
