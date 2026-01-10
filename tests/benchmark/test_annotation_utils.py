"""Tests for annotation utility functions.

These tests run locally without API keys - they test helper functions like
tag extraction, Jaccard similarity, and annotation formatting.

Run: pytest tests/benchmark/test_annotation_utils.py -v
"""

import pytest

from multiview.benchmark.triplets.utils import (
    extract_active_tags,
    format_annotation_for_display,
    jaccard_similarity,
)

pytestmark = pytest.mark.dev

class TestExtractActiveTags:
    """Tests for extract_active_tags helper."""

    def test_extract_tags_basic(self):
        """Test basic tag extraction."""
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

    def test_extract_spurious_tags(self):
        """Test spurious tag extraction."""
        annotation = {
            "spurious_tags": {
                "spur1": True,
                "spur2": False,
                "spur3": True,
            }
        }

        active = extract_active_tags(annotation, "spurious_tags")
        assert active == {"spur1", "spur3"}

    def test_extract_all_false_tags(self):
        """Test extraction when all tags are false."""
        annotation = {
            "tags": {
                "tag1": False,
                "tag2": False,
            }
        }

        active = extract_active_tags(annotation, "tags")
        assert active == set()

    def test_extract_all_true_tags(self):
        """Test extraction when all tags are true."""
        annotation = {
            "tags": {
                "tag1": True,
                "tag2": True,
                "tag3": True,
            }
        }

        active = extract_active_tags(annotation, "tags")
        assert active == {"tag1", "tag2", "tag3"}

    def test_extract_empty_tags(self):
        """Test extraction with no tags."""
        annotation = {"tags": {}}

        active = extract_active_tags(annotation, "tags")
        assert active == set()

    def test_extract_missing_key(self):
        """Test extraction when key doesn't exist."""
        annotation = {}

        active = extract_active_tags(annotation, "tags")
        assert active == set()

    def test_extract_wrong_key(self):
        """Test extraction with wrong key name."""
        annotation = {
            "tags": {"tag1": True}
        }

        # Asking for spurious_tags when only tags exist
        active = extract_active_tags(annotation, "spurious_tags")
        assert active == set()


class TestJaccardSimilarity:
    """Tests for jaccard_similarity helper."""

    def test_identical_sets(self):
        """Test Jaccard similarity of identical sets."""
        a = {"tag1", "tag2", "tag3"}
        b = {"tag1", "tag2", "tag3"}

        sim = jaccard_similarity(a, b)
        assert sim == 1.0

    def test_disjoint_sets(self):
        """Test Jaccard similarity of disjoint sets."""
        a = {"tag1", "tag2"}
        b = {"tag3", "tag4"}

        sim = jaccard_similarity(a, b)
        assert sim == 0.0

    def test_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        a = {"tag1", "tag2", "tag3"}
        b = {"tag2", "tag3", "tag4"}

        # Intersection: {tag2, tag3} = 2
        # Union: {tag1, tag2, tag3, tag4} = 4
        # Jaccard = 2/4 = 0.5
        sim = jaccard_similarity(a, b)
        assert sim == 0.5

    def test_subset(self):
        """Test Jaccard similarity when one is subset of other."""
        a = {"tag1", "tag2"}
        b = {"tag1", "tag2", "tag3", "tag4"}

        # Intersection: {tag1, tag2} = 2
        # Union: {tag1, tag2, tag3, tag4} = 4
        # Jaccard = 2/4 = 0.5
        sim = jaccard_similarity(a, b)
        assert sim == 0.5

    def test_both_empty(self):
        """Test Jaccard similarity of two empty sets."""
        a = set()
        b = set()

        sim = jaccard_similarity(a, b)
        assert sim == 0.0  # By convention

    def test_one_empty(self):
        """Test Jaccard similarity when one set is empty."""
        a = {"tag1", "tag2"}
        b = set()

        sim = jaccard_similarity(a, b)
        assert sim == 0.0

    def test_single_element_match(self):
        """Test Jaccard similarity with single matching element."""
        a = {"tag1"}
        b = {"tag1"}

        sim = jaccard_similarity(a, b)
        assert sim == 1.0

    def test_single_element_no_match(self):
        """Test Jaccard similarity with single non-matching element."""
        a = {"tag1"}
        b = {"tag2"}

        sim = jaccard_similarity(a, b)
        assert sim == 0.0


class TestFormatAnnotationForDisplay:
    """Tests for format_annotation_for_display helper."""

    def test_format_complete_annotation(self):
        """Test formatting with all fields present."""
        annotation = {
            "category": "addition",
            "tags": {
                "small_numbers": True,
                "money": False,
                "multiple_ops": True,
            },
            "summary": {
                "annotation_trace": "This problem involves...",
                "final_summary": "Addition: 5 + 3 = 8",
            },
        }

        formatted = format_annotation_for_display(annotation)

        # Should include category
        assert "Category: addition" in formatted

        # Should include active tags (sorted)
        assert "Tags:" in formatted
        assert "small_numbers" in formatted
        assert "multiple_ops" in formatted

        # Should NOT include inactive tags
        assert "money" not in formatted

        # Should include summary
        assert "Summary:" in formatted
        assert "Addition: 5 + 3 = 8" in formatted

    def test_format_with_no_category(self):
        """Test formatting without category."""
        annotation = {
            "tags": {"tag1": True},
            "summary": {"final_summary": "Summary text"},
        }

        formatted = format_annotation_for_display(annotation)

        assert "Category:" not in formatted
        assert "Tags:" in formatted
        assert "Summary:" in formatted

    def test_format_with_no_active_tags(self):
        """Test formatting with all tags false."""
        annotation = {
            "category": "addition",
            "tags": {"tag1": False, "tag2": False},
            "summary": {"final_summary": "Summary text"},
        }

        formatted = format_annotation_for_display(annotation)

        assert "Category: addition" in formatted
        assert "Tags:" not in formatted  # No active tags
        assert "Summary:" in formatted

    def test_format_with_no_summary(self):
        """Test formatting without summary."""
        annotation = {
            "category": "addition",
            "tags": {"tag1": True},
        }

        formatted = format_annotation_for_display(annotation)

        assert "Category: addition" in formatted
        assert "Tags:" in formatted
        assert "Summary:" not in formatted

    def test_format_empty_annotation(self):
        """Test formatting completely empty annotation."""
        annotation = {}

        formatted = format_annotation_for_display(annotation)

        assert formatted == "No annotation"

    def test_format_tags_are_sorted(self):
        """Test that tags are sorted in output."""
        annotation = {
            "tags": {
                "zebra": True,
                "apple": True,
                "monkey": True,
            }
        }

        formatted = format_annotation_for_display(annotation)

        # Tags should appear in alphabetical order
        lines = formatted.split("\n")
        tags_line = [line for line in lines if "Tags:" in line][0]

        # Extract tag names from the line
        # Should be: "Tags: apple, monkey, zebra"
        assert "apple, monkey, zebra" in tags_line

    def test_format_summary_as_string(self):
        """Test formatting when summary is string instead of dict."""
        annotation = {
            "category": "test",
            "summary": "Just a string summary",
        }

        formatted = format_annotation_for_display(annotation)

        assert "Category: test" in formatted
        assert "Summary: Just a string summary" in formatted

    def test_format_summary_dict_with_only_final(self):
        """Test formatting when summary dict only has final_summary."""
        annotation = {
            "summary": {
                "final_summary": "Final summary only",
            }
        }

        formatted = format_annotation_for_display(annotation)

        assert "Summary: Final summary only" in formatted


class TestAnnotationUtilsIntegration:
    """Integration tests combining multiple utilities."""

    def test_extract_and_jaccard(self):
        """Test using extract_active_tags with jaccard_similarity."""
        ann1 = {
            "tags": {"tag1": True, "tag2": True, "tag3": False}
        }
        ann2 = {
            "tags": {"tag2": True, "tag3": True, "tag4": False}
        }

        tags1 = extract_active_tags(ann1, "tags")
        tags2 = extract_active_tags(ann2, "tags")

        # tags1 = {tag1, tag2}
        # tags2 = {tag2, tag3}
        # Intersection = {tag2}
        # Union = {tag1, tag2, tag3}
        # Jaccard = 1/3 = 0.333...

        sim = jaccard_similarity(tags1, tags2)
        assert abs(sim - 1/3) < 0.001

    def test_extract_format_and_display(self):
        """Test full pipeline: extract, format, display."""
        annotation = {
            "category": "test_category",
            "tags": {
                "active1": True,
                "active2": True,
                "inactive": False,
            },
            "spurious_tags": {
                "spur1": True,
                "spur2": False,
            },
            "summary": {
                "annotation_trace": "Trace text",
                "final_summary": "Final text",
            },
        }

        # Extract active tags
        active = extract_active_tags(annotation, "tags")
        assert len(active) == 2

        # Extract spurious
        spurious = extract_active_tags(annotation, "spurious_tags")
        assert len(spurious) == 1

        # Format for display
        display = format_annotation_for_display(annotation)
        assert "test_category" in display
        assert "active1" in display
        assert "active2" in display
        assert "inactive" not in display


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
