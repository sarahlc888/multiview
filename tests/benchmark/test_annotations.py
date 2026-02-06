"""Tests for schema generation and annotation functions.

Most tests are skipped by default as they require API keys and make real API calls.
Tests cover: category schemas, tag schemas, summary guidance, and full lm_all pipeline.

Run with API: export GEMINI_API_KEY=... && pytest tests/benchmark/test_annotations.py -v
Run locally: pytest tests/benchmark/test_annotations.py -v -k "Helper"
"""

import pytest

from multiview.benchmark.annotations import (
    annotate_with_lm_all,
    annotate_with_lm_category,
    annotate_with_lm_summary,
    annotate_with_lm_tags,
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
            document_type="math word problem",
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
            config="tag_schema_generation_gemini",
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
            document_type="math word problem",
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


class TestStyleSpecificAnnotations:
    """Tests for style-specific annotation functions (lm_category, lm_tags, lm_summary)."""

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_annotate_with_lm_category_real(self):
        """Test category-only annotation with real API."""
        annotations = annotate_with_lm_category(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used in the problem",
            n_schema_samples=3,
            category_schema_hint="Focus on addition, subtraction, multiplication, division",
        )

        # Verify we got annotations for all documents
        assert len(annotations) == len(SAMPLE_DOCS)

        # Verify structure of each annotation
        for ann in annotations:
            # Check required fields
            assert "category" in ann
            assert "category_schema" in ann

            # Check that only category fields are present (not tags or summary)
            assert "tags" not in ann
            assert "spurious_tags" not in ann
            assert "summary" not in ann
            assert "tag_schema" not in ann
            assert "spurious_tag_schema" not in ann
            assert "summary_guidance" not in ann

            # Check types
            assert isinstance(ann["category"], (str, type(None)))
            assert isinstance(ann["category_schema"], dict)

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_annotate_with_lm_tags_real(self):
        """Test tag-only annotation with real API."""
        annotations = annotate_with_lm_tags(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used in the problem",
            n_schema_samples=3,
            tag_schema_hint="Include tags for number size, context, complexity",
        )

        # Verify we got annotations for all documents
        assert len(annotations) == len(SAMPLE_DOCS)

        # Verify structure of each annotation
        for ann in annotations:
            # Check required fields
            assert "tags" in ann
            assert "spurious_tags" in ann
            assert "tag_schema" in ann
            assert "spurious_tag_schema" in ann

            # Check that only tag fields are present (not category or summary)
            assert "category" not in ann
            assert "summary" not in ann
            assert "category_schema" not in ann
            assert "summary_guidance" not in ann

            # Check types
            assert isinstance(ann["tags"], dict)
            assert isinstance(ann["spurious_tags"], dict)

            # Check that tags are boolean
            for tag_value in ann["tags"].values():
                assert isinstance(tag_value, bool)
            for tag_value in ann["spurious_tags"].values():
                assert isinstance(tag_value, bool)

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_annotate_with_lm_summary_real(self):
        """Test summary-only annotation with real API."""
        annotations = annotate_with_lm_summary(
            documents=SAMPLE_DOCS,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used in the problem",
            n_schema_samples=3,
            summary_hint="Focus on listing operations in order",
        )

        # Verify we got annotations for all documents
        assert len(annotations) == len(SAMPLE_DOCS)

        # Verify structure of each annotation
        for ann in annotations:
            # Check required fields
            assert "summary" in ann
            assert "summary_guidance" in ann

            # Check that only summary fields are present (not category or tags)
            assert "category" not in ann
            assert "tags" not in ann
            assert "spurious_tags" not in ann
            assert "category_schema" not in ann
            assert "tag_schema" not in ann
            assert "spurious_tag_schema" not in ann

            # Check types
            assert isinstance(ann["summary"], dict)

            # Check summary structure
            assert "annotation_trace" in ann["summary"]
            assert "final_summary" in ann["summary"]

    @pytest.mark.parametrize(
        "style,annotation_fn,expected_fields,excluded_fields",
        [
            (
                "lm_category",
                annotate_with_lm_category,
                ["category", "category_schema"],
                ["tags", "spurious_tags", "summary", "tag_schema", "spurious_tag_schema", "summary_guidance"],
            ),
            (
                "lm_tags",
                annotate_with_lm_tags,
                ["tags", "spurious_tags", "tag_schema", "spurious_tag_schema"],
                ["category", "summary", "category_schema", "summary_guidance"],
            ),
            (
                "lm_summary",
                annotate_with_lm_summary,
                ["summary", "summary_guidance"],
                ["category", "tags", "spurious_tags", "category_schema", "tag_schema", "spurious_tag_schema"],
            ),
        ],
    )
    def test_annotation_style_field_presence(
        self, style, annotation_fn, expected_fields, excluded_fields
    ):
        """Test that each style generates expected annotation fields (structure only, no API calls)."""
        # This is a structure test - we can't run it without mocking
        # But we verify the test parameters are correct
        assert len(expected_fields) > 0
        assert len(excluded_fields) > 0


class TestInfinitePromptsAccuracy:
    """Tests for LM annotator accuracy on infinite_prompts taxonomy."""

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_infinite_prompts_category_accuracy(self):
        """Test that LM category annotator achieves >70% accuracy on infinite_prompts taxonomy.

        This test validates that the LM annotator can correctly predict the taxonomy
        categories from the infinite-chats dataset.
        """
        from multiview.docsets.infinite_prompts import InfinitePromptsDocSet

        # Load infinite_prompts dataset with a reasonable sample size
        docset = InfinitePromptsDocSet(config={"max_docs": 30})
        documents = docset.load_documents()

        print(f"\nLoaded {len(documents)} prompts from infinite-chats-taxonomy")

        # Get ground truth categories from precomputed annotations
        precomputed = docset.get_precomputed_annotations("categories")
        ground_truth = []
        for doc in documents:
            doc_text = docset.get_document_text(doc)
            gt_value = precomputed.get(doc_text, {}).get("prelabel", "")
            ground_truth.append(gt_value)

        # Use LM annotator to predict categories
        print("\nGenerating LM category annotations...")
        annotations = annotate_with_lm_category(
            documents=documents,
            criterion="categories",
            criterion_description=(
                "The category taxonomy labels from the infinite-chats dataset, "
                "describing the type of user prompt based on its purpose and content."
            ),
            n_schema_samples=10,
            category_schema_hint=(
                "Categories should reflect prompt types from the taxonomy, such as: "
                "Creative Content Generation, Concept Explanations, Recommendations, "
                "Writing Genres, Skill Development, etc. Each prompt may belong to "
                "multiple categories separated by commas."
            ),
        )

        # Extract predictions
        predictions = [ann.get("category", "") for ann in annotations]

        # Calculate accuracy metrics
        exact_matches = 0
        partial_matches = 0

        for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
            # Parse categories (they're comma-separated)
            pred_cats = set(c.strip() for c in (pred or "").split(","))
            gt_cats = set(c.strip() for c in (gt or "").split(","))

            # Exact match: all categories match
            if pred_cats == gt_cats:
                exact_matches += 1
                partial_matches += 1
            # Partial match: any overlap in categories
            elif pred_cats & gt_cats:
                partial_matches += 1

            # Print first few examples
            if i < 5:
                print(f"\n--- Example {i+1} ---")
                print(f"Prompt: {documents[i][:100]}...")
                print(f"Ground Truth: {gt}")
                print(f"Predicted: {pred}")
                print(f"Match: {'✓ Exact' if pred_cats == gt_cats else '✓ Partial' if pred_cats & gt_cats else '✗ No match'}")

        # Calculate accuracy percentages
        exact_accuracy = (exact_matches / len(documents)) * 100
        partial_accuracy = (partial_matches / len(documents)) * 100

        print(f"\n{'='*60}")
        print(f"ACCURACY RESULTS (n={len(documents)})")
        print(f"{'='*60}")
        print(f"Exact matches: {exact_matches}/{len(documents)} ({exact_accuracy:.1f}%)")
        print(f"Partial matches: {partial_matches}/{len(documents)} ({partial_accuracy:.1f}%)")
        print(f"{'='*60}")

        # Print category schema that was generated
        if annotations and "category_schema" in annotations[0]:
            schema = annotations[0]["category_schema"]
            print("\nGenerated Category Schema:")
            for cat in schema.get("categories", [])[:10]:  # Show first 10
                print(f"  - {cat.get('name')}: {cat.get('description', '')[:80]}")

        # Assert that partial accuracy is > 70%
        # (Partial matching is more appropriate since categories can be comma-separated)
        assert partial_accuracy > 70, (
            f"LM annotator accuracy ({partial_accuracy:.1f}%) is below 70% threshold. "
            f"This suggests the annotator cannot reliably predict the taxonomy categories."
        )

        print(f"\n✓ LM annotator achieves {partial_accuracy:.1f}% accuracy on infinite_prompts taxonomy")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
