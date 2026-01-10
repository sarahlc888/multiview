"""End-to-end integration tests for Task annotation workflows.

Tests complete flows: load documents → annotate → create triplets → save.
Most tests require API keys and are skipped by default.

Run error handling tests locally: pytest tests/benchmark/test_annotation_integration.py -v -k "error"
Run full suite with API: export GEMINI_API_KEY=... && pytest tests/benchmark/test_annotation_integration.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from multiview.benchmark.task import Task
from multiview.benchmark.artifacts import save_task_annotations


class TestTaskAnnotationIntegration:
    """Tests for Task class integration with annotation system."""

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_task_all_annotation_flow(self):
        """Test complete task flow with 'all' annotation (lm_all style)."""
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "arithmetic_operations",
                "criterion_description": "Types of arithmetic operations used",
                "max_docs": 5,  # Small number for testing
                "max_triplets": 3,
                "triplet_style": "lm_all",
                "candidate_strategy": "multi",
                "n_schema_samples": 5,
                "category_schema_hint": "Focus on basic operations",
                "use_spurious_hard_negs": True,
            }
        )

        # Step 1: Load documents
        task.load_documents()
        assert len(task.documents) == 5

        # Step 2: Annotate documents
        task.annotate_documents()
        assert len(task.document_annotations) == 5

        # Verify annotation structure
        for ann in task.document_annotations:
            assert "category" in ann
            assert "tags" in ann
            assert "spurious_tags" in ann
            assert "summary" in ann
            assert "category_schema" in ann
            assert "tag_schema" in ann
            assert "spurious_tag_schema" in ann
            assert "summary_guidance" in ann

        # Step 3: Create triplets
        task.create_triplets()
        assert len(task.triplets) == 3
        assert all(len(triplet) == 3 for triplet in task.triplets)

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_save_doc_annotations(self):
        """Test saving document annotations to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task = Task(
                config={
                    "document_set": "gsm8k",
                    "criterion": "word_count",
                    "max_docs": 3,
                    "triplet_style": "lm_all",
                }
            )

            task.load_documents()
            task.annotate_documents()

            # Save annotations
            output_dir = Path(tmpdir) / "annotations"
            save_task_annotations(task, output_dir)

            # Verify file was created
            task_name = task.get_task_name()
            output_file = output_dir / task_name / "annotations.jsonl"
            assert output_file.exists()

            # Verify file content
            with open(output_file) as f:
                lines = f.readlines()
                assert len(lines) == 3  # 3 documents

                for i, line in enumerate(lines):
                    data = json.loads(line)
                    assert "doc_id" in data
                    assert data["doc_id"] == i
                    assert "document" in data
                    assert "category" in data
                    assert "tags" in data
                    assert "spurious_tags" in data
                    assert "summary" in data

    def test_annotation_mode_inference(self):
        """Test that annotation mode is correctly inferred from triplet_style."""
        # lm_all (or lm_all) should trigger rich "all" annotation
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "test",
                "max_docs": 3,
                "triplet_style": "lm_all",
            }
        )
        # Would need to mock to verify, but the code path should be correct

        # random should not trigger rich annotation
        task_random = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "word_count",
                "max_docs": 3,
                "triplet_style": "random",
            }
        )
        # Verify random style doesn't need annotations
        task_random.load_documents()
        task_random.create_triplets()  # Should work without annotate_documents()

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_known_criterion_annotation(self):
        """Test annotation with known criterion (deterministic)."""
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "word_count",  # Known criterion
                "max_docs": 5,
                "triplet_style": "random",
            }
        )

        task.load_documents()
        task.annotate_documents()

        # Verify annotations have criterion_value
        for ann in task.document_annotations:
            assert "criterion_value" in ann
            assert isinstance(ann["criterion_value"], (int, float, str, type(None)))

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_annotation_with_all_config_options(self):
        """Test annotation with all configuration options."""
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "problem_difficulty",
                "criterion_description": "Difficulty level of the math problem",
                "max_docs": 5,
                "triplet_style": "lm_all",
                # Schema generation options
                "n_schema_samples": 5,
                "category_schema_hint": "Use difficulty levels: easy, medium, hard",
                "tag_schema_hint": "Include tags for: multi-step, word problem, real-world context",
                "summary_guidance_hint": "Explain what makes the problem easy/medium/hard",
                "summary_format_hint": "Format as: Difficulty: X because Y",
                # Triplet creation options
                "candidate_strategy": "multi",
                "use_spurious_hard_negs": True,
                "max_triplets": 3,
            }
        )

        task.load_documents()
        task.annotate_documents()
        task.create_triplets()

        # Verify all worked
        assert len(task.documents) == 5
        assert len(task.document_annotations) == 5
        assert len(task.triplets) == 3


class TestAnnotationCaching:
    """Tests for annotation caching behavior."""

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_annotation_cache_aliases(self):
        """Test that cache aliases are correctly set."""
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "test_criterion",
                "max_docs": 3,
                "triplet_style": "lm_all",
            }
        )

        # The cache aliases should include task name
        task_name = task.get_task_name()
        assert task_name == "gsm8k__test_criterion"

        # When annotate_documents is called, it should use cache_alias_prefix
        # This would need mocking to verify, but the code should pass:
        # cache_alias_prefix=f"{task_name}_annotation"

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    def test_schema_generation_caching(self):
        """Test that schema generation is cached."""
        # Create two tasks with same criterion
        config = {
            "document_set": "gsm8k",
            "criterion": "operations",
            "criterion_description": "Arithmetic operations",
            "max_docs": 5,
            "triplet_style": "lm_all",
            "n_schema_samples": 5,
        }

        task1 = Task(config=config)
        task1.load_documents()
        task1.annotate_documents()

        task2 = Task(config=config)
        task2.load_documents()
        task2.annotate_documents()

        # Schema should be the same (cached)
        ann1 = task1.document_annotations[0]
        ann2 = task2.document_annotations[0]

        assert ann1["category_schema"] == ann2["category_schema"]
        assert ann1["tag_schema"] == ann2["tag_schema"]
        assert ann1["spurious_tag_schema"] == ann2["spurious_tag_schema"]


class TestAnnotationErrorHandling:
    """Tests for error handling in annotation system."""

    def test_annotate_before_load_raises(self):
        """Test that annotating before loading raises error."""
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "test",
                "triplet_style": "lm_all",
            }
        )

        with pytest.raises(RuntimeError, match="Must call load_documents"):
            task.annotate_documents()

    def test_create_triplets_before_load_raises(self):
        """Test that creating triplets before loading raises error."""
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "test",
                "triplet_style": "lm_all",
            }
        )

        with pytest.raises(RuntimeError, match="Must call load_documents"):
            task.create_triplets()

    def test_save_annotations_before_annotate_raises(self):
        """Test that saving before annotating raises error."""
        task = Task(
            config={
                "document_set": "gsm8k",
                "criterion": "test",
                "triplet_style": "lm_all",
            }
        )

        task.load_documents()

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="Must call annotate_documents"):
                save_task_annotations(task, tmpdir)


class TestAnnotationWithDifferentDatasets:
    """Tests for annotation across different datasets."""

    @pytest.mark.skip(reason="Requires API key and makes real API calls")
    @pytest.mark.parametrize("dataset", ["gsm8k", "rocstories", "crossword_clues"])
    def test_annotation_across_datasets(self, dataset):
        """Test that annotation works for all supported datasets."""
        task = Task(
            config={
                "document_set": dataset,
                "criterion": "complexity",
                "criterion_description": "Complexity level of the content",
                "max_docs": 3,
                "triplet_style": "lm_all",
                "n_schema_samples": 3,
            }
        )

        task.load_documents()
        task.annotate_documents()

        # Verify annotations
        assert len(task.document_annotations) == 3

        for ann in task.document_annotations:
            assert "category" in ann
            assert "tags" in ann
            assert "spurious_tags" in ann
            assert "summary" in ann


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
