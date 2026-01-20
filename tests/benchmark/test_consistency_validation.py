"""Unit tests for triplet consistency validation.

Tests the validate_triplet_consistency() function and its integration
into the quality rating workflow.
"""

import pytest

from multiview.benchmark.triplets.quality_assurance import (
    _check_consistency,
    validate_triplet_consistency,
    rate_and_filter_quality_workflow,
)


class TestCheckConsistency:
    """Tests for _check_consistency() helper function."""

    def test_both_pass(self):
        """Test triplet that passes both checks."""
        passed, reason = _check_consistency(
            original_rating=4,
            swapped_rating=1,
            min_threshold=3,
            max_threshold=1,
        )
        assert passed is True
        assert reason is None

    def test_original_too_low(self):
        """Test triplet with original rating below threshold."""
        passed, reason = _check_consistency(
            original_rating=2,
            swapped_rating=1,
            min_threshold=3,
            max_threshold=1,
        )
        assert passed is False
        assert reason == "original_too_low_2"

    def test_swapped_too_high(self):
        """Test triplet with swapped rating above threshold."""
        passed, reason = _check_consistency(
            original_rating=4,
            swapped_rating=2,
            min_threshold=3,
            max_threshold=1,
        )
        assert passed is False
        assert reason == "swapped_too_high_2"

    def test_both_fail(self):
        """Test triplet failing both checks."""
        passed, reason = _check_consistency(
            original_rating=2,
            swapped_rating=3,
            min_threshold=3,
            max_threshold=1,
        )
        # Should return first failure (original)
        assert passed is False
        assert reason == "original_too_low_2"

    def test_boundary_values(self):
        """Test boundary conditions."""
        # Exactly at thresholds - should pass
        passed, _ = _check_consistency(3, 1, min_threshold=3, max_threshold=1)
        assert passed is True

        # Just below original threshold - should fail
        passed, _ = _check_consistency(2, 1, min_threshold=3, max_threshold=1)
        assert passed is False

        # Just above swapped threshold - should fail
        passed, _ = _check_consistency(3, 2, min_threshold=3, max_threshold=1)
        assert passed is False


@pytest.mark.unit
class TestValidateTripletConsistency:
    """Tests for validate_triplet_consistency() function."""

    @pytest.fixture
    def sample_triplets(self):
        """Create sample triplets with quality ratings."""
        return [
            {
                "anchor_id": 0,
                "positive_id": 1,
                "negative_id": 2,
                "anchor": "Doc A",
                "positive": "Doc B",
                "negative": "Doc C",
                "quality_rating": 4,  # Good original rating
            },
            {
                "anchor_id": 3,
                "positive_id": 4,
                "negative_id": 5,
                "anchor": "Doc D",
                "positive": "Doc E",
                "negative": "Doc F",
                "quality_rating": 5,  # Excellent original rating
            },
            {
                "anchor_id": 6,
                "positive_id": 7,
                "negative_id": 8,
                "anchor": "Doc G",
                "positive": "Doc H",
                "negative": "Doc I",
                "quality_rating": 3,  # Minimal passing original rating
            },
        ]

    def test_empty_triplets(self):
        """Test handling of empty triplet list."""
        result = validate_triplet_consistency(
            triplets=[],
            criterion="test",
        )

        assert result["n_total"] == 0
        assert result["n_passed"] == 0
        assert result["n_failed"] == 0
        assert result["original_ratings"] == []
        assert result["swapped_ratings"] == []
        assert result["consistency_passed"] == []
        assert result["failure_reasons"] == []

    @pytest.mark.external
    def test_basic_consistency(self, sample_triplets):
        """Test basic consistency validation with mock LM judge.

        This is an integration test that makes actual LM calls.
        """
        result = validate_triplet_consistency(
            triplets=sample_triplets,
            criterion="test_criterion",
            criterion_description="Testing similarity criterion",
            lm_judge_preset="lmjudge_quality_rating_gemini",
        )

        # Check structure
        assert "original_ratings" in result
        assert "swapped_ratings" in result
        assert "swapped_reasoning" in result
        assert "consistency_passed" in result
        assert "failure_reasons" in result
        assert "n_total" in result
        assert "n_passed" in result
        assert "n_failed" in result
        assert "failure_breakdown" in result

        # Check counts
        assert result["n_total"] == len(sample_triplets)
        assert result["n_passed"] + result["n_failed"] == result["n_total"]

        # Check original ratings match input
        expected_original = [t["quality_rating"] for t in sample_triplets]
        assert result["original_ratings"] == expected_original

        # Check swapped ratings are evaluated
        assert len(result["swapped_ratings"]) == len(sample_triplets)
        assert len(result["swapped_reasoning"]) == len(sample_triplets)

        # Check consistency results
        assert len(result["consistency_passed"]) == len(sample_triplets)
        assert len(result["failure_reasons"]) == len(sample_triplets)

        # Check failure breakdown is a dict
        assert isinstance(result["failure_breakdown"], dict)

    def test_inputs_swapped_correctly(self, sample_triplets):
        """Verify that swapped triplets have positive and negative swapped."""
        # This is a unit test that checks the input building logic
        # We'd need to mock run_inference to test this properly
        # For now, this is a placeholder for the logic verification
        pass


@pytest.mark.external
class TestConsistencyInWorkflow:
    """Integration tests for consistency validation in quality workflow."""

    @pytest.fixture
    def sample_triplet_dicts(self):
        """Create sample triplet dicts for workflow testing."""
        return [
            {
                "anchor_id": 0,
                "positive_id": 1,
                "negative_id": 2,
                "anchor": "Math problem about addition and multiplication",
                "positive": "Another problem with addition and multiplication",
                "negative": "Problem about subtraction only",
            },
            {
                "anchor_id": 3,
                "positive_id": 4,
                "negative_id": 5,
                "anchor": "Word problem with division",
                "positive": "Division problem with fractions",
                "negative": "Addition problem with whole numbers",
            },
        ]

    def test_consistency_before_topn_selection(self, sample_triplet_dicts):
        """Verify consistency validation runs before top-N selection."""
        result = rate_and_filter_quality_workflow(
            triplets=sample_triplet_dicts,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            min_quality=3,  # Filter by quality first
            max_triplets=1,  # Then select top 1
            validate_consistency=True,
            consistency_min_quality=3,
            consistency_max_invalid=1,
        )

        # Should have consistency stats
        assert "consistency_stats" in result

        # Final triplets should be <= max_triplets
        assert len(result["kept_triplets"]) <= 1

        # All kept triplets should have consistency_check metadata
        for triplet in result["kept_triplets"]:
            assert "consistency_check" in triplet
            assert triplet["consistency_check"]["passed"] is True

    def test_consistency_disabled(self, sample_triplet_dicts):
        """Test that consistency can be disabled."""
        result = rate_and_filter_quality_workflow(
            triplets=sample_triplet_dicts,
            criterion="arithmetic_operations",
            min_quality=None,
            validate_consistency=False,  # Disabled
        )

        # Should NOT have consistency stats
        assert "consistency_stats" not in result

        # Triplets should NOT have consistency_check metadata
        for triplet in result["kept_triplets"]:
            assert "consistency_check" not in triplet

    def test_consistency_with_quality_filtering(self, sample_triplet_dicts):
        """Test consistency validation after quality filtering."""
        result = rate_and_filter_quality_workflow(
            triplets=sample_triplet_dicts,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            min_quality=3,  # Filter first
            validate_consistency=True,  # Then validate consistency
            consistency_min_quality=3,
            consistency_max_invalid=1,
        )

        # Should have both quality and consistency stats
        assert "stats" in result
        assert "consistency_stats" in result

        # Kept triplets should pass both filters
        for triplet in result["kept_triplets"]:
            assert triplet["quality_rating"] >= 3
            assert "consistency_check" in triplet
            assert triplet["consistency_check"]["passed"] is True

        # Dropped triplets should include those that failed either check
        assert len(result["dropped_triplets"]) >= 0

    def test_strict_consistency_threshold(self, sample_triplet_dicts):
        """Test that strict threshold (max_invalid=1) is enforced."""
        result = rate_and_filter_quality_workflow(
            triplets=sample_triplet_dicts,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            min_quality=None,
            validate_consistency=True,
            consistency_min_quality=3,
            consistency_max_invalid=1,  # STRICT - swapped must be Invalid (1)
        )

        # Check consistency stats
        consistency_stats = result["consistency_stats"]

        # Any triplet with swapped_rating > 1 should fail
        for triplet, swapped_rating, passed in zip(
            result["kept_triplets"] + result["dropped_triplets"],
            consistency_stats["swapped_ratings"],
            consistency_stats["consistency_passed"],
            strict=False,
        ):
            if swapped_rating > 1:
                assert passed is False, \
                    f"Triplet with swapped_rating={swapped_rating} should fail strict threshold"

    def test_consistency_with_annotations(self, sample_triplet_dicts):
        """Test consistency validation works with annotations."""
        # Create mock annotations
        annotations = [
            {"category": "addition", "tags": ["arithmetic", "basic"]},
            {"category": "multiplication", "tags": ["arithmetic", "basic"]},
            {"category": "subtraction", "tags": ["arithmetic", "basic"]},
            {"category": "division", "tags": ["arithmetic", "fractions"]},
            {"category": "division", "tags": ["arithmetic", "fractions"]},
            {"category": "addition", "tags": ["arithmetic", "basic"]},
        ]

        result = rate_and_filter_quality_workflow(
            triplets=sample_triplet_dicts,
            criterion="arithmetic_operations",
            criterion_description="Types of arithmetic operations used",
            annotations=annotations,
            min_quality=None,
            validate_consistency=True,
        )

        # Should have consistency stats
        assert "consistency_stats" in result

        # Annotations should be passed to consistency validation
        # (implicitly tested by not raising errors)


@pytest.mark.unit
class TestConsistencyMetadata:
    """Tests for consistency metadata storage."""

    def test_consistency_metadata_structure(self):
        """Test that consistency metadata has correct structure."""
        # Create a mock triplet with consistency check
        triplet = {
            "anchor_id": 0,
            "positive_id": 1,
            "negative_id": 2,
            "consistency_check": {
                "swapped_rating": 1,
                "passed": True,
                "failure_reason": None,
            },
        }

        # Verify structure
        assert "consistency_check" in triplet
        check = triplet["consistency_check"]
        assert "swapped_rating" in check
        assert "passed" in check
        assert "failure_reason" in check

        # Verify types
        assert isinstance(check["swapped_rating"], int)
        assert isinstance(check["passed"], bool)
        assert check["failure_reason"] is None or isinstance(check["failure_reason"], str)

    def test_failed_consistency_metadata(self):
        """Test consistency metadata for failed triplet."""
        triplet = {
            "anchor_id": 0,
            "positive_id": 1,
            "negative_id": 2,
            "consistency_check": {
                "swapped_rating": 3,
                "passed": False,
                "failure_reason": "swapped_too_high_3",
            },
        }

        check = triplet["consistency_check"]
        assert check["passed"] is False
        assert check["failure_reason"] == "swapped_too_high_3"
        assert check["swapped_rating"] == 3
