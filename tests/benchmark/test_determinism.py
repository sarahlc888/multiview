"""Tests for determinism in triplet creation and caching.

These tests verify that:
1. Candidate selection is deterministic across multiple runs
2. Prompts generated are identical (same hash)
3. Cache hits work correctly (no redundant LM calls)
"""

import numpy as np
import pytest

from multiview.benchmark.triplets.candidate_selection import (
    merge_candidate_pools,
    select_candidates_bm25,
    select_candidates_jaccard,
    select_spurious_hard_negatives,
)
from multiview.inference import cost_tracker


def test_argsort_stability():
    """Test that np.argsort with kind='stable' is deterministic with tied scores."""
    # Create scores with many ties
    scores = np.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.5])

    # Run argsort multiple times with stable sort
    results = []
    for _ in range(5):
        result = np.argsort(scores, kind='stable')[::-1]
        results.append(result.tolist())

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result, "Stable sort should produce identical results"

    # Verify tied elements maintain consistent order (they'll be reversed due to [::-1])
    # Indices with score 0.0 are [1, 3, 4, 6, 7]
    # After stable sort descending, they should appear in reverse: [7, 6, 4, 3, 1]
    zero_indices_in_result = [idx for idx in first_result if scores[idx] == 0.0]
    expected_zero_order = [7, 6, 4, 3, 1]  # Reversed from original insertion order
    assert (
        zero_indices_in_result == expected_zero_order
    ), f"Tied elements should maintain stable order. Got {zero_indices_in_result}"


def test_bm25_candidate_selection_deterministic():
    """Test that BM25 candidate selection is deterministic across runs."""
    # Create sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A lazy cat sleeps on the warm couch",
        "The dog chases the cat in the garden",
        "Quick movements scare the lazy animals",
        "Brown and white cats are common pets",
    ]

    # Create annotations (minimal - just need something)
    annotations = [{"summary": doc} for doc in documents]

    # Run candidate selection multiple times
    anchor_idx = 0
    k = 3

    results = []
    for _ in range(3):
        candidates = select_candidates_bm25(
            documents=documents,
            annotations=annotations,
            anchor_idx=anchor_idx,
            k=k,
            use_summary=False,
        )
        # Extract just the indices (not scores)
        indices = [idx for idx, score in candidates]
        results.append(indices)

    # All runs should produce the same candidate order
    first_result = results[0]
    for result in results[1:]:
        assert (
            result == first_result
        ), f"BM25 candidate selection should be deterministic. Got {result} vs {first_result}"


def test_jaccard_candidate_selection_deterministic():
    """Test that Jaccard candidate selection is deterministic across runs."""
    # Create annotations with tags
    annotations = [
        {"tags": {"tag_a": True, "tag_b": True}},
        {"tags": {"tag_b": True, "tag_c": True}},
        {"tags": {"tag_a": True, "tag_c": True}},
        {"tags": {"tag_d": True}},
        {"tags": {"tag_a": True, "tag_b": True, "tag_d": True}},
    ]

    anchor_idx = 0
    k = 3

    results = []
    for _ in range(3):
        candidates = select_candidates_jaccard(
            annotations=annotations,
            anchor_idx=anchor_idx,
            k=k,
            use_spurious=False,
        )
        indices = [idx for idx, score in candidates]
        results.append(indices)

    # All runs should produce the same candidate order
    first_result = results[0]
    for result in results[1:]:
        assert (
            result == first_result
        ), f"Jaccard candidate selection should be deterministic. Got {result} vs {first_result}"


def test_spurious_hard_negatives_deterministic():
    """Test that spurious hard negative selection is deterministic."""
    annotations = [
        {
            "tags": {"real_a": True, "real_b": True},
            "spurious_tags": {"surface_x": True, "surface_y": True},
        },
        {
            "tags": {"real_c": True},
            "spurious_tags": {"surface_x": True, "surface_y": True},
        },
        {
            "tags": {"real_d": True},
            "spurious_tags": {"surface_y": True, "surface_z": True},
        },
        {
            "tags": {"real_a": True},
            "spurious_tags": {"surface_z": True},
        },
    ]

    anchor_idx = 0
    k = 3

    results = []
    for _ in range(3):
        candidates = select_spurious_hard_negatives(
            annotations=annotations, anchor_idx=anchor_idx, k=k
        )
        indices = [idx for idx, score in candidates]
        results.append(indices)

    first_result = results[0]
    for result in results[1:]:
        assert (
            result == first_result
        ), f"Spurious hard negative selection should be deterministic. Got {result} vs {first_result}"


def test_rrf_merge_deterministic():
    """Test that RRF merging is deterministic with tied scores."""
    # Create candidate lists with some overlap
    bm25_candidates = [(0, 0.5), (1, 0.3), (2, 0.2)]
    jaccard_candidates = [(1, 0.8), (3, 0.6), (0, 0.4)]
    spurious_candidates = [(2, 0.7), (3, 0.5), (4, 0.3)]

    results = []
    for _ in range(3):
        merged = merge_candidate_pools(
            bm25_candidates, jaccard_candidates, spurious_candidates, use_rrf=True
        )
        results.append(merged)

    first_result = results[0]
    for result in results[1:]:
        assert (
            result == first_result
        ), f"RRF merge should be deterministic. Got {result} vs {first_result}"


@pytest.mark.external
def test_cache_hits_on_second_run():
    """Test that running the same triplet creation twice uses cache on second run.

    This is an integration test that verifies the full pipeline produces
    identical prompts and cache hits work correctly.
    """
    from multiview.benchmark.task import Task

    # Reset cost tracker
    cost_tracker.reset()

    config = {
        "document_set": "crossword_clues",
        "criterion": "topic",
        "max_docs": 20,
        "max_triplets": 5,
        "triplet_style": "lm_tags",
        "use_cache": True,
        "run_name": "test_determinism",
        "seed": 42,
    }

    # First run - should make LM calls
    task1 = Task(config)
    task1.load_documents()
    task1.annotate_documents()
    task1.create_triplets()

    requests_after_first = cost_tracker.get_total_requests()
    cache_hits_after_first = cost_tracker.get_total_cache_hits()

    print(f"\nFirst run - Requests: {requests_after_first}, Cache hits: {cache_hits_after_first}")

    # Second run - should use cache
    task2 = Task(config)
    task2.load_documents()
    task2.annotate_documents()
    task2.create_triplets()

    requests_after_second = cost_tracker.get_total_requests()
    cache_hits_after_second = cost_tracker.get_total_cache_hits()

    new_requests = requests_after_second - requests_after_first
    new_cache_hits = cache_hits_after_second - cache_hits_after_first

    print(f"Second run - New requests: {new_requests}, New cache hits: {new_cache_hits}")

    # The second run should have made NO new API calls
    assert (
        new_requests == 0
    ), f"Second run should use cache only, but made {new_requests} new API calls"

    # The second run should have some cache hits
    assert (
        new_cache_hits > 0
    ), "Second run should have cache hits from annotations/triplet selection"

    # Verify triplets are identical
    assert len(task1.triplets) == len(
        task2.triplets
    ), "Both runs should produce same number of triplets"
    assert (
        task1.triplets == task2.triplets
    ), "Both runs should produce identical triplets"


def test_prompt_hash_consistency():
    """Test that prompts generate consistent hashes across runs."""
    from multiview.inference.caching import hash_prompt

    # Test with various prompts
    prompts = [
        "Simple prompt",
        "Prompt with numbers: 123, 456, 789",
        "Prompt with special chars: !@#$%^&*()",
        "Multi\nline\nprompt",
    ]

    for prompt in prompts:
        hashes = []
        for _ in range(5):
            h = hash_prompt(prompt)
            hashes.append(h)

        # All hashes should be identical
        assert len(set(hashes)) == 1, f"Hash should be deterministic for: {prompt}"


def test_candidate_ordering_with_tied_scores():
    """Test that candidates with tied scores maintain consistent ordering."""
    # Create documents where many will have identical BM25 scores
    documents = [
        "unique anchor document with specific words",
        "completely different topic A",
        "completely different topic B",
        "completely different topic C",
        "completely different topic D",
        "completely different topic E",
    ]

    # Documents 1-5 should all have very similar (likely 0.0) BM25 scores to anchor
    annotations = [{"summary": doc} for doc in documents]

    anchor_idx = 0
    results = []

    # Run multiple times
    for _ in range(5):
        candidates = select_candidates_bm25(
            documents=documents,
            annotations=annotations,
            anchor_idx=anchor_idx,
            k=5,
            use_summary=False,
        )
        indices = [idx for idx, score in candidates]
        results.append(indices)

    # Verify all runs produce identical ordering
    first = results[0]
    for i, result in enumerate(results[1:], 1):
        assert (
            result == first
        ), f"Run {i+1} produced different ordering: {result} vs {first}"


@pytest.mark.external
def test_end_to_end_cache_with_reuse_false():
    """Test that LM calls are cached even with reuse_cached_triplets=false.

    This is critical: reuse_cached_triplets=false means we regenerate the triplet
    selection logic (run the pipeline again), but the LM calls to generate annotations
    and judge triplets should still be cached.
    """
    from multiview.benchmark.task import Task

    # Reset cost tracker
    cost_tracker.reset()

    config = {
        "document_set": "gsm8k",
        "criterion": "problem_type",
        "max_docs": 20,
        "max_triplets": 5,
        "triplet_style": "lm_tags",
        "use_cache": True,
        "reuse_cached_triplets": False,  # Force regeneration of triplets
        "run_name": "test_e2e_determinism",
        "seed": 42,
    }

    print("\n=== First Run ===")
    # First run - should make LM calls
    task1 = Task(config)
    task1.load_documents()
    task1.annotate_documents()
    task1.create_triplets()

    requests_run1 = cost_tracker.get_total_requests()
    cache_hits_run1 = cost_tracker.get_total_cache_hits()

    print(f"Run 1 - API requests: {requests_run1}, Cache hits: {cache_hits_run1}")
    print(f"Run 1 - Created {len(task1.triplets)} triplets")

    # Verify some requests were made
    assert requests_run1 > 0, "First run should make some API requests"

    print("\n=== Second Run (reuse_cached_triplets=False) ===")
    # Second run - same config, reuse_cached_triplets=False
    # This forces triplet creation logic to run again, but LM calls should be cached
    task2 = Task(config)
    task2.load_documents()
    task2.annotate_documents()
    task2.create_triplets()

    requests_run2 = cost_tracker.get_total_requests()
    cache_hits_run2 = cost_tracker.get_total_cache_hits()

    new_requests = requests_run2 - requests_run1
    new_cache_hits = cache_hits_run2 - cache_hits_run1

    print(f"Run 2 - New API requests: {new_requests}, New cache hits: {new_cache_hits}")
    print(f"Run 2 - Created {len(task2.triplets)} triplets")

    # Critical assertion: NO new API calls should have been made
    assert (
        new_requests == 0
    ), f"Second run should make 0 API calls (all cached), but made {new_requests}"

    # Should have cache hits from annotations and triplet judgments
    assert new_cache_hits > 0, "Second run should have cache hits"

    # Verify triplets are identical despite reuse_cached_triplets=False
    # This proves the pipeline is deterministic
    assert len(task1.triplets) == len(
        task2.triplets
    ), "Both runs should produce same number of triplets"

    # Sort triplets for comparison (order might differ)
    triplets1_sorted = sorted(task1.triplets)
    triplets2_sorted = sorted(task2.triplets)

    assert (
        triplets1_sorted == triplets2_sorted
    ), "Both runs should produce identical triplets (deterministic pipeline)"

    print("\n✅ SUCCESS: Pipeline is deterministic, all LM calls cached on second run")
    print(f"   Total API calls: {requests_run1}")
    print(f"   Total cache hits on run 2: {new_cache_hits}")


@pytest.mark.external
def test_multiple_datasets_cache_consistency():
    """Test cache consistency across multiple datasets and criteria.

    Ensures that different tasks properly isolate their caches and that
    running the same task twice always uses cache.
    """
    from multiview.benchmark.task import Task

    cost_tracker.reset()

    # Test multiple dataset/criterion combinations
    configs = [
        {
            "document_set": "crossword_clues",
            "criterion": "topic",
            "max_docs": 15,
            "max_triplets": 3,
            "triplet_style": "lm_tags",
        },
        {
            "document_set": "gsm8k",
            "criterion": "arithmetic",
            "max_docs": 15,
            "max_triplets": 3,
            "triplet_style": "lm_tags",
        },
    ]

    results = {}

    for i, base_config in enumerate(configs):
        config = {
            **base_config,
            "use_cache": True,
            "reuse_cached_triplets": False,
            "run_name": "test_multi_dataset",
            "seed": 42,
        }

        dataset_name = f"{config['document_set']}/{config['criterion']}"
        print(f"\n=== Dataset {i+1}: {dataset_name} ===")

        # First run
        requests_before = cost_tracker.get_total_requests()

        task1 = Task(config)
        task1.load_documents()
        task1.annotate_documents()
        task1.create_triplets()

        requests_after_first = cost_tracker.get_total_requests()
        first_run_requests = requests_after_first - requests_before

        # Second run
        task2 = Task(config)
        task2.load_documents()
        task2.annotate_documents()
        task2.create_triplets()

        requests_after_second = cost_tracker.get_total_requests()
        second_run_requests = requests_after_second - requests_after_first

        print(f"  First run: {first_run_requests} API calls")
        print(f"  Second run: {second_run_requests} API calls")

        # Second run should make NO new requests
        assert (
            second_run_requests == 0
        ), f"{dataset_name}: Second run should use cache only"

        # Triplets should be identical
        assert (
            sorted(task1.triplets) == sorted(task2.triplets)
        ), f"{dataset_name}: Triplets should be identical"

        results[dataset_name] = {
            "first_run_requests": first_run_requests,
            "second_run_requests": second_run_requests,
            "num_triplets": len(task1.triplets),
        }

    print("\n=== Summary ===")
    for dataset_name, stats in results.items():
        print(
            f"{dataset_name}: {stats['first_run_requests']} requests → "
            f"0 requests (cached), {stats['num_triplets']} triplets ✅"
        )
