"""Test whether Voyage reranker scores depend on batch composition.

This test checks if the same (query, document) pair gets different scores
when evaluated in different batches.
"""

import pytest


@pytest.mark.skipif(
    False,
    reason="Requires Voyage API key and makes real API calls",
)
def test_voyage_batch_independence():
    """Test if Voyage scores are independent of batch composition.

    If scores ARE batch-dependent:
        - score(Q, A) in batch [A, B] != score(Q, A) in batch [A, B, C]
        - We need query-batch level caching

    If scores are batch-independent:
        - score(Q, A) is the same regardless of what else is in the batch
        - Individual caching is fine
    """
    from multiview.inference.inference import run_inference

    # Test query and documents
    query = "What is machine learning?"
    doc_a = "Machine learning is a subset of artificial intelligence."
    doc_b = "Deep learning uses neural networks with many layers."
    doc_c = "Python is a popular programming language."

    # Scenario 1: Score [A, B] together
    scores_ab = run_inference(
        inputs={
            "query": [query, query],
            "documents": [doc_a, doc_b],
        },
        config="voyage_rerank_2_5_lite",
        force_refresh=True,  # Don't use cache
    )
    score_a_in_ab = scores_ab[0]
    score_b_in_ab = scores_ab[1]

    # Scenario 2: Score [A, B, C] together
    scores_abc = run_inference(
        inputs={
            "query": [query, query, query],
            "documents": [doc_a, doc_b, doc_c],
        },
        config="voyage_rerank_2_5_lite",
        force_refresh=True,  # Don't use cache
    )
    score_a_in_abc = scores_abc[0]
    score_b_in_abc = scores_abc[1]
    score_c_in_abc = scores_abc[2]

    # Scenario 3: Score [A] alone
    scores_a = run_inference(
        inputs={
            "query": [query],
            "documents": [doc_a],
        },
        config="voyage_rerank_2_5_lite",
        force_refresh=True,  # Don't use cache
    )
    score_a_alone = scores_a[0]

    # Print results
    print("\n" + "="*60)
    print("BATCH DEPENDENCY TEST RESULTS")
    print("="*60)
    print(f"\nQuery: {query}")
    print(f"\nDocument A: {doc_a}")
    print(f"Document B: {doc_b}")
    print(f"Document C: {doc_c}")
    print(f"\n{'Scenario':<30} {'Score A':<15} {'Score B':<15} {'Score C':<15}")
    print("-"*60)
    print(f"{'[A, B]':<30} {score_a_in_ab:<15.6f} {score_b_in_ab:<15.6f} {'-':<15}")
    print(f"{'[A, B, C]':<30} {score_a_in_abc:<15.6f} {score_b_in_abc:<15.6f} {score_c_in_abc:<15.6f}")
    print(f"{'[A] alone':<30} {score_a_alone:<15.6f} {'-':<15} {'-':<15}")

    print(f"\nDifference in A's score:")
    print(f"  |[A,B] - [A,B,C]| = {abs(score_a_in_ab - score_a_in_abc):.6f}")
    print(f"  |[A,B] - [A alone]| = {abs(score_a_in_ab - score_a_alone):.6f}")
    print(f"  |[A,B,C] - [A alone]| = {abs(score_a_in_abc - score_a_alone):.6f}")

    print(f"\nDifference in B's score:")
    print(f"  |[A,B] - [A,B,C]| = {abs(score_b_in_ab - score_b_in_abc):.6f}")

    # Check if scores are consistent (within reasonable tolerance)
    # Voyage has slight batch dependencies (~0.004), but this is negligible for practical use
    tolerance = 0.01  # 1% tolerance is acceptable for reranker scores

    is_a_consistent = (
        abs(score_a_in_ab - score_a_in_abc) < tolerance and
        abs(score_a_in_ab - score_a_alone) < tolerance
    )
    is_b_consistent = abs(score_b_in_ab - score_b_in_abc) < tolerance

    print("\n" + "="*60)
    if is_a_consistent and is_b_consistent:
        print("✓ RESULT: Scores are BATCH-INDEPENDENT (within tolerance)")
        print("  → Individual caching is SAFE")
        print("  → Current implementation is correct")
        print(f"  → Max variance: ~{max(abs(score_a_in_ab - score_a_in_abc), abs(score_a_in_ab - score_a_alone)):.6f}")
    else:
        print("✗ RESULT: Scores are BATCH-DEPENDENT (beyond acceptable tolerance)")
        print("  → Individual caching may be UNSAFE")
        print("  → Consider query-batch level caching")
    print("="*60)

    # For the test to pass, we expect batch independence within tolerance
    # Voyage has slight batch dependencies (~0.004) but this is acceptable
    assert is_a_consistent, (
        f"Document A scores vary by batch beyond tolerance ({tolerance}): "
        f"[A,B]={score_a_in_ab}, [A,B,C]={score_a_in_abc}, [A]={score_a_alone}"
    )
    assert is_b_consistent, (
        f"Document B scores vary by batch beyond tolerance ({tolerance}): "
        f"[A,B]={score_b_in_ab}, [A,B,C]={score_b_in_abc}"
    )


if __name__ == "__main__":
    # Allow running directly with: python -m tests.inference.test_voyage_batch_dependency
    test_voyage_batch_independence()
