"""Integration tests for pipeline quality validation.

Tests that the full pipeline generates data meeting minimum quality thresholds.
Enables quick iteration to tune pipeline until quality bar is met.

Two test types:
1. test_custom_triplets: LM-generated triplets with quality rating thresholds
2. test_prelabeled_triplets: Ground-truth labeled triplets with LM judge accuracy thresholds

Run with: pytest tests/benchmark/test_quality_validation.py -v -s
"""

import pytest

from multiview.benchmark.benchmark import Benchmark
from multiview.benchmark.task import Task
from multiview.benchmark.evaluation_utils import build_triplet_dicts
from multiview.eval.lm_judge import evaluate_with_lm_judge_triplet

pytestmark = [pytest.mark.integration, pytest.mark.external]

# ============================================================================
# Configuration Constants
# ============================================================================

# Quality thresholds
MIN_IDEAL_PCT = 40.0  # At least 40% rated 4 (ideal)
MAX_INVALID_PCT = 10.0  # At most 10% rated 1 (invalid)
MIN_ACCURACY_PCT = 70.0  # At least 70% LM judge accuracy for prelabeled

# LM Judge configuration
LM_JUDGE_PRESET = "lmjudge_triplet_plaintext_binaryhard_gemini_flash"
LM_JUDGE_BIDIRECTIONAL = True  # Evaluate each triplet in both directions

# Custom triplets (LM-generated) configuration
CUSTOM_MAX_DOCS = 10
CUSTOM_MAX_TRIPLETS = 5
CUSTOM_N_SCHEMA_SAMPLES = 5

# Prelabeled triplets configuration
PRELABELED_MAX_DOCS = 60  # Enough for good triplet diversity
PRELABELED_MAX_TRIPLETS = 20  # Results in 40 evals with bidirectional
PRELABELED_SELECTION_STRATEGY = "hard_negatives"  # "random" or "hard_negatives"
PRELABELED_SEED = 42

# Display configuration
N_EXAMPLE_TRIPLETS = 3  # Number of example triplets to show
N_EXAMPLE_JUDGMENTS = 4  # Number of example judgments to show


# ============================================================================
# Shared Helpers
# ============================================================================

def evaluate_with_lm_judge(task, criterion_description=None):
    """Shared helper to evaluate triplets with LM judge.

    Args:
        task: Task object with documents and triplets
        criterion_description: Optional description, uses metadata if not provided

    Returns:
        Dict with outcomes, accuracy, and detailed logs
    """
    # Get document texts
    document_texts = [
        task.document_set.get_document_text(doc) for doc in task.documents
    ]

    # Build triplet dicts
    triplet_dicts = build_triplet_dicts(document_texts, task.triplets)

    # Get criterion description from metadata if not provided
    if not criterion_description:
        criterion_metadata = task.document_set.get_criterion_metadata(task.criterion_name) or {}
        criterion_description = criterion_metadata.get("description")

    # Evaluate with LM judge
    results = evaluate_with_lm_judge_triplet(
        triplets=triplet_dicts,
        criterion=task.criterion_name,
        criterion_description=criterion_description,
        lm_judge_preset=LM_JUDGE_PRESET,
        cache_alias="dev",
        bidirectional=LM_JUDGE_BIDIRECTIONAL,
    )

    # Calculate accuracy
    outcomes = results.get("outcomes", [])
    n_total = len(outcomes)
    n_correct = sum(1 for o in outcomes if o == 1)
    n_incorrect = sum(1 for o in outcomes if o == -1)
    n_ties = sum(1 for o in outcomes if o == 0)
    n_judged = n_correct + n_incorrect
    accuracy = (n_correct / n_judged * 100) if n_judged > 0 else 0

    return {
        "outcomes": outcomes,
        "accuracy": accuracy,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_ties": n_ties,
        "triplet_logs": results.get("triplet_logs", []),
    }


def print_lm_judge_results(dataset, criterion, results):
    """Print LM judge evaluation results in a consistent format.

    Args:
        dataset: Dataset name
        criterion: Criterion name
        results: Results dict from evaluate_with_lm_judge
    """
    print(f"\n{'=' * 80}")
    print(f"LM JUDGE RESULTS: {dataset.upper()} / {criterion}")
    print(f"{'=' * 80}")
    print(f"Total evaluations: {results['n_total']} (bidirectional)")
    print(f"Correct: {results['n_correct']}")
    print(f"Incorrect: {results['n_incorrect']}")
    print(f"Ties: {results['n_ties']}")
    print(f"Accuracy (excluding ties): {results['accuracy']:.1f}%")
    print(f"{'=' * 80}")


def print_example_triplets(task, triplet_dicts, criterion, n_examples=N_EXAMPLE_TRIPLETS):
    """Print example triplets with their labels.

    Args:
        task: Task object
        triplet_dicts: List of triplet dicts
        criterion: Criterion name
        n_examples: Number of examples to show
    """
    print("\n" + "-" * 80)
    print("EXAMPLE TRIPLETS:")
    print("-" * 80)

    # Helper to get label from document
    def get_doc_label(doc_text):
        """Get criterion label for a document."""
        # For string documents with precomputed annotations
        if task.document_set.has_precomputed_annotations(criterion):
            precomputed = task.document_set.get_precomputed_annotations(criterion)
            return precomputed.get(doc_text, {}).get("prelabel", "unknown")
        # For dict documents with embedded labels
        for doc in task.documents:
            if task.document_set.get_document_text(doc) == doc_text:
                if isinstance(doc, dict):
                    return doc.get(criterion, "unknown")
        return "unknown"

    for i, triplet in enumerate(triplet_dicts[:n_examples]):
        anchor_text = triplet["anchor"]
        pos_text = triplet["positive"]
        neg_text = triplet["negative"]

        anchor_label = get_doc_label(anchor_text)
        pos_label = get_doc_label(pos_text)
        neg_label = get_doc_label(neg_text)

        print(f"\nTriplet {i}:")
        print(f"  Anchor: {anchor_text[:80]}...")
        print(f"    {criterion}: {anchor_label}")
        print(f"  Positive: {pos_text[:80]}...")
        print(f"    {criterion}: {pos_label}")
        print(f"  Negative: {neg_text[:80]}...")
        print(f"    {criterion}: {neg_label}")


def print_example_judgments(results, n_examples=N_EXAMPLE_JUDGMENTS):
    """Print example LM judge judgments with reasoning.

    Args:
        results: Results dict from evaluate_with_lm_judge
        n_examples: Number of examples to show
    """
    triplet_logs = results.get("triplet_logs", [])
    if not triplet_logs:
        return

    print("\nEXAMPLE JUDGMENTS:")
    print("-" * 80)
    for i, log in enumerate(triplet_logs[:n_examples]):
        outcome = log.get("outcome")
        direction = log.get("direction", "unknown")
        reasoning = log.get("lm_judge_reasoning", "")

        outcome_str = "✓ CORRECT" if outcome == 1 else "✗ INCORRECT" if outcome == -1 else "~ TIE"
        print(f"\n[{i}] {outcome_str} ({direction} direction)")
        print(f"Reasoning: {reasoning[:200]}...")


# ============================================================================
# Test 1: Custom (LM-generated) Triplets with Quality Rating
# ============================================================================

def test_custom_triplets():
    """Test LM-generated triplets meet quality thresholds.

    Validates:
    - At least 40% of triplets are rated "ideal" (4)
    - At most 10% of triplets are rated "invalid" (1)
    """
    # Config (small for fast iteration)
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic_operations",
        "criterion_description": "Types of arithmetic operations used",
        "max_docs": CUSTOM_MAX_DOCS,
        "max_triplets": CUSTOM_MAX_TRIPLETS,
        "triplet_style": "lm_all",
        "candidate_strategy": "multi",
        "use_spurious_hard_negs": True,
        "n_schema_samples": CUSTOM_N_SCHEMA_SAMPLES,
    }

    print("\n" + "=" * 80)
    print("TEST: CUSTOM TRIPLETS (LM-GENERATED) QUALITY")
    print("=" * 80)

    # Run pipeline (same as run_eval.py)
    task = Task(config=config)
    task.load_documents()
    task.augment_with_synthetic_documents()
    task.annotate_documents()
    task.create_triplets()
    quality_stats = task.rate_and_filter_quality(min_quality=None)

    # Evaluate
    benchmark = Benchmark([task])
    results, _ = benchmark.evaluate({
        "bm25": [{"name": "bm25"}],
        "lm_judge_triplet": [{
            "preset": "lmjudge_triplet_with_annotation_gemini",
            "name": "gemini_with_annotation",
        }],
    })

    # Validate quality thresholds
    percentages = quality_stats["percentages"]
    ideal_pct = percentages[4]
    invalid_pct = percentages[1]

    assert ideal_pct >= MIN_IDEAL_PCT, \
        f"Ideal {ideal_pct:.1f}% < {MIN_IDEAL_PCT}%"
    assert invalid_pct <= MAX_INVALID_PCT, \
        f"Invalid {invalid_pct:.1f}% > {MAX_INVALID_PCT}%"

    # Print summary
    print("\n" + "=" * 60)
    print("Quality Distribution:")
    for level in [4, 3, 2, 1]:
        label = ["", "Invalid", "Ambiguous", "Trivial", "Ideal"][level]
        print(f"  {level} ({label:10s}): {quality_stats['counts'][level]:3d} "
              f"({percentages[level]:5.1f}%)")

    print("\nEvaluation Results:")
    task_name = list(results.keys())[0]
    for method, metrics in results[task_name].items():
        print(f"  {method:25s}: {metrics['accuracy']:6.2%} "
              f"({metrics['n_correct']}/{metrics['n_total']})")
    print("=" * 60)


# ============================================================================
# Test 2: Prelabeled Triplets with LM Judge Accuracy
# ============================================================================

@pytest.mark.parametrize("dataset,criterion,docset_config,triplet_style", [
    ("abstractsim", "abstract_similarity", {"split": "validation"}, "prelabeled"),
    ("abstractsim", "abstraction_level", {"split": "validation"}, "prelabeled"),
    ("arxiv_abstract_sentences", "source_abstract", {}, "prelabeled"),
    ("inb_nytclustering", "topic", {"subset": "topic"}, "prelabeled"),
    ("inb_ratemyprof", "cluster", {}, "prelabeled"),
    ("inb_feedbacks", "cluster", {}, "prelabeled"),
    ("inb_fewrel", "cluster", {}, "prelabeled"),
    ("inb_fewnerd", "cluster", {}, "prelabeled"),
    ("inb_fewevent", "cluster", {}, "prelabeled"),
    ("inb_intent_emotion", "intent_similarity", {"subset": "intent"}, "intent_emotion"),
    ("inspired", "movie_recommendation", {"split": "train"}, "prelabeled"),
    ("bills", "topic", {"text_field": "summary"}, "prelabeled"),
    ("bills", "subtopic", {"text_field": "summary"}, "prelabeled"),
    ("trex", "relation", {"split": "validation", "min_relation_freq": 3}, "kgc"),
    ("infinite_prompts", "categories", {}, "prelabeled"),
])
def test_prelabeled_triplets(dataset, criterion, docset_config, triplet_style):
    """Test prelabeled triplets meet LM judge accuracy threshold.

    Validates that triplets from ground-truth labels are discriminative
    enough for an LM judge to identify the odd-one-out with >70% accuracy.

    Args:
        dataset: Dataset name
        criterion: Criterion to use for triplet creation
        docset_config: Dataset-specific configuration
        triplet_style: Triplet generation style
    """
    # Create task with prelabeled triplets
    task = Task(
        config={
            "document_set": dataset,
            "criterion": criterion,
            "max_docs": PRELABELED_MAX_DOCS,
            "max_triplets": PRELABELED_MAX_TRIPLETS,
            "triplet_style": triplet_style,
            "selection_strategy": PRELABELED_SELECTION_STRATEGY,
            "seed": PRELABELED_SEED,
            "config": docset_config,
        }
    )

    print("\n" + "=" * 80)
    print(f"TEST: PRELABELED TRIPLETS - {dataset.upper()} / {criterion}")
    print("=" * 80)

    # Load documents and create triplets
    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    print(f"\nLoaded {len(task.documents)} documents from {dataset}")
    print(f"Created {len(task.triplets)} triplets using criterion: {criterion}")

    # Get triplet dicts for display
    document_texts = [
        task.document_set.get_document_text(doc) for doc in task.documents
    ]
    triplet_dicts = build_triplet_dicts(document_texts, task.triplets)

    # Show example triplets
    print_example_triplets(task, triplet_dicts, criterion)

    # Evaluate with LM judge
    print("\n" + "-" * 80)
    print("EVALUATING WITH LM JUDGE...")
    print("-" * 80)

    results = evaluate_with_lm_judge(task)

    # Print results
    print_lm_judge_results(dataset, criterion, results)
    print_example_judgments(results)

    # Assert accuracy > 70%
    accuracy = results["accuracy"]
    assert accuracy > MIN_ACCURACY_PCT, (
        f"LM judge accuracy ({accuracy:.1f}%) is below {MIN_ACCURACY_PCT}% threshold for {dataset}/{criterion}. "
        f"This suggests the triplets are not discriminative enough, "
        f"or the LM judge cannot reliably distinguish {criterion} differences."
    )

    print(f"\n✓ LM judge achieves {accuracy:.1f}% accuracy on {dataset}/{criterion} triplets")
    print("=" * 80)
