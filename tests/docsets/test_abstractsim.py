"""Tests for AbstractSim dataset."""

import pytest

from multiview.docsets import DOCSETS
from multiview.benchmark.task import Task


def test_abstractsim_load_documents():
    """Test that AbstractSim loads documents with correct n*2 labeling."""
    docset = DOCSETS["abstractsim"]({"max_docs": 30, "split": "validation"})
    docs = docset.load_documents()

    # Should load documents
    assert len(docs) > 0, "Should load at least some documents"
    assert len(docs) <= 40, "Should respect max_docs limit (with some tolerance for full rows)"

    # All documents should be dicts with required fields
    for doc in docs:
        assert isinstance(doc, dict), "Documents should be dicts"
        assert "text" in doc, "Documents should have 'text' field"
        assert "abstract_similarity" in doc, "Documents should have 'abstract_similarity' field"
        assert isinstance(doc["text"], str), "Text should be a string"
        assert len(doc["text"]) > 0, "Text should not be empty"

    # Check n*2 labeling scheme
    labels = [doc["abstract_similarity"] for doc in docs]
    class_0_labels = [l for l in labels if "_class_0" in l]
    class_1_labels = [l for l in labels if "_class_1" in l]

    assert len(class_0_labels) > 0, "Should have class_0 labels"
    assert len(class_1_labels) > 0, "Should have class_1 labels"

    # Verify label format
    for label in labels:
        assert label.startswith("row_"), f"Label should start with 'row_': {label}"
        assert ("_class_0" in label) or ("_class_1" in label), f"Label should contain '_class_0' or '_class_1': {label}"

    print(f"\nLoaded {len(docs)} documents:")
    print(f"  {len(class_0_labels)} with class_0 labels")
    print(f"  {len(class_1_labels)} with class_1 labels")


def test_abstractsim_precomputed_annotations():
    """Test that AbstractSim builds precomputed annotations correctly."""
    docset = DOCSETS["abstractsim"]({"max_docs": 20, "split": "validation"})
    docs = docset.load_documents()

    # Should have precomputed annotations
    assert docset.has_precomputed_annotations("abstract_similarity"), "Should have precomputed annotations for abstract_similarity"

    annotations = docset.get_precomputed_annotations("abstract_similarity")

    # All documents should have annotations
    assert len(annotations) == len(docs), "All documents should have annotations"

    # Verify annotation structure
    for doc in docs:
        text = doc["text"]
        label = doc["abstract_similarity"]

        assert text in annotations, f"Document text should be in annotations: {text[:50]}"
        assert "criterion_value" in annotations[text], "Annotation should have criterion_value"
        assert annotations[text]["criterion_value"] == label, "Annotation criterion_value should match document label"

    print(f"\nPrecomputed annotations verified for {len(annotations)} documents")


def test_abstractsim_within_row_grouping():
    """Test that n*2 labeling creates correct within-row groups."""
    docset = DOCSETS["abstractsim"]({"max_docs": 50, "split": "validation"})
    docs = docset.load_documents()

    # Group documents by row
    from collections import defaultdict
    row_groups = defaultdict(lambda: {"class_0": [], "class_1": []})

    for doc in docs:
        label = doc["abstract_similarity"]
        # Extract row number (e.g., "row_5_class_0" -> "row_5")
        row_id = "_".join(label.split("_")[:2])  # "row_X"

        if "_class_0" in label:
            row_groups[row_id]["class_0"].append(doc["text"])
        elif "_class_1" in label:
            row_groups[row_id]["class_1"].append(doc["text"])

    # Check that we have multiple rows
    assert len(row_groups) > 0, "Should have at least one row group"

    # Each row should have class_0 (sentence + good descriptions might not always have class_1)
    for row_id, group in row_groups.items():
        assert len(group["class_0"]) > 0, f"Row {row_id} should have class_0 labels"

    print(f"\nFound {len(row_groups)} row groups:")
    for row_id, group in list(row_groups.items())[:3]:  # Show first 3 rows
        print(f"  {row_id}: {len(group['class_0'])} class_0, {len(group['class_1'])} class_1")


def test_abstractsim_create_prelabeled_triplets():
    """Test that AbstractSim can create prelabeled triplets."""
    task = Task(
        config={
            "document_set": "abstractsim",
            "criterion": "abstract_similarity",
            "max_docs": 100,
            "max_triplets": 10,
            "triplet_style": "prelabeled",
            "selection_strategy": "hard_negatives",
            "split": "validation",
        }
    )

    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    print(f"\n=== Testing abstractsim prelabeled triplets ===")
    print(f"Created {len(task.triplets)} triplets from {len(task.documents)} documents")

    # Should create triplets
    assert len(task.triplets) > 0, "Should create at least some triplets"
    assert len(task.triplets) <= 10, "Should respect max_triplets limit"

    # Verify triplet structure
    for i, triplet in enumerate(task.triplets[:3]):  # Show first 3
        anchor_idx, positive_idx, negative_idx = triplet

        # Get labels
        anchor_label = task.documents[anchor_idx]["abstract_similarity"]
        pos_label = task.documents[positive_idx]["abstract_similarity"]
        neg_label = task.documents[negative_idx]["abstract_similarity"]

        print(f"\nTriplet {i}:")
        print(f"  Anchor: {anchor_label} | {task.documents[anchor_idx]['text'][:60]}...")
        print(f"  Positive: {pos_label} | {task.documents[positive_idx]['text'][:60]}...")
        print(f"  Negative: {neg_label} | {task.documents[negative_idx]['text'][:60]}...")

        # Verify: anchor and positive should share label (both positive or both from same row)
        # This is the n*2 labeling guarantee
        assert anchor_idx != positive_idx, "Anchor and positive should be different documents"
        assert anchor_idx != negative_idx, "Anchor and negative should be different documents"
        assert positive_idx != negative_idx, "Positive and negative should be different documents"


def test_abstractsim_prelabeled_triplet_semantics():
    """Test that prelabeled triplets automatically use sentences as anchors."""
    task = Task(
        config={
            "document_set": "abstractsim",
            "criterion": "abstract_similarity",
            "max_docs": 200,
            "max_triplets": 20,
            "triplet_style": "prelabeled",
            "selection_strategy": "hard_negatives",
            "split": "validation",
        }
    )

    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    print(f"\n=== Testing prelabeled triplet semantics ===")
    print(f"Created {len(task.triplets)} triplets from {len(task.documents)} documents")

    # Verify each triplet has sentence as anchor
    for i, triplet in enumerate(task.triplets):
        anchor_idx, positive_idx, negative_idx = triplet

        anchor = task.documents[anchor_idx]
        pos = task.documents[positive_idx]
        neg = task.documents[negative_idx]

        anchor_label = anchor["abstract_similarity"]
        pos_label = pos["abstract_similarity"]
        neg_label = neg["abstract_similarity"]

        # CRITICAL: Anchor must be marked as anchor
        assert anchor.get("is_anchor", False), \
            f"Triplet {i}: Anchor must be marked with is_anchor=True. Got label={anchor_label}, text={anchor['text'][:50]}"

        # Anchor and positive must both be class_0 (sentence + good descriptions)
        assert "_class_0" in anchor_label, \
            f"Triplet {i}: Anchor must be class_0, got {anchor_label}"
        assert "_class_0" in pos_label, \
            f"Triplet {i}: Positive must be class_0, got {pos_label}"

        # They must be from the same row
        anchor_row = "_".join(anchor_label.split("_")[:2])
        pos_row = "_".join(pos_label.split("_")[:2])
        neg_row = "_".join(neg_label.split("_")[:2])

        assert anchor_row == pos_row, \
            f"Triplet {i}: Anchor and positive must be from same row. " \
            f"Got anchor_row={anchor_row}, pos_row={pos_row}"

        # Negative must have different criterion value (can be from same or different row)
        # For abstractsim: either class_1 from same row, or any class from different row
        same_row = (neg_row == anchor_row)
        if same_row:
            # If same row, must be class_1 (bad description)
            assert "_class_1" in neg_label, \
                f"Triplet {i}: Negative from same row must be class_1, got {neg_label}"
        # else: different row is OK, any class is fine as negative

        # All three documents must be distinct
        assert anchor_idx != positive_idx, f"Triplet {i}: Anchor and positive are the same document"
        assert anchor_idx != negative_idx, f"Triplet {i}: Anchor and negative are the same document"
        assert positive_idx != negative_idx, f"Triplet {i}: Positive and negative are the same document"

        if i < 5:  # Print first 5 for inspection
            print(f"\nTriplet {i}:")
            print(f"  Anchor [SENTENCE] ({anchor_label}):")
            print(f"    {anchor['text']}")
            print(f"  Positive [good desc] ({pos_label}):")
            print(f"    {pos['text']}")
            print(f"  Negative [bad desc] ({neg_label}):")
            print(f"    {neg['text']}")

    print(f"\n✓ All {len(task.triplets)} triplets have sentences as anchors with correct structure")


def test_abstractsim_sentence_coverage_in_triplets():
    """Test that triplets automatically use is_anchor marked documents as anchors."""
    task = Task(
        config={
            "document_set": "abstractsim",
            "criterion": "abstract_similarity",
            "max_docs": 200,
            "max_triplets": 30,
            "triplet_style": "prelabeled",  # Automatically uses is_anchor markers
            "selection_strategy": "hard_negatives",
            "split": "validation",
        }
    )

    task.load_documents()
    task.annotate_documents()
    task.create_triplets()

    print(f"\n=== Sentence coverage in triplets ===")
    print(f"Created {len(task.triplets)} triplets from {len(task.documents)} documents")

    # Count how many triplets include the actual sentence
    triplets_with_sentence = 0
    triplets_without_sentence = 0

    for i, triplet in enumerate(task.triplets):
        anchor_idx, pos_idx, neg_idx = triplet

        anchor = task.documents[anchor_idx]
        pos = task.documents[pos_idx]
        neg = task.documents[neg_idx]

        # Check if anchor is marked as anchor
        is_anchor = anchor.get("is_anchor", False)
        is_sentence = anchor.get("is_sentence", False)

        if is_anchor:
            triplets_with_sentence += 1
            if i < 5:  # Show first 5
                print(f"\nTriplet {i} (✓ MARKED ANCHOR):")
                print(f"  Anchor [is_anchor=True, is_sentence={is_sentence}]: {anchor['text']}")
                print(f"  Positive [good desc]: {pos['text']}")
                print(f"  Negative [bad desc]: {neg['text']}")
        else:
            triplets_without_sentence += 1
            if triplets_without_sentence <= 3:  # Show first 3 failures
                print(f"\nTriplet {i} (✗ ANCHOR NOT MARKED):")
                print(f"  Anchor [is_anchor=False]: {anchor['text']}")
                print(f"  Positive: {pos['text']}")
                print(f"  Negative: {neg['text']}")

    print(f"\n📊 Summary:")
    print(f"  Triplets WITH sentence as anchor: {triplets_with_sentence}/{len(task.triplets)} ({100*triplets_with_sentence/len(task.triplets):.1f}%)")
    print(f"  Triplets WITHOUT sentence: {triplets_without_sentence}/{len(task.triplets)} ({100*triplets_without_sentence/len(task.triplets):.1f}%)")

    # With sentence_anchor triplet style, ALL triplets should have sentence as anchor
    assert triplets_with_sentence == len(task.triplets), \
        f"Expected all {len(task.triplets)} triplets to have sentence as anchor, but only {triplets_with_sentence} do"
    assert triplets_without_sentence == 0, \
        f"Expected 0 triplets without sentences, but got {triplets_without_sentence}"

    print("\n✓ All triplets have sentences as anchors!")


def test_abstractsim_random_triplets():
    """Test that AbstractSim works with random triplet creation."""
    task = Task(
        config={
            "document_set": "abstractsim",
            "criterion": "abstract_similarity",  # Not used for random style
            "max_docs": 20,
            "max_triplets": 5,
            "triplet_style": "random",
            "split": "validation",
        }
    )

    task.load_documents()
    task.create_triplets()

    # Basic assertions
    assert len(task.triplets) == 5, "Should create exactly max_triplets"
    assert all(len(triplet) == 3 for triplet in task.triplets), "Each triplet should have 3 elements"

    # Check that each triplet has distinct documents
    for triplet in task.triplets:
        anchor_idx, pos_idx, neg_idx = triplet
        assert anchor_idx != pos_idx != neg_idx, "All indices in triplet should be distinct"

    print(f"\n=== Random triplets test passed ===")
    print(f"Created {len(task.triplets)} random triplets")
