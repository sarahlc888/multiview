"""Tests for triplet utilities."""

import pytest

from multiview.benchmark.task import Task


@pytest.mark.parametrize("dataset", ["gsm8k", "rocstories", "crossword_clues", "hackernews", "analogies", "infinite_prompts", "infinite_chats", "dickinson", "moralfables"])
def test_create_random_triplets(dataset):
    """Test random triplet creation across different datasets."""
    # Create a task with random triplets
    task = Task(
        config={
            "document_set": dataset,
            "criterion": "word_count",  # Not used for random style, but required
            "max_docs": 10,
            "max_triplets": 5,
            "triplet_style": "random",
        }
    )

    task.load_documents()
    # Skip annotation for random style
    if task.triplet_style != "random":
        task.annotate_documents()
    task.create_triplets()

    print(f"\n=== Testing {dataset} ===")
    print(f"Created {len(task.triplets)} triplets:")

    def format_doc(doc):
        """Format document for display, handling both string and dict formats."""
        if isinstance(doc, str):
            return doc
        elif isinstance(doc, dict):
            return str(doc)
        else:
            return str(doc)

    for i, triplet in enumerate(task.triplets):
        anchor_idx, positive_idx, negative_idx = triplet
        print(f"\nTriplet {i}:")
        print(f"  [Anchor {anchor_idx}] {format_doc(task.documents[anchor_idx])}")
        print(f"  [Positive {positive_idx}] {format_doc(task.documents[positive_idx])}")
        print(f"  [Negative {negative_idx}] {format_doc(task.documents[negative_idx])}")

    # Basic assertions
    assert len(task.triplets) == 5
    assert all(len(triplet) == 3 for triplet in task.triplets)
    # Check that each triplet has distinct documents
    assert all(len({id(doc) for doc in triplet}) == 3 for triplet in task.triplets)
