"""Tests for AidanBench dataset."""

import pytest

from multiview.benchmark.task import Task
from multiview.docsets import DOCSETS


def test_aidanbench_load_documents():
    """Test that AidanBench loads documents correctly."""
    docset = DOCSETS["aidanbench"]({"question_id": 0, "max_docs": 10})
    docs = docset.load_documents()

    # Should load documents
    assert len(docs) > 0, "Should load at least some documents"
    assert len(docs) <= 10, "Should respect max_docs limit"

    # All documents should be strings (answer text)
    for doc in docs:
        assert isinstance(doc, str), "Documents should be strings"
        assert len(doc) > 0, "Documents should not be empty"

    print(f"\nLoaded {len(docs)} answer documents for question 0")


def test_aidanbench_list_questions():
    """Test that AidanBench can list available questions."""
    from multiview.docsets.aidanbench import AidanBenchDocSet

    questions = AidanBenchDocSet.list_questions()

    # Should have questions
    assert len(questions) > 0, "Should have at least some questions"
    assert all(isinstance(q, str) for q in questions), "All questions should be strings"
    assert all(len(q) > 0 for q in questions), "All questions should be non-empty"

    print(f"\nFound {len(questions)} questions in AidanBench")
    print(f"First 3 questions:")
    for i, q in enumerate(questions[:3]):
        print(f"  {i}: {q[:80]}...")


def test_aidanbench_question_selection_by_id():
    """Test that AidanBench can select questions by ID."""
    # Load first question
    docset1 = DOCSETS["aidanbench"]({"question_id": 0, "max_docs": 5})
    docs1 = docset1.load_documents()

    # Load second question
    docset2 = DOCSETS["aidanbench"]({"question_id": 1, "max_docs": 5})
    docs2 = docset2.load_documents()

    # Both should load documents
    assert len(docs1) > 0, "Question 0 should have documents"
    assert len(docs2) > 0, "Question 1 should have documents"

    # Documents should be strings
    assert all(isinstance(d, str) for d in docs1), "Question 0 documents should be strings"
    assert all(isinstance(d, str) for d in docs2), "Question 1 documents should be strings"

    print(f"\nQuestion 0: {len(docs1)} answers")
    print(f"Question 1: {len(docs2)} answers")


def test_aidanbench_question_selection_by_text():
    """Test that AidanBench can select questions by text search."""
    from multiview.docsets.aidanbench import AidanBenchDocSet

    questions = AidanBenchDocSet.list_questions()

    # Find a question with a common word
    search_term = None
    for word in ["create", "design", "describe", "write"]:
        matching = [q for q in questions if word.lower() in q.lower()]
        if matching:
            search_term = word
            break

    if search_term:
        docset = DOCSETS["aidanbench"]({"question_text": search_term, "max_docs": 5})
        docs = docset.load_documents()

        assert len(docs) > 0, f"Should load documents for question containing '{search_term}'"
        assert all(isinstance(d, str) for d in docs), "Documents should be strings"

        print(f"\nFound question containing '{search_term}'")
        print(f"Loaded {len(docs)} answers")
    else:
        pytest.skip("No common search terms found in questions")


def test_aidanbench_deduplication():
    """Test that AidanBench deduplicates answers correctly."""
    docset = DOCSETS["aidanbench"]({"question_id": 0, "max_docs": 100})
    docs = docset.load_documents()

    # Check that all documents are unique
    unique_docs = set(docs)
    assert len(unique_docs) == len(docs), "All documents should be unique (deduplicated)"

    print(f"\nLoaded {len(docs)} unique answers (all deduplicated)")


def test_aidanbench_max_docs_limit():
    """Test that max_docs parameter is respected."""
    # Load with small limit
    docset_small = DOCSETS["aidanbench"]({"question_id": 0, "max_docs": 5})
    docs_small = docset_small.load_documents()

    # Load with larger limit
    docset_large = DOCSETS["aidanbench"]({"question_id": 0, "max_docs": 20})
    docs_large = docset_large.load_documents()

    # Small should have exactly 5 or fewer
    assert len(docs_small) <= 5, "Should respect max_docs=5"

    # Large should have more than small (assuming enough answers exist)
    # But still respect the limit
    assert len(docs_large) <= 20, "Should respect max_docs=20"

    print(f"\nmax_docs=5: {len(docs_small)} documents")
    print(f"max_docs=20: {len(docs_large)} documents")


def test_aidanbench_with_task_class():
    """Test that AidanBench integrates with Task class."""
    task = Task(
        {
            "document_set": "aidanbench",
            "question_id": 0,
            "criterion": "word_count",
            "max_docs": 10,
            "triplet_style": "random",
        }
    )

    task.load_documents()
    task.create_triplets()

    # Should create documents and triplets
    assert len(task.documents) > 0, "Should load documents"
    assert len(task.documents) <= 10, "Should respect max_docs limit"
    assert len(task.triplets) > 0, "Should create triplets"

    # All triplets should have 3 distinct indices
    for triplet in task.triplets:
        anchor_idx, pos_idx, neg_idx = triplet
        assert anchor_idx != pos_idx, "Anchor and positive should be different"
        assert anchor_idx != neg_idx, "Anchor and negative should be different"
        assert pos_idx != neg_idx, "Positive and negative should be different"

    print(f"\nTask created with {len(task.documents)} documents and {len(task.triplets)} triplets")


def test_aidanbench_random_triplets():
    """Test that AidanBench works with random triplet creation."""
    task = Task(
        {
            "document_set": "aidanbench",
            "question_id": 0,
            "criterion": "word_count",  # Not used for random style
            "max_docs": 20,
            "max_triplets": 10,
            "triplet_style": "random",
        }
    )

    task.load_documents()
    task.create_triplets()

    # Basic assertions
    assert len(task.triplets) == 10, "Should create exactly max_triplets"
    assert all(
        len(triplet) == 3 for triplet in task.triplets
    ), "Each triplet should have 3 elements"

    # Check that each triplet has distinct documents
    for triplet in task.triplets:
        anchor_idx, pos_idx, neg_idx = triplet
        assert anchor_idx != pos_idx != neg_idx, "All indices in triplet should be distinct"

    print(f"\n=== Random triplets test passed ===")
    print(f"Created {len(task.triplets)} random triplets from {len(task.documents)} documents")


def test_aidanbench_multiple_questions_in_benchmark():
    """Test creating tasks for multiple questions in a benchmark."""
    task_configs = [
        {"question_id": 0, "max_docs": 10},
        {"question_id": 1, "max_docs": 10},
    ]

    tasks = []
    for config in task_configs:
        full_config = {
            "document_set": "aidanbench",
            "criterion": "word_count",
            "triplet_style": "random",
            "max_triplets": 5,
            **config,
        }
        task = Task(full_config)
        task.load_documents()
        task.create_triplets()
        tasks.append(task)

    # All tasks should have loaded documents and created triplets
    for i, task in enumerate(tasks):
        assert len(task.documents) > 0, f"Task {i} should have documents"
        assert len(task.triplets) > 0, f"Task {i} should have triplets"

    print(f"\n=== Multiple question tasks test passed ===")
    for i, task in enumerate(tasks):
        print(f"Task {i}: {len(task.documents)} documents, {len(task.triplets)} triplets")


def test_aidanbench_get_document_text():
    """Test that get_document_text works correctly."""
    docset = DOCSETS["aidanbench"]({"question_id": 0, "max_docs": 5})
    docs = docset.load_documents()

    # Test get_document_text on each document
    for doc in docs:
        text = docset.get_document_text(doc)
        assert isinstance(text, str), "get_document_text should return string"
        assert text == doc, "get_document_text should return the document itself"

    print(f"\nget_document_text works correctly for {len(docs)} documents")


def test_aidanbench_error_on_invalid_question_id():
    """Test that invalid question_id raises error."""
    with pytest.raises(ValueError, match="question_id .* out of range"):
        docset = DOCSETS["aidanbench"]({"question_id": 99999})
        docset.load_documents()

    print("\n✓ Correctly raises error for invalid question_id")


def test_aidanbench_error_on_invalid_question_text():
    """Test that invalid question_text raises error."""
    with pytest.raises(ValueError, match="No question found matching"):
        docset = DOCSETS["aidanbench"](
            {"question_text": "THISSTRINGDEFINITELYDOESNOTEXISTINANYQUESTION123456789"}
        )
        docset.load_documents()

    print("\n✓ Correctly raises error for invalid question_text")
