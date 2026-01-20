import json

import pytest

from multiview.benchmark.task import Task
from multiview.benchmark.artifacts import save_task_triplets

pytestmark = pytest.mark.dev


def test_save_triplets_includes_annotations_when_available(tmp_path):
    """Test that annotations are included in triplet output when they exist."""
    task = Task(
        config={
            "document_set": "gsm8k",
            "criterion": "arithmetic",
            "triplet_style": "random",
            "max_docs": 3,
            "max_triplets": 1,
        }
    )

    # Set minimal internal state directly to avoid dataset/LM calls
    task.documents = ["doc a", "doc b", "doc c"]
    task.triplets = [(0, 1, 2)]
    task.document_annotations = [
        {"category": "TypeA", "tags": {"tag1": True}},
        {"category": "TypeB", "tags": {"tag2": False}},
        {"category": "TypeC", "tags": {"tag3": True}},
    ]

    save_task_triplets(task, tmp_path)

    triplet_file = tmp_path / task.get_task_name() / "triplets.json"
    content = json.loads(triplet_file.read_text())

    # Should be an array with one triplet
    assert isinstance(content, list)
    assert len(content) == 1

    payload = content[0]

    # Verify annotations are included
    assert "anchor_annotation" in payload
    assert "positive_annotation" in payload
    assert "negative_annotation" in payload

    assert payload["anchor_annotation"]["category"] == "TypeA"
    assert payload["positive_annotation"]["category"] == "TypeB"
    assert payload["negative_annotation"]["category"] == "TypeC"


def test_save_triplets_without_annotations(tmp_path):
    """Test that triplets work fine without annotations."""
    task = Task(
        config={
            "document_set": "gsm8k",
            "criterion": "arithmetic",
            "triplet_style": "random",
            "max_docs": 3,
            "max_triplets": 1,
        }
    )

    # Set minimal internal state, no annotations
    task.documents = ["doc a", "doc b", "doc c"]
    task.triplets = [(0, 1, 2)]
    task.document_annotations = None

    save_task_triplets(task, tmp_path)

    triplet_file = tmp_path / task.get_task_name() / "triplets.json"
    content = json.loads(triplet_file.read_text())

    # Should be an array with one triplet
    assert isinstance(content, list)
    assert len(content) == 1

    payload = content[0]

    # Verify annotations are NOT included
    assert "anchor_annotation" not in payload
    assert "positive_annotation" not in payload
    assert "negative_annotation" not in payload
