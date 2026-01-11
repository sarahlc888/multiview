import json

import pytest

from multiview.benchmark.task import Task
from multiview.benchmark.artifacts import save_task_triplets

pytestmark = pytest.mark.dev


def test_save_triplets_includes_quality_class_when_available(tmp_path):
    task = Task(
        config={
            "document_set": "gsm8k",
            "criterion": "arithmetic",
            "triplet_style": "random",
            "max_docs": 3,
            "max_triplets": 1,
        }
    )

    # Avoid any dataset / LM calls: set minimal internal state directly.
    task.documents = ["a", "b", "c"]
    task.triplets = [(0, 1, 2)]
    task.triplet_quality_ratings = [4]

    save_task_triplets(task, tmp_path)

    triplet_file = tmp_path / task.get_task_name() / "triplets.json"
    content = json.loads(triplet_file.read_text())

    assert isinstance(content, list)
    assert len(content) == 1

    payload = content[0]

    assert payload["quality_rating"] == 4
    assert payload["quality_label"] == "ideal"
    assert payload["quality_class"] == "Ideal"
