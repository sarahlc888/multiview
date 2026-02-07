from types import SimpleNamespace

from multiview.benchmark.artifacts import _extract_triplet_generation_config


def test_triplet_config_uses_resolved_task_criterion_description():
    task = SimpleNamespace(
        config={
            "document_set": "ut_zappos50k",
            "criterion": "functional_type",
            "triplet_style": "lm_tags",
            "max_triplets": 5,
            # Intentionally omitted to simulate metadata-derived description.
        },
        criterion_description="The functional type of the shoe.",
    )

    extracted = _extract_triplet_generation_config(task)

    assert extracted["criterion_description"] == "The functional type of the shoe."
