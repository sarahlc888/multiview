"""Tests for Task class and task naming."""

import pytest

from multiview.benchmark.task import Task


def test_get_task_name_with_suffix():
    """Test task name includes config suffix."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        "triplet_style": "lm",
        "use_config_suffix": True,
    }

    task = Task(config)
    task_name = task.get_task_name()

    # Should be: gsm8k__arithmetic__hn__300
    assert task_name == "gsm8k__arithmetic__hn__300"


def test_get_task_name_different_styles():
    """Test different triplet styles produce different names."""
    config1 = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        "triplet_style": "lm",
    }

    config2 = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        "triplet_style": "random",
    }

    task1 = Task(config1)
    task2 = Task(config2)

    # Should have different names
    assert task1.get_task_name() == "gsm8k__arithmetic__hn__300"
    assert task2.get_task_name() == "gsm8k__arithmetic__rnd__300"


def test_get_task_name_different_counts():
    """Test different max_triplets produce different names."""
    config1 = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        "triplet_style": "lm",
    }

    config2 = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 500,
        "triplet_style": "lm",
    }

    task1 = Task(config1)
    task2 = Task(config2)

    # Should have different names
    assert task1.get_task_name() == "gsm8k__arithmetic__hn__300"
    assert task2.get_task_name() == "gsm8k__arithmetic__hn__500"


def test_get_task_name_legacy_mode():
    """Test legacy mode without suffix."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "use_config_suffix": False,
    }

    task = Task(config)
    task_name = task.get_task_name()

    # Should be: gsm8k__arithmetic (no suffix)
    assert task_name == "gsm8k__arithmetic"


def test_get_task_name_lm_all_style():
    """Test lm_all style uses 'hn' abbreviation."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        "triplet_style": "lm_all",
    }

    task = Task(config)
    task_name = task.get_task_name()

    assert task_name == "gsm8k__arithmetic__hn__300"


def test_get_task_name_prelabeled_style():
    """Test prelabeled style uses 'pre' abbreviation."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 200,
        "triplet_style": "prelabeled",
    }

    task = Task(config)
    task_name = task.get_task_name()

    assert task_name == "gsm8k__arithmetic__pre__200"


def test_get_task_name_category_style():
    """Test lm_category style uses 'cat' abbreviation."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        "triplet_style": "lm_category",
    }

    task = Task(config)
    task_name = task.get_task_name()

    assert task_name == "gsm8k__arithmetic__cat__300"


def test_get_task_name_tags_style():
    """Test lm_tags style uses 'tag' abbreviation."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        "triplet_style": "lm_tags",
    }

    task = Task(config)
    task_name = task.get_task_name()

    assert task_name == "gsm8k__arithmetic__tag__300"


def test_get_task_name_default_style():
    """Test default triplet style (should be 'lm')."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 300,
        # No triplet_style specified - should default to "lm"
    }

    task = Task(config)
    task_name = task.get_task_name()

    assert task_name == "gsm8k__arithmetic__hn__300"


def test_get_task_name_zero_triplets():
    """Test with max_triplets=0."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "max_triplets": 0,
        "triplet_style": "lm",
    }

    task = Task(config)
    task_name = task.get_task_name()

    assert task_name == "gsm8k__arithmetic__hn__0"


def test_default_hint_fallback():
    """Test that default_hint is used as fallback for specific hints.

    ArXiv CS criteria use default_hint, so all three hint types should
    have the same value from default_hint.
    """
    config = {
        "document_set": "arxiv_cs",
        "criterion": "research_sensibility",
    }

    task = Task(config)

    # All three hints should have the same value from default_hint
    assert task.category_schema_hint is not None
    assert task.tag_schema_hint is not None
    assert task.summary_hint is not None
    assert task.category_schema_hint == task.tag_schema_hint
    assert task.tag_schema_hint == task.summary_hint
    # Should contain the hint content
    assert "arxiv_research_sensibility.txt" in task.category_schema_hint or len(task.category_schema_hint) > 0


def test_specific_hint_overrides_default():
    """Test that specific hints override default_hint when provided."""
    config = {
        "document_set": "gsm8k",
        "criterion": "arithmetic",
        "category_schema_hint": "Custom category hint",
    }

    task = Task(config)

    # Config override should be used
    assert task.category_schema_hint == "Custom category hint"
