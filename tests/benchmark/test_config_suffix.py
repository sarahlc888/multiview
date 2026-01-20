"""Tests for config suffix generation."""

from multiview.benchmark.task import _make_config_suffix


def test_make_config_suffix_lm():
    """Test suffix for lm triplet style (treated as hard_negative)."""
    config = {
        "triplet_style": "lm",
        "max_triplets": 300,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "hn__300"


def test_make_config_suffix_lm_all():
    """Test suffix for lm_all triplet style."""
    config = {
        "triplet_style": "lm_all",
        "max_triplets": 300,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "hn__300"


def test_make_config_suffix_random():
    """Test suffix for random triplet style."""
    config = {
        "triplet_style": "random",
        "max_triplets": 500,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "rnd__500"


def test_make_config_suffix_prelabeled():
    """Test suffix for prelabeled triplet style."""
    config = {
        "triplet_style": "prelabeled",
        "max_triplets": 200,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "pre__200"


def test_make_config_suffix_category():
    """Test suffix for lm_category triplet style."""
    config = {
        "triplet_style": "lm_category",
        "max_triplets": 300,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "cat__300"


def test_make_config_suffix_tags():
    """Test suffix for lm_tags triplet style."""
    config = {
        "triplet_style": "lm_tags",
        "max_triplets": 400,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "tag__400"


def test_make_config_suffix_summary_dict():
    """Test suffix for lm_summary_dict triplet style."""
    config = {
        "triplet_style": "lm_summary_dict",
        "max_triplets": 300,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "sdict__300"


def test_make_config_suffix_summary_sentence():
    """Test suffix for lm_summary_sentence triplet style."""
    config = {
        "triplet_style": "lm_summary_sentence",
        "max_triplets": 300,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "ssent__300"


def test_make_config_suffix_unknown_style():
    """Test suffix for unknown triplet style (should truncate to first 4 chars)."""
    config = {
        "triplet_style": "custom_strategy",
        "max_triplets": 100,
    }

    suffix = _make_config_suffix(config)
    # Unknown styles use first 4 chars
    assert suffix == "cust__100"


def test_make_config_suffix_default_style():
    """Test suffix with default triplet style (should be 'lm')."""
    config = {
        "max_triplets": 300,
        # No triplet_style - should default to "lm"
    }

    suffix = _make_config_suffix(config)
    assert suffix == "hn__300"


def test_make_config_suffix_default_max_triplets():
    """Test suffix with default max_triplets (should be 0)."""
    config = {
        "triplet_style": "lm",
        # No max_triplets - should default to 0
    }

    suffix = _make_config_suffix(config)
    assert suffix == "hn__0"


def test_make_config_suffix_large_count():
    """Test suffix with large triplet count."""
    config = {
        "triplet_style": "lm",
        "max_triplets": 10000,
    }

    suffix = _make_config_suffix(config)
    assert suffix == "hn__10000"
