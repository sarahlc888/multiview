"""Pytest configuration."""

import pytest

from multiview.utils.logging_utils import setup_logging


def pytest_addoption(parser):
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.external (calls external APIs / network).",
    )


def pytest_configure(config):
    """Configure logging for tests."""
    setup_logging(level="DEBUG")


def pytest_collection_modifyitems(config, items):
    """Skip external tests unless explicitly requested."""
    if config.getoption("--run-external"):
        return

    skip_external = pytest.mark.skip(reason="needs external APIs; run with --run-external")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip_external)
