"""Pytest configuration."""

from multiview.utils.logging_utils import setup_logging


def pytest_configure(config):
    """Configure logging for tests."""
    setup_logging(level="INFO")
