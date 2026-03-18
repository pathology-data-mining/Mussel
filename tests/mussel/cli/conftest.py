"""
Shared pytest configuration and fixtures for CLI tests.

Fixtures from tests/conftest.py are automatically available here via pytest's
conftest discovery — no explicit imports needed.
"""

import pytest


@pytest.fixture
def test_data_path_cli(test_data_path: str) -> str:
    """Return path to test data directory (alias for CLI tests)."""
    return test_data_path
