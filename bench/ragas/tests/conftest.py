"""Shared pytest fixtures for bench/ragas tests."""

import sys
from pathlib import Path

import pytest

# Make the parent bench/ragas directory importable so tests can `from corpus import ...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
