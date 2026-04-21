"""Shared pytest fixtures for bench/ragas tests.

Module path resolution lives in ../pyproject.toml (pythonpath = ["."])
rather than a sys.path hack here — so broken packaging fails loudly.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
