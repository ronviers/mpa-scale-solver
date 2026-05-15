"""Shared pytest fixtures for mpa-scale-solver tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def seed_corpus_root(repo_root: Path) -> Path:
    """Three v2.0 driver-profiles live in mpa-conform's seed corpus."""
    return repo_root.parent / "mpa-conform" / "output" / "seed-corpus"
