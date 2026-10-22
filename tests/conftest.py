"""Shared pytest fixtures for the Growize Vision assessment build."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def demo_dir() -> Path:
    return DEMO


@pytest.fixture(scope="session")
def service():
    from growize_vision.inference import InferenceService

    return InferenceService()
