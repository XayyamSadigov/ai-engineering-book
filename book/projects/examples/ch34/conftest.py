# path: book/projects/examples/ch34/conftest.py
"""Registers the ``integration`` marker and skips those tests unless ``--run-integration`` is given."""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-integration", action="store_true", default=False, help="run tests needing real services")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: needs a real provider, engine, or aie_core install")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    skip = pytest.mark.skip(reason="integration test; pass --run-integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
