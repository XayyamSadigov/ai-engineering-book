# path: book/projects/reliability/tests/conftest.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `examples` importable

from reliability import ManualClock  # noqa: E402


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()
