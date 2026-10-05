# path: book/projects/aie_core/tests/conftest.py
import pytest

from ._helpers import FakeClock  # noqa: F401


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
