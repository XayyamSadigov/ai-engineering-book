# path: book/projects/examples/ch31/tests/conftest.py
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from incident_sim import simulate  # noqa: E402
from join_signals import join  # noqa: E402
from trace_store import TraceStore  # noqa: E402


@pytest.fixture(scope="session")
def sim():
    return simulate(seed=7)


@pytest.fixture()
def store(sim):
    st = TraceStore.from_spans(sim.spans)
    join(st, sim.evals, sim.feedback)
    return st
