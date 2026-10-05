# path: book/projects/examples/ch08/conftest.py
"""Makes `embedlab` importable when pytest runs from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: needs a real embedding provider; skipped by default")
