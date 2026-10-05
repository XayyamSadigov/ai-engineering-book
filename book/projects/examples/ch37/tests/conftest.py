# path: book/projects/examples/ch37/tests/conftest.py
"""Puts the chapter directory on sys.path so the example modules import from any working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
