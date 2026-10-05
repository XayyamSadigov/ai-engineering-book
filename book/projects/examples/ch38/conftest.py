# path: book/projects/examples/ch38/conftest.py
"""Puts this directory on sys.path so the chapter modules import from any working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
