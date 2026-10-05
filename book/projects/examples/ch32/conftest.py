# path: book/projects/examples/ch32/conftest.py
"""Puts this directory on sys.path so `import northwind_triage` works from any working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
