# path: book/projects/examples/ch05/conftest.py
"""Puts this directory on sys.path so `import context` works from any working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
