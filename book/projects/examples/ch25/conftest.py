# path: book/projects/examples/ch25/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "ci", ROOT.parents[1] / "shared-data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest_plugins = ["pytest_evalplugin"]
