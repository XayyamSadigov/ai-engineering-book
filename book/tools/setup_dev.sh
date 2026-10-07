#!/usr/bin/env bash
# Create the repo-root virtual environment with every book library and project installed editable.
# Usage: book/tools/setup_dev.sh                       (creates ./.venv with Python 3.12)
#        PYTHON=3.11 VENV=/tmp/aie-venv book/tools/setup_dev.sh
#
# Requires uv (https://docs.astral.sh/uv/). Do not use plain `pip install -e .` on the projects:
# pip ignores [tool.uv.sources], so it cannot find the sibling libraries, and several library names
# (ragkit, evalkit, toolkit, agentkit, guardrails, reliability, memorykit) belong to unrelated
# packages on PyPI that pip would install instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || { echo "uv not found: install it from https://docs.astral.sh/uv/" >&2; exit 1; }

VENV="${VENV:-.venv}"
uv venv --python "${PYTHON:-3.12}" "$VENV"

args=()
for d in book/projects/aie_core book/projects/shared-data \
         book/projects/{toolkit,ragkit,evalkit,guardrails,reliability,agentkit,memorykit} \
         book/projects/p{1,2,3,4,5,6}-* book/capstone/northwind-assist; do
  [ -f "$d/pyproject.toml" ] || continue
  if grep -q '^dev = \[' "$d/pyproject.toml"; then args+=(-e "$d[dev]"); else args+=(-e "$d"); fi
done

# Extra packages the per-chapter examples import (they run from their own folders, uninstalled).
uv pip install --python "$VENV" "${args[@]}" \
  "numpy>=1.26" "scikit-learn>=1.4" "jinja2>=3.1" "pyyaml>=6" "opentelemetry-sdk>=1.24" \
  "fastapi>=0.110" "httpx>=0.25" "fakeredis>=2.20" "lupa>=2" "redis>=5" "sqlglot>=25" "tiktoken>=0.7" "pytest>=8" "pytest-timeout>=2.2" "hypothesis>=6"

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Run one project:      cd book/projects/p3-rag-assistant && pytest -q"
echo "Run everything:       book/tools/verify_code.sh"
