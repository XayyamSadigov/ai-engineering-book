#!/usr/bin/env bash
# Compile every Python file in the book and run every test suite offline.
# Usage: book/tools/verify_code.sh [extra pytest args]
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

echo "== py_compile =="
fail=0
while IFS= read -r f; do
  "$PY" -m py_compile "$f" 2>>/tmp/aie_compile_errors.txt || { echo "COMPILE FAIL: $f"; fail=1; }
done < <(find book/projects book/capstone -name '*.py' -not -path '*/.venv/*' -not -path '*/node_modules/*' 2>/dev/null)
[ $fail -eq 0 ] && echo "all python files compile"

echo
echo "== pytest per project =="
summary=()
while IFS= read -r d; do
  if find "$d" -maxdepth 3 -name 'test_*.py' -o -maxdepth 3 -name '*_test.py' | grep -q .; then
    echo "--- $d"
    out=$("$PY" -m pytest "$d" -q -p no:cacheprovider -x --timeout=300 "$@" 2>&1 | tail -3) || true
    echo "$out"
    summary+=("$d :: $(echo "$out" | tail -1)")
  fi
done < <( { find book/projects -mindepth 1 -maxdepth 1 -type d ! -name examples; find book/projects/examples -mindepth 1 -maxdepth 1 -type d -name "ch*"; [ -d book/capstone ] && find book/capstone -mindepth 1 -maxdepth 1 -type d; } 2>/dev/null | sort )

echo
echo "== summary =="
printf '%s\n' "${summary[@]}"
