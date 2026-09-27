#!/bin/bash
# CPU test suite for the ported exllamav3 tree (out/rebase-dev-r3).
# Usage: tests/run_cpu_tests.sh [python]   (default: .venv/bin/python, else python3)
# No GPU and no extension build: `exllamav3.ext` and `triton` are stubbed by tests/_runner.py,
# and every module import and flag dispatch is exercised against real torch CPU tensors.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${1:-}"
if [ -z "$PY" ]; then
  if [ -x "$HERE/../../../.venv/bin/python" ]; then PY="$HERE/../../../.venv/bin/python"
  else PY=python3; fi
fi
cd "$HERE"
echo "python: $PY ($($PY --version 2>&1))"
echo "tree:   $(dirname "$HERE")"
echo
FAILED=0
for t in test_flag_selection.py test_flag_selection_r2.py test_flag_selection_r3.py test_cache_rewind.py test_e3_det_index.py test_pagetable.py test_tabby_surface.py test_tabby_callsites.py; do
  echo "----- $t -----"
  if "$PY" "$t"; then echo "     [suite OK]"; else echo "     [suite FAILED]"; FAILED=1; fi
  echo
done
echo "===== ALL SUITES: $FAILED (0 = pass) ====="
exit "$FAILED"
