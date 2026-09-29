#!/usr/bin/env bash
# Offline (CPU) checks for prefill-merge r1, outside Docker:
#   1. a scratch copy of the served tree (docker/overlays/rebase-dev-r3/out/rebase-dev-r3, which
#      docker/overlays/rebase-dev-r3/prepare-tree.sh produces) + fix.patch (--fuzz=0)
#   2. this directory's tests against it (PM_TREE)
#   3. the served CPU suite (rebase-dev r3 tests/, ext + triton stubbed) against the PATCHED tree: with both knobs
#      unset the served flag selection, rewind, pagetable and TabbyAPI call-site checks must still pass
# Usage: tests/run_offline.sh [python]   (needs torch (CPU), pydantic, safetensors, tokenizers; see README.md)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PM=$(dirname "$HERE")
SERVED_ROOT=${SERVED_ROOT:-$PM/../rebase-dev-r3/out/rebase-dev-r3}
[ -d "$SERVED_ROOT/exllamav3" ] || { echo "no served tree at $SERVED_ROOT: run docker/overlays/rebase-dev-r3/prepare-tree.sh first"; exit 1; }
PY=${1:-python3}
T=$(mktemp -d "${TMPDIR:-/tmp}/pm-r1.XXXXXX"); trap 'rm -rf "$T"' EXIT
cp -R "$SERVED_ROOT/exllamav3" "$T/exllamav3"
cp -R "$SERVED_ROOT/tests" "$T/tests"
patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -d "$T/exllamav3" < "$PM/fix.patch" > /dev/null \
  || { echo "fix.patch does not apply to the served tree"; exit 1; }
FAILED=0
for t in test_partition.py test_split_equiv.py test_forward_wiring.py test_gpu_gate_logic.py; do
  echo "----- $t -----"
  PM_TREE=$T "$PY" "$HERE/$t" 2>&1 | grep -v 'OMP: Warning' || true
  PM_TREE=$T "$PY" "$HERE/$t" > /dev/null 2>&1 || FAILED=1
done
echo "----- served CPU suite (rebase-dev r3 tests/) against the patched tree -----"
bash "$T/tests/run_cpu_tests.sh" "$PY" 2>&1 | grep -E 'passed|suite|ALL SUITES' | grep -v 'OMP: Warning'
bash "$T/tests/run_cpu_tests.sh" "$PY" > /dev/null 2>&1 || FAILED=1
echo "===== prefill-merge r1 offline: $([ $FAILED = 0 ] && echo PASS || echo FAIL) ====="
exit $FAILED
