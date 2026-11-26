#!/usr/bin/env bash
# rebase167 NATIVE CPU tests. Run inside the built image with no GPU and no network:
#   sudo docker run --rm --network=none -e CUDA_VISIBLE_DEVICES= --entrypoint bash <image> /opt/rebase167/tools/run-cpu-tests.sh
# (also baked into the build: /opt/rebase167/cpu-tests.txt). Outside the image, point REBASE167_PKG at an exllamav3/
# package directory; tests that need torch / the package skip when it does not import.
# Uses pytest when importable, else tests/_minipytest.py (the served image has no pytest).
# REBASE167_NO_SKIP=1 (the image build sets it): every skip counts as a failure (always runs _minipytest.py).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=${REBASE167_TESTS:-}
if [ -z "$T" ]; then
  if [ -d "$HERE/tests" ]; then T="$HERE/tests"; else T=/opt/rebase167/tests; fi
fi
PY=${PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
cd /tmp 2>/dev/null || cd "$T"
echo "python: $PY ($($PY --version 2>&1)); tests: $T"
FILES=$(ls "$T"/test_*.py)
if [ "${REBASE167_NO_SKIP:-0}" != 1 ] && $PY -c 'import pytest' 2>/dev/null; then
  $PY -m pytest -q -p no:cacheprovider $FILES
else
  $PY "$T/_minipytest.py" $FILES
fi
rc=$?
echo "===== NATIVE CPU TESTS rc=$rc (0 = pass) ====="
exit $rc
