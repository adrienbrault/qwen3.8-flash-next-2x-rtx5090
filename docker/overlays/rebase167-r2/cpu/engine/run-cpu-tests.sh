#!/bin/bash
# rebase165 PY CPU tests. No GPU, no network, no pytest needed. Inside the built image:
#   docker run --rm --network=none -v <W>/out/py:/py-tests:ro -v <exllamav3 checkout>:/src:ro --entrypoint bash <image> \
#       /py-tests/run-cpu-tests.sh /src
# Argument / EXL3_SRC: the exllamav3 checkout to test (the directory holding the exllamav3 package). Default: the
# directory holding the installed exllamav3 package, else ../wt-py next to out/.
# EXL3_STUB_EXT=1 forces the no-op extension stub for the two package-import suites (default: real extension when
# `import exllamav3` works, else the stub). PY selects the interpreter (default python3).
# Every suite runs in its own process. Exit 0 = all suites passed (skips count as passes; the log says why).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
export PYTHONDONTWRITEBYTECODE=1
if [ $# -ge 1 ]; then EXL3_SRC="$1"; fi
if [ -z "${EXL3_SRC:-}" ]; then
  EXL3_SRC="$(cd /tmp && "$PY" -c 'import importlib.util as u, os; s = u.find_spec("exllamav3"); print(os.path.dirname(os.path.dirname(s.origin)) if s else "")' 2>/dev/null)"
  [ -n "$EXL3_SRC" ] || EXL3_SRC="$(cd "$HERE/../.." && pwd)/wt-py"
fi
export EXL3_SRC
echo "python: $PY ($("$PY" --version 2>&1)); torch: $("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo absent)"
echo "exllamav3 under test: $EXL3_SRC ($(git -C "$EXL3_SRC" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo
FAILED=0
cd "$HERE/tests" || exit 2
for t in test_maskfix test_checkpoint_policy test_generator_policy test_tokenizer_offloop \
         test_recurrent_pairing test_generator_construct test_prefill_merge_native; do
  echo "----- $t -----"
  if "$PY" -B -W ignore "$t.py" -v 2>&1; then echo "     [suite OK]"; else echo "     [suite FAILED]"; FAILED=1; fi
  echo
done
[ $FAILED = 0 ] && echo "== PY CPU tests: PASS ==" || echo "== PY CPU tests: FAIL =="
exit $FAILED
