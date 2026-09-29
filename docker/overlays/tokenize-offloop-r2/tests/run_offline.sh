#!/usr/bin/env bash
# Offline (CPU) checks for tokenize-offloop r2, outside Docker:
#   1. scratch trees: the served exllamav3 package (rebase-dev r3 out/) + exl3.patch, and the served /app
#      (served-tree-stack-r3-rows32/tabbyAPI + loop-think r5's src/ files) + app.patch, both --fuzz=0, hashed against
#      SHA256SUMS.*.src
#   2. tests/test_identity.py, tests/test_lock.py, tests/test_gil.py, tests/test_event_loop.py, tests/test_hops.py (synthetic tokenizer + the Qwen3.8-27B
#      tokenizer.json from the HF cache when present)
#   3. the landing check (tests/landing_offline.py: stub package, real /app code)
#   4. the served rebase-dev r3 CPU suite against the patched package, and its TabbyAPI call-site audit against the
#      patched /app (TABBY_BACKENDS / TABBY_EXTRA as rebase-dev r3's Dockerfile runs it, plus the two new/changed files)
# Usage: tests/run_offline.sh [python]   (needs torch (CPU), tokenizers, and TabbyAPI's CPU deps: fastapi, pydantic,
#        loguru, jinja2, ...; work/.venv holds them on the build host, see README.md)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
O=$(dirname "$HERE")
PATCHES=$(cd "$O/../.." && pwd)
PY=${1:-$O/work/.venv/bin/python}
TABBY=${TABBY:-$PATCHES/exllamav3/served-tree-stack-r3-rows32/tabbyAPI}   # /app of the served image chain (IMAGE_ID there)
SERVED=$PATCHES/exllamav3/rebase-dev/r3/out/rebase-dev-r3
QWEN=${QWEN:-$(ls -d "$HOME"/.cache/huggingface/hub/models--Vontra--Qwen3.8-27B-MLX-4bit/snapshots/*/ 2>/dev/null | head -1)}
T=$(mktemp -d "${TMPDIR:-/tmp}/tok-offloop.XXXXXX"); trap 'rm -rf "$T"' EXIT
cp -R "$SERVED/exllamav3" "$T/exllamav3"
patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -d "$T/exllamav3" < "$O/exl3.patch" > /dev/null \
  || { echo "exl3.patch does not apply to the served package"; exit 1; }
mkdir -p "$T/app"
( cd "$TABBY" && tar cf - --exclude .git --exclude '__pycache__' --exclude 'models' --exclude 'loras' . ) | ( cd "$T/app" && tar xf - )
( cd "$PATCHES/tabbyapi/loop-think-r5/src" && tar cf - --exclude '__pycache__' . ) | ( cd "$T/app" && tar xf - )
( cd "$T/app" && sha256sum -c --quiet "$O/SHA256SUMS.app.base" ) || { echo "scratch /app != base/"; exit 1; }
patch -p1 --fuzz=0 --forward --no-backup-if-mismatch -d "$T/app" < "$O/app.patch" > /dev/null \
  || { echo "app.patch does not apply to the served /app"; exit 1; }
( cd "$T/exllamav3" && sha256sum -c --quiet "$O/SHA256SUMS.exl3.src" ) || { echo "patched package != src/"; exit 1; }
( cd "$T/app" && sha256sum -c --quiet "$O/SHA256SUMS.app.src" ) || { echo "patched /app != src/"; exit 1; }
export TOK_EXL3=$T/exllamav3 TOK_APP=$T/app TOK_EXL3_BASE=$O/base/exllamav3/tokenizer/tokenizer.py KMP_WARNINGS=0
FAILED=0
QARGS=(); [ -n "$QWEN" ] && [ -f "$QWEN/tokenizer.json" ] && QARGS=(--tokenizer-dir "$QWEN")
[ ${#QARGS[@]} = 0 ] && echo "NOTE: no Qwen tokenizer in the HF cache: the tests run on the synthetic tokenizer only"
run(){ echo "----- $(basename "$2") -----"; "$@" > "$T/out.txt" 2>&1; local rc=$?
  grep -v 'OMP: Warning' "$T/out.txt" | grep -vE '\| (DEBUG|INFO|ERROR) +\| common\.'; [ "$rc" = 0 ] || FAILED=1; }
( cd "$O" && sha256sum -c --quiet SHA256SUMS.tests ) || { echo "tests/landing differ from SHA256SUMS.tests: run mkpatch.sh"; FAILED=1; }
run "$PY" "$HERE/test_identity.py" "${QARGS[@]}"
run "$PY" "$HERE/test_lock.py" "${QARGS[@]}"
run "$PY" "$HERE/test_gil.py" "${QARGS[@]}"
run "$PY" "$HERE/test_event_loop.py" "${QARGS[@]}"
run "$PY" "$HERE/test_hops.py" "${QARGS[@]}"
run "$PY" "$HERE/landing_offline.py"
cp -R "$SERVED/tests" "$T/tests"
echo "----- served CPU suite (rebase-dev r3 tests/) against the patched package -----"
bash "$T/tests/run_cpu_tests.sh" "$PY" > "$T/suite.txt" 2>&1 || FAILED=1
grep -E 'passed|ALL SUITES' "$T/suite.txt"
( cd "$T/tests" && EXL3_TREE=$T TABBY_BACKENDS=$T/app/backends/exllamav3 \
    TABBY_EXTRA=$T/app/endpoints/OAI/utils/chat_completion.py:$T/app/common/sampling.py:$T/app/common/tokenize_offloop.py:$T/app/endpoints/OAI/router.py \
    "$PY" test_tabby_callsites.py > "$T/callsites.txt" 2>&1 ) || FAILED=1
echo "----- TabbyAPI call-site audit against the patched /app -----"; tail -2 "$T/callsites.txt"
echo "===== tokenize-offloop r2 offline: $([ $FAILED = 0 ] && echo PASS || echo FAIL) ====="
exit $FAILED
