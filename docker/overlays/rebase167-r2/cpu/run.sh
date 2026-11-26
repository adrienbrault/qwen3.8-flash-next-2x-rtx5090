#!/usr/bin/env bash
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EXL3_TREE=${EXL3_TREE:-$(cd "$here/../../exl3" && pwd)}
export TABBY_DIR=${TABBY_DIR:-$(cd "$here/../../tabby" && pwd)}
export PY=${PY:-python3}
export EXL3_SRC=$EXL3_TREE
export PYTHONDONTWRITEBYTECODE=1
bash "$here/engine/run-cpu-tests.sh" "$EXL3_TREE"
"$PY" "$here/engine/tests/test_rebase167.py" -v
bash "$here/tabby/run-cpu-tests.sh"
