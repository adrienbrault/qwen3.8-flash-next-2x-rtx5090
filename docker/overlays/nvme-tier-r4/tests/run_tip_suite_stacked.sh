#!/bin/sh
# Run recurrent-tip-r1's own CPU tests against the stacked tree: src/ with this round's recurrent.py, pagetable.py,
# disk_cache.py and async_generator.py, plus tip r1's overlay. tip's harness resolves src/ as <root>/src/exllamav3
# and its deliverable as <root>/ref/recurrent-tip-r1, so both are laid out in a temporary root.
# generator.py stays pristine in that src/: tip's test_default_path checks tip's generator.py against it.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
DELIV=$(dirname "$HERE")
WS=$(dirname "$(dirname "$DELIV")")
PY=${PY:-$WS/.venv/bin/python}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/src" "$TMP/ref"
cp -R "$WS/src/exllamav3" "$TMP/src/exllamav3"
for f in cache/recurrent.py generator/pagetable.py generator/disk_cache.py generator/async_generator.py; do
  cp "$DELIV/overlay/exllamav3/$f" "$TMP/src/exllamav3/$f"
done
cp -R "$WS/ref/recurrent-tip-r1" "$TMP/ref/recurrent-tip-r1"
cd "$TMP/ref/recurrent-tip-r1/tests"
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q -p no:cacheprovider test_tip_runtime.py test_tip_policy.py "$@"
# test_default_path.py pins tip's manifest "depends" hashes to the pristine src/ (recurrent.py, pagetable.py): run it
# for the record; its depends check is expected to report exactly those two files.
PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q -p no:cacheprovider test_default_path.py || true
