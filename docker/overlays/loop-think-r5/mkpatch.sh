#!/usr/bin/env bash
# Regenerates fix.patch (pristine base/ -> src/) and r4-to-r5.patch (loop-think r4's src/ -> src/) over the files
# src/ carries, as `diff -ruN a b` so both apply with -p1 in /app. Never hand-edit the patches: edit src/ and rerun.
set -euo pipefail
cd "$(dirname "$0")"
files=$(cd src && find . -type f -name '*.py' ! -path '*/__pycache__/*' ! -path '*/.claude/*' | sed 's|^\./||' | sort)
gen() {  # $1 = old tree, $2 = output
  local t; t=$(mktemp -d "${TMPDIR:-/tmp}/loopthink.XXXXXX")
  for f in $files; do
    mkdir -p "$t/a/$(dirname "$f")" "$t/b/$(dirname "$f")"
    cp -p "$1/$f" "$t/a/$f"; cp -p "src/$f" "$t/b/$f"
  done
  (cd "$t" && diff -ruN a b) > "$2" || [ $? -eq 1 ]
  rm -rf "$t"
}
gen base fix.patch
gen ../loop-think-r4/src r4-to-r5.patch
grep -c '^diff ' fix.patch r4-to-r5.patch
