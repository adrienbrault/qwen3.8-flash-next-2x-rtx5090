#!/usr/bin/env bash
# hcfast-r2.patch must apply at --fuzz=0 on the pristine slotfix-r1 exllamav3 tree alone, after
# moefast-r1.patch, and before it (both touch exllamav3_ext/bindings.cpp, in disjoint regions).
#   bash check-composition.sh <pristine exllamav3 dir> <moefast-r1.patch>
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SRC=${1:?usage: check-composition.sh <pristine exllamav3 dir> [moefast-r1.patch]}
MF=${2:-$here/../moefast-r1/moefast-r1.patch}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
echo "# patch composition $(date -u +%FT%TZ); pristine = $(basename "$SRC") tree; patch -p1 --fuzz=0"
fail=0
base=$(find "$SRC" -name '*.rej' -o -name '*.orig' | wc -l | tr -d ' ')   # the served tree ships model_ls.py.orig
for order in "hcfast" "moefast hcfast" "hcfast moefast"; do
  rm -rf "$T/pkg"; cp -R "$SRC" "$T/pkg"
  echo "== order: $order"
  for p in $order; do
    if [ "$p" = moefast ]; then f=$MF; else f=$here/hcfast-r2.patch; fi
    out=$(patch -p1 --fuzz=0 --batch --forward --dry-run -d "$T/pkg" < "$f" 2>&1); rc=$?
    echo "-- $p dry run rc=$rc"; echo "$out" | sed 's/^/   /'
    [ $rc = 0 ] || fail=1
    patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -s -d "$T/pkg" < "$f" || fail=1
  done
  n=$(find "$T/pkg" -name '*.rej' -o -name '*.orig' | wc -l | tr -d ' ')
  echo "-- .rej/.orig files: $n (pristine tree has $base)"; [ "$n" = "$base" ] || fail=1
  grep -n 'hc_mix_v3\|moe_coop_v3' "$T/pkg/exllamav3_ext/bindings.cpp" | sed 's/^/   bindings.cpp:/'
  echo "   bindings.cpp md5 $( (md5sum "$T/pkg/exllamav3_ext/bindings.cpp" 2>/dev/null || md5 -r "$T/pkg/exllamav3_ext/bindings.cpp") | cut -d' ' -f1)"
done
[ $fail = 0 ] && echo "COMPOSITION OK" || { echo "COMPOSITION FAILED"; exit 1; }
