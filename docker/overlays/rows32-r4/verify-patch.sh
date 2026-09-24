#!/usr/bin/env bash
# CPU-only: rows32-r4.patch against reconstructed stack-r3 trees, by exit code, at --fuzz=0 (no GPU, no docker).
#   the target   src + hcfast-r1 + moefast-r1 (= stack-r2) + stack-r3 series INCLUDE="mf3 dg2" (= tabbyapi:stack-r3)
#   also         INCLUDE "mf3 dg1" applies too (06-dense-lcguard is the same with dg1); the gate and the landing refuse a
#                dg1 base anyway (densegemm-r1's rows32 twins spill, R710)
#   negatives    INCLUDE "dg2" / "dg1" (no moefast-r3: the start_shared hunk has no context; a stack-r3 without MF3
#                needs a re-diff), bare stack-r2 (no densegemm, no lcguard)
# Then: the patched target tree passes test_rows32_cpu.py --tree (parts 1-3, 5) and stack-r3's lcguard CPU test.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# SRC (required) = the installed exllamav3 package of tabbyapi:slotfix-r1; FLAN = docker/overlays (hcfast-r1/, moefast-r1/,
# stack-r3/); R3 = rows32-r3.patch for the r3-lines check, skipped when unset (rows32 r1-r3 are not in this repository).
SRC=${SRC:?SRC = the installed exllamav3 package of tabbyapi:slotfix-r1}
FLAN=${FLAN:-$(cd "$here/.." && pwd)}
R3=${R3:-}
S3=${S3:-$FLAN/stack-r3}
P=$here/rows32-r4.patch
tmp=$(mktemp -d "${TMPDIR:-/tmp}/rows32-r4-verify.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
ap(){ sed 's/^$/ /' "$2" | patch -s -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$1" > "$tmp/ap.log" 2>&1; }
clean(){ find "$1" \( -name '*.orig' -o -name '*.rej' \) ! -path '*/model/model_ls.py.orig' -delete
         find "$1" -name __pycache__ -type d -prune -exec rm -rf {} +; }
cp -R "$SRC" "$tmp/r2"; clean "$tmp/r2"
ap "$tmp/r2" "$FLAN/hcfast-r1/hcfast-r1.patch" && ap "$tmp/r2" "$FLAN/moefast-r1/moefast-r1.patch" || { echo "cannot rebuild stack-r2"; exit 2; }
clean "$tmp/r2"
echo "# rows32-r4.patch sha256 $(shasum -a 256 "$P" 2>/dev/null | cut -c1-64 || sha256sum "$P" | cut -c1-64); $(patch --version 2>&1 | head -1)"
rc=0
tree(){ local inc=$1 d=$2; rm -rf "$d"; cp -R "$tmp/r2" "$d"
  [ "$inc" = none ] || INCLUDE="$inc" bash "$S3/series/apply-series.sh" "$d" > "$tmp/s.log" 2>&1 || { cat "$tmp/s.log"; return 1; }
  clean "$d"; }
try(){ local inc=$1 want=$2 d=$tmp/t; tree "$inc" "$d" || { echo "  FAIL  INCLUDE='$inc': series does not apply"; rc=1; return; }
  local dry=0 real=0
  sed 's/^$/ /' "$P" | patch -s -p1 --fuzz=0 --batch --forward --dry-run -d "$d" > "$tmp/dry.log" 2>&1 || dry=1
  if [ $dry = 0 ]; then sed 's/^$/ /' "$P" | patch -s -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$d" > "$tmp/real.log" 2>&1 || real=1; fi
  find "$d" -name '*.rej' | grep -q . && real=1
  local got=OK; [ $dry = 0 ] && [ $real = 0 ] || got=REJECT
  if [ "$got" = "$want" ]; then echo "  ok    INCLUDE='$inc': $got (expected)"; else echo "  FAIL  INCLUDE='$inc': $got (expected $want)"; rc=1; fi
  [ "$got" = OK ] && [ "$inc" = "mf3 dg2" ] && { clean "$d"; cp -R "$d" "$tmp/target"; }; }
try "mf3 dg2" OK
try "mf3 dg1" OK
try "dg2" REJECT
try "dg1" REJECT
try none REJECT
if [ -d "$tmp/target" ]; then
  python3 "$here/test_rows32_cpu.py" --tree "$tmp/target" > "$tmp/cpu.log" 2>&1 && echo "  ok    test_rows32_cpu.py --tree <target>: $(tail -1 "$tmp/cpu.log")" \
    || { echo "  FAIL  test_rows32_cpu.py:"; tail -5 "$tmp/cpu.log"; rc=1; }
  python3 "$S3/tests/test_dense_lcguard_cpu.py" --tree "$tmp/target" > "$tmp/lcg.log" 2>&1 && echo "  ok    stack-r3 test_dense_lcguard_cpu.py --tree <target>: $(tail -1 "$tmp/lcg.log")" \
    || { echo "  FAIL  lcguard CPU test:"; grep FAIL "$tmp/lcg.log" | head; rc=1; }
  # the r4 patch is r3's hunks + the r4 additions: every r3 added line is in r4 (only with R3 set)
  if [ -n "$R3" ] && [ -f "$R3" ]; then
  miss=$(grep '^+[^+]' "$R3" | sed 's/rows32 r3/rows32 rX/g' | sort -u > "$tmp/r3.add"; \
         grep '^+[^+]' "$P" | sed 's/rows32 r[34]/rows32 rX/g' | sort -u > "$tmp/r4.add"; comm -23 "$tmp/r3.add" "$tmp/r4.add" | wc -l)
  echo "  info  r3 added lines not in r4 (revision / message bumps only expected): $(echo $miss)"
  comm -23 "$tmp/r3.add" "$tmp/r4.add" | sed 's/^/          /' | head -8
  else echo "  info  R3 unset: r3-lines check skipped"; fi
fi
[ $rc = 0 ] && echo "VERIFY OK" || { echo "VERIFY FAILED"; exit 1; }
