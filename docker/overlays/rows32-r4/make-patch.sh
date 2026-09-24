#!/usr/bin/env bash
# rows32 r4: regenerate rows32-r4.patch against the stack-r3 tree (tabbyapi:stack-r3, INCLUDE="mf3 dg2").
#
# work/a/exllamav3 = the stack-r3 package, rebuilt exactly as stack-r3/series/check-subsets.sh rebuilds it:
#   src/exllamav3 (the pristine served package of tabbyapi:slotfix-r1) + hcfast-r1.patch + moefast-r1.patch at fuzz 0
#   (= stack-r2, the daily's package) + stack-r3/series/apply-series.sh with INCLUDE (sha256-pinned series, fuzz 0).
# work/b/exllamav3 = work/a + the rows32 r4 edits (made in place; --init seeds it from work/a + rows32-r3.patch).
#
#   make-patch.sh --init     rebuild work/a, seed work/b = work/a + rows32-r3.patch (fuzz 0); edit work/b afterwards
#   make-patch.sh            rebuild work/a, diff work/a work/b -> rows32-r4.patch, then round-trip: a fresh copy of
#                            work/a + the patch at fuzz 0 must equal work/b byte for byte
# Paths inside the patch are a/<package path> / b/<package path>:
#   patch -p1 --fuzz=0 -d <site-packages>/exllamav3 < rows32-r4.patch
# Env: SRC (required: the installed exllamav3 package of tabbyapi:slotfix-r1, copied out of the image), FLAN
#      (default docker/overlays, two levels above this file: holds hcfast-r1/, moefast-r1/ and stack-r3/), INCLUDE
#      (default "mf3 dg2"), R3 (rows32-r3.patch, for --init only; rows32 r1-r3 are not in this repository).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SRC=${SRC:?SRC = the installed exllamav3 package of tabbyapi:slotfix-r1}
FLAN=${FLAN:-$(cd "$here/.." && pwd)}
S3=${S3:-$FLAN/stack-r3}
INCLUDE=${INCLUDE:-mf3 dg2}
R3=${R3:-}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/rows32-r4.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
ap(){ sed 's/^$/ /' "$2" | patch -s -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$1" > "$tmp/ap.log" 2>&1 \
        || { cat "$tmp/ap.log"; echo "make-patch: $(basename "$2") does not apply at fuzz 0"; exit 1; }; }
clean(){ find "$1" \( -name '*.orig' -o -name '*.rej' \) ! -path '*/model/model_ls.py.orig' -delete
         find "$1" -name __pycache__ -type d -prune -exec rm -rf {} +; }

build_a(){   # $1 = destination package dir
  rm -rf "$1"; mkdir -p "$(dirname "$1")"; cp -R "$SRC" "$1"; clean "$1"
  ap "$1" "$FLAN/hcfast-r1/hcfast-r1.patch"
  ap "$1" "$FLAN/moefast-r1/moefast-r1.patch"
  clean "$1"
  INCLUDE="$INCLUDE" bash "$S3/series/apply-series.sh" "$1" > "$tmp/series.log" 2>&1 \
    || { cat "$tmp/series.log"; echo "make-patch: stack-r3 series does not apply"; exit 1; }
  clean "$1"
}

build_a "$here/work/a/exllamav3"
echo "make-patch: work/a = stack-r3 (src + hcfast-r1 + moefast-r1 + series INCLUDE='$INCLUDE'): $(grep -c 'applied' "$tmp/series.log") series patches"

if [ "${1:-}" = --init ]; then
  [ -f "$R3" ] || { echo "make-patch: --init needs R3=<rows32-r3.patch>"; exit 1; }
  rm -rf "$here/work/b"; mkdir -p "$here/work/b"; cp -R "$here/work/a/exllamav3" "$here/work/b/exllamav3"
  ap "$here/work/b/exllamav3" "$R3"; clean "$here/work/b/exllamav3"
  echo "make-patch: work/b seeded = work/a + $(basename "$R3") (fuzz 0); edit work/b/exllamav3, then run make-patch.sh"
  exit 0
fi
[ -d "$here/work/b/exllamav3" ] || { echo "make-patch: work/b missing (run --init first)"; exit 1; }
clean "$here/work/b/exllamav3"

mkdir -p "$here/.diffroot"
ln -sfn ../work/a/exllamav3 "$here/.diffroot/a"
ln -sfn ../work/b/exllamav3 "$here/.diffroot/b"
rc=0; ( cd "$here/.diffroot" && command diff -ruN -x __pycache__ -x '*.orig' -x '*.rej' a/ b/ ) > "$tmp/p" || rc=$?
[ "$rc" = 1 ] || { echo "make-patch: diff rc=$rc (expected 1 = differences found)"; exit 1; }
# drop the timestamps from the headers: the patch must not change when nothing but mtimes changed
sed -E 's/^(---|\+\+\+) ([^	]+)	.*/\1 \2/' "$tmp/p" > "$here/rows32-r4.patch"
rm -rf "$here/.diffroot"

build_a "$tmp/rt"
patch -s -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$tmp/rt" < "$here/rows32-r4.patch"
clean "$tmp/rt"
command diff -r -x __pycache__ "$tmp/rt" "$here/work/b/exllamav3" > /dev/null \
  || { echo "make-patch: ROUND TRIP FAILED"; exit 1; }
echo "make-patch: rows32-r4.patch round trip OK ($(grep -c '^diff ' "$here/rows32-r4.patch") files, sha256 $(shasum -a 256 "$here/rows32-r4.patch" 2>/dev/null | cut -c1-16 || sha256sum "$here/rows32-r4.patch" | cut -c1-16))"
