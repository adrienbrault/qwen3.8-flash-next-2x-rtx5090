#!/usr/bin/env bash
# stack-r3 series applier. Applies the patches listed in SERIES (same directory), in order, to an exllamav3 package
# directory that holds the stack-r2 sources (slotfix-r1 + hcfast-r1 + moefast-r1: the installed package of
# tabbyapi:stack-r2, or ../src/exllamav3 + hcfast-r1.patch + moefast-r1.patch).
#
#   INCLUDE="mf3 dg1" apply-series.sh <exllamav3 dir>
#   INCLUDE=... apply-series.sh --list          print the patches that INCLUDE selects, apply nothing
#
# INCLUDE names the optional components, space- or comma-separated:
#   mf3   moefast-r3 (EXL3_MOE_COOP_V3=3, EXL3_SHARED_EXPERT_EARLY / _PRIO)
#   dg1   densegemm-r1 (EXL3_DENSE_V2, gemv twin = mode 3; its gemm/mgemm twins spill: R710)
#   dg2   densegemm-r2 (same gemv twin, fixed gemm/mgemm twins; needs R714's DGV2_NVCC_DEFS at build time)
# dg1 and dg2 exclude each other. 06-dense-lcguard is applied automatically with either (class auto:dg).
# The core (hcfast r1 -> r2, latchain-r1b) is always applied.
#
# Per patch: the sha256 must match SERIES; empty lines are restored to blank context (" ", a no-op for patches
# stored with the space); a --dry-run at --fuzz=0 must succeed, then the real apply at --fuzz=0; any new .rej fails;
# new .orig files are removed (the served tree ships model/model_ls.py.orig, which is left alone). Exit 0 = every
# selected patch applied cleanly. Works with GNU patch (flan) and BSD/Apple patch.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
series=$here/SERIES
sha(){ if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }
die(){ echo "apply-series: $*" >&2; exit 1; }

inc=" $(echo "${INCLUDE:-}" | tr ',' ' ' | tr -s ' ') "
for w in $inc; do case $w in mf3|dg1|dg2) ;; *) die "unknown INCLUDE component '$w' (known: mf3 dg1 dg2)";; esac; done
has(){ case "$inc" in *" $1 "*) return 0;; *) return 1;; esac; }
if has dg1 && has dg2; then die "dg1 and dg2 exclude each other"; fi
selected(){   # $1 = class column
  case $1 in
    core) return 0 ;;
    opt:*) has "${1#opt:}" ;;
    auto:dg) has dg1 || has dg2 ;;
    *) die "SERIES: unknown class '$1'" ;;
  esac; }

[ -f "$series" ] || die "missing $series"
list=()
while read -r order file want class _; do
  case "$order" in ''|\#*) continue ;; esac
  if selected "$class"; then list+=("$file:$want"); fi
done < "$series"
[ ${#list[@]} -ge 2 ] || die "SERIES selects fewer than the 2 core patches"

if [ "${1:-}" = --list ]; then for e in "${list[@]}"; do echo "${e%%:*}"; done; exit 0; fi
pkg=${1:?usage: INCLUDE=... apply-series.sh <exllamav3 dir> | --list}
[ -d "$pkg/exllamav3_ext" ] || die "$pkg is not an exllamav3 package dir"

# the tree must be stack-r2: hcfast r1 + moefast r1 present, nothing of this series yet
b=$pkg/exllamav3_ext/bindings.cpp
grep -q 'm.attr("hc_mix_v3_revision") = 1;' "$b" || die "base is not stack-r2 (hc_mix_v3_revision 1 missing: hcfast r1 absent or already r2)"
grep -q 'm.attr("moe_coop_v3_revision") = 1;' "$b" || die "base is not stack-r2 (moe_coop_v3_revision 1 missing)"
if grep -qE 'latchain_revision|dense_v2_revision' "$b"; then die "base already carries latchain or densegemm"; fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/stack-r3.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
find "$pkg" \( -name '*.orig' -o -name '*.rej' \) | sort > "$tmp/pre"
grep -q '\.rej$' "$tmp/pre" && die "base tree already has .rej files"
for e in "${list[@]}"; do
  f=${e%%:*}; want=${e#*:}
  [ -f "$here/$f" ] || die "missing $f"
  got=$(sha "$here/$f"); [ "$got" = "$want" ] || die "$f sha256 $got != SERIES $want"
  sed 's/^$/ /' "$here/$f" > "$tmp/p"
  patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$tmp/p" > "$tmp/dry" 2>&1 \
    || { sed 's/^/  /' "$tmp/dry" >&2; die "$f: dry run failed at fuzz 0"; }
  patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$tmp/p" > "$tmp/out" 2>&1 \
    || { sed 's/^/  /' "$tmp/out" >&2; die "$f: apply failed at fuzz 0"; }
  find "$pkg" \( -name '*.orig' -o -name '*.rej' \) | sort > "$tmp/post"
  new=$(comm -13 "$tmp/pre" "$tmp/post" || true)
  if echo "$new" | grep -q '\.rej$'; then echo "$new" >&2; die "$f left .rej files"; fi
  if [ -n "$new" ]; then echo "$new" | while read -r o; do [ -n "$o" ] && rm -f "$o"; done; fi
  n=$(grep -c '^diff ' "$here/$f" || true)
  echo "apply-series: $f applied ($n files, fuzz 0)"
done
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "apply-series: OK INCLUDE='$(echo $inc)' patches=${#list[@]}"
