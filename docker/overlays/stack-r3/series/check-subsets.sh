#!/usr/bin/env bash
# Subset matrix for the stack-r3 series, CPU only. Rebuilds the stack-r2 tree from the pristine served package
# (src/exllamav3 = tabbyapi:slotfix-r1's package) + hcfast-r1.patch + moefast-r1.patch at fuzz 0 (as every
# make-patch.sh of the stack-r2 patches does), then for every subset of the optional components:
#   1. apply-series.sh must succeed (fuzz 0, no .rej);
#   2. the same patches applied in a PERMUTED order (optional components first, core last, guard right after its
#      densegemm) must give a byte-identical tree: the hunks commute, so no patch depends on another's context;
#   3. bindings.cpp carries exactly one line per revision marker, with the expected values;
#   4. the densegemm side-branch detection is the first statement of dense_v2_launch_gemm and dense_v2_launch_gemv.
# Negative controls: INCLUDE="dg1 dg2" is refused; a second application on the same tree is refused; the original
# latchain-r1 + moefast-r3 pair rejects in both orders (the R712 blocker this series fixes).
#   check-subsets.sh SRC [OVERLAYS]   -> prints the matrix; exit 0 = every row OK
# SRC = the installed exllamav3 package of tabbyapi:slotfix-r1, copied out of the image; OVERLAYS = docker/overlays
# (default: two levels above this file), which holds hcfast-r1/, moefast-r1/, latchain-r1/ and moefast-r3/.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
SRC=${1:?usage: check-subsets.sh SRC [OVERLAYS]}
P=${2:-$(cd "$here/../.." && pwd)}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/stack-r3-subsets.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
ap(){ sed 's/^$/ /' "$2" | patch -s -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$1" > "$tmp/ap.log" 2>&1; }
clean(){ find "$1" \( -name '*.orig' -o -name '*.rej' \) ! -path '*/model/model_ls.py.orig' -delete; }
cp -R "$SRC" "$tmp/stack"; find "$tmp/stack" -name __pycache__ -prune -exec rm -rf {} +
ap "$tmp/stack" "$P/hcfast-r1/hcfast-r1.patch" && ap "$tmp/stack" "$P/moefast-r1/moefast-r1.patch" || { echo "cannot rebuild stack-r2"; exit 2; }
clean "$tmp/stack"
echo "# stack-r3 subset matrix $(date -u +%FT%TZ); base = src + hcfast-r1 + moefast-r1 (fuzz 0); patch: $(patch --version 2>&1 | head -1)"
fail=0
row(){ echo "$*"; case "$*" in *FAIL*) fail=1 ;; esac; }
for inc in "" "mf3" "dg1" "dg2" "mf3 dg1" "mf3 dg2"; do
  t=$tmp/s; rm -rf "$t"; cp -R "$tmp/stack" "$t"
  if ! INCLUDE="$inc" bash "$here/apply-series.sh" "$t" > "$tmp/fwd.log" 2>&1; then
    row "INCLUDE='$inc': apply-series FAIL"; sed 's/^/    /' "$tmp/fwd.log" | tail -8; continue; fi
  npatch=$(grep -c ' applied (' "$tmp/fwd.log")
  # permuted order: optional first (densegemm then its guard, then moefast-r3), core last (latchain-r1b, hcfast)
  perm=(); for f in $(INCLUDE="$inc" bash "$here/apply-series.sh" --list); do perm+=("$f"); done
  order=(); for f in "${perm[@]}"; do case $f in 04-*|05-*) order+=("$f");; esac; done
  for f in "${perm[@]}"; do case $f in 06-*|03-*) order+=("$f");; esac; done
  for f in "${perm[@]}"; do case $f in 02-*) order+=("$f");; esac; done
  for f in "${perm[@]}"; do case $f in 01-*) order+=("$f");; esac; done
  u=$tmp/u; rm -rf "$u"; cp -R "$tmp/stack" "$u"; pok=1
  for f in "${order[@]}"; do ap "$u" "$here/$f" || { pok=0; break; }; done
  clean "$u"
  if [ $pok = 1 ] && command diff -r -x __pycache__ "$t" "$u" > /dev/null; then perm_res="permuted order (${order[*]}) identical"
  else perm_res="permuted order FAIL ($([ $pok = 1 ] && echo 'trees differ' || echo "rejects: $(head -2 "$tmp/ap.log" | tr '\n' ' ')"))"; fi
  b=$t/exllamav3_ext/bindings.cpp
  want_mf=1; case " $inc " in *" mf3 "*) want_mf=3 ;; esac
  mk="markers"
  c(){ grep -cE "$1" "$b" || true; }
  [ "$(c 'm.attr\("hc_mix_v3_revision"\) = 2;')" = 1 ] && mk="$mk hc=2" || mk="$mk hc=FAIL"
  [ "$(c 'm.attr\("latchain_revision"\) = 1;')" = 1 ] && mk="$mk lc=1" || mk="$mk lc=FAIL"
  [ "$(c "m.attr\(\"moe_coop_v3_revision\"\) = $want_mf;")" = 1 ] && [ "$(c 'moe_coop_v3_revision')" = 1 ] && mk="$mk mf=$want_mf" || mk="$mk mf=FAIL"
  case " $inc " in
    *" dg1 "*|*" dg2 "*)
      want_dg=1; case " $inc " in *" dg2 "*) want_dg=2 ;; esac
      grep -q "#define DENSE_V2_REVISION $want_dg" "$t/exllamav3_ext/quant/exl3_dense_v2.cuh" && mk="$mk dg=$want_dg" || mk="$mk dg=FAIL"
      [ "$(c 'm.attr\("dense_v2_lc_guard"\)')" = 1 ] && mk="$mk guard=1" || mk="$mk guard=FAIL"
      g=$(python3 - "$t/exllamav3_ext/quant/exl3_dense_v2.cu" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
ok = []
for fn in ("dense_v2_launch_gemm", "dense_v2_launch_gemv"):
    m = re.search(r"\nvoid\* " + fn + r"\n\(.*?\)\n\{\n(.*?)\n", t, re.S)
    ok.append(bool(m) and m.group(1).strip().startswith("bool side = lc_side_branch(kernel_args, device);"))
print("guard-first" if all(ok) else "guard-first=FAIL")
EOF
); mk="$mk $g" ;;
    *) [ "$(c 'dense_v2')" = 0 ] && mk="$mk dg=none" || mk="$mk dg=FAIL(unexpected)" ;;
  esac
  row "INCLUDE='$inc': OK $npatch patches; $perm_res; $mk"
  # a second application must be refused (base check)
  if INCLUDE="$inc" bash "$here/apply-series.sh" "$t" > /dev/null 2>&1; then row "  re-apply on the same tree: accepted -> FAIL"; fi
done
if INCLUDE="dg1 dg2" bash "$here/apply-series.sh" --list > /dev/null 2>&1; then row "INCLUDE='dg1 dg2': accepted -> FAIL"
else row "INCLUDE='dg1 dg2': refused (OK)"; fi
for o in "latchain-r1/latchain-r1.patch moefast-r3/moefast-r3.patch" "moefast-r3/moefast-r3.patch latchain-r1/latchain-r1.patch"; do
  set -- $o; u=$tmp/n; rm -rf "$u"; cp -R "$tmp/stack" "$u"; ap "$u" "$P/$1"; r1=$?; ap "$u" "$P/$2"; r2=$?
  if [ $r1 = 0 ] && [ $r2 != 0 ]; then row "control: original $1 then $2 rejects (the R712 blocker; OK)"
  else row "control: original $1 then $2 rc $r1/$r2 -> FAIL (expected the second to reject)"; fi
done
[ $fail = 0 ] && echo "SUBSET MATRIX OK" || { echo "SUBSET MATRIX FAILED"; exit 1; }
