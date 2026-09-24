#!/usr/bin/env bash
# stack-r3 install: run by Dockerfile.box on BASE=tabbyapi:stack-r2 (or by hand in a --rm build container without
# --gpus). One patch series, one rebuild, one set of checks:
#   0. the base is stack-r2 (hcfast r1, moefast r1, no latchain / densegemm), checked on the installed package + .so
#   1. SASS + resource usage of the base .so -> FILES (cuobjdump output is never kept in a shell variable: R703)
#   2. series/apply-series.sh with INCLUDE (fuzz 0, sha256 per patch, any .rej fails)
#   3. ONE rebuild of exllamav3_ext for sm_120 (densegemm-r2's rebuild-native.py: the common builder plus
#      DGV2_NVCC_DEFS, which must be empty unless dg2 is included; R714's bisect choice when it is)
#   4. served-SASS identity (tools/sass_identity_stack.py): every stack-r2 function unchanged; new = exactly the
#      INCLUDE set (hcv4 >= 1, 8 _rr, [36 + 1 moefast r2], [32 densegemm twins])
#   5. STACK bound (tools/stack_bound.py): every new kernel <= its served reference + 64 B (cuobjdump -res-usage
#      STACK, not LOCAL: R710); dg2 additionally runs densegemm-r2's compile_bar.py on the real .so
#   6. landing (tools/landing.py): torch first, every revision marker, flags default off, HC2 tables parse
#   7. CPU tests: latchain (data flow + source lints incl. host call order vs the saved stack-r2 tree), hcfast r2
#      flow, [moefast r3 trace + r3 (part 2 required)], [densegemm model + the stack-r3 guard test]
# Env: INCLUDE ("", "mf3", "dg1", "dg2", "mf3 dg1", "mf3 dg2"), DGV2_NVCC_DEFS, MAX_JOBS, STACK_INSTALL_OUT.
# Nothing is enabled here: unset EXL3_* = the served stack-r2 path.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${STACK_INSTALL_OUT:-$here/install}
mkdir -p "$out"
INCLUDE=$(echo "${INCLUDE:-}" | tr ',' ' ' | xargs)
DGV2_NVCC_DEFS=${DGV2_NVCC_DEFS:-}
case " $INCLUDE " in *" dg2 "*) ;; *) [ -z "$DGV2_NVCC_DEFS" ] || { echo "stack-r3: DGV2_NVCC_DEFS='$DGV2_NVCC_DEFS' without dg2"; exit 1; } ;; esac
echo "stack-r3: INCLUDE='$INCLUDE' DGV2_NVCC_DEFS='$DGV2_NVCC_DEFS' MAX_JOBS=${MAX_JOBS:-4}"
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc >/dev/null && command -v cuobjdump >/dev/null

# 0. base = stack-r2
python3 -c '
import torch, exllamav3_ext as e
assert getattr(e, "hc_mix_v3_revision", None) == 1, "base lacks hcfast r1 (or already has r2)"
assert getattr(e, "moe_coop_v3_revision", None) == 1, "base lacks moefast r1"
assert not hasattr(e, "latchain_revision") and not hasattr(e, "dense_v2_revision"), "base already carries latchain / densegemm"
print("stack-r3: base markers ok (hcfast 1, moefast 1)")'
rm -rf "$out/stack-r2-pkg" && cp -R "$pkg" "$out/stack-r2-pkg"      # for latchain's host call-order lint
so=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
echo "stack-r3: base .so $so"

# 1. base SASS + resources
nice -n 19 cuobjdump -sass "$so" > "$out/base.sass"
python3 "$here/tools/sass_hashes.py" "$out/base.sass" > "$out/base-sass.txt"
rm -f "$out/base.sass"
nice -n 19 cuobjdump -res-usage "$so" > "$out/base-res-usage.txt" 2>&1
echo "stack-r3: base .so has $(wc -l < "$out/base-sass.txt") function bodies, $(grep -c 'Function' "$out/base-res-usage.txt") res-usage entries"

# 2. series
INCLUDE="$INCLUDE" bash "$here/series/apply-series.sh" "$pkg" | tee "$out/apply-series.log"

# 3. rebuild
build=$(mktemp -d "${TMPDIR:-/tmp}/stack-r3-build.XXXXXX")
( cd "$build" && DGV2_NVCC_DEFS="$DGV2_NVCC_DEFS" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-4}" nice -n 19 \
    python3 "$here/tools/rebuild-native.py" build_ext --force --build-lib "$site" ) > "$out/build-native.log" 2>&1 \
  || { tail -40 "$out/build-native.log"; echo "stack-r3: REBUILD FAILED"; exit 1; }
rm -rf "$build"
so2=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")

# 4. served-SASS identity
nice -n 19 cuobjdump -sass "$so2" > "$out/rebuilt.sass"
python3 "$here/tools/sass_hashes.py" "$out/rebuilt.sass" > "$out/rebuilt-sass.txt"
nice -n 19 cuobjdump -res-usage "$so2" > "$out/rebuilt-res-usage.txt" 2>&1
python3 "$here/tools/sass_identity_stack.py" "$out/base-sass.txt" "$out/rebuilt-sass.txt" --include "$INCLUDE" | tee "$out/sass-identity.txt"

# 5. STACK bound
python3 "$here/tools/stack_bound.py" --base "$out/base-res-usage.txt" --new "$out/rebuilt-res-usage.txt" --include "$INCLUDE" \
  | tee "$out/stack-bound.txt" | grep -E "^STACK-BOUND|FAIL|over" || true
grep -q "^STACK-BOUND PASS" "$out/stack-bound.txt" || { echo "stack-r3: STACK BOUND FAILED (see $out/stack-bound.txt)"; exit 1; }
case " $INCLUDE " in *" dg2 "*)
  python3 "$here/tools/compile_bar.py" --base "$out/base-res-usage.txt" --twins "$out/rebuilt-res-usage.txt" \
    --sass "$out/rebuilt.sass" --label "stack-r3 ${DGV2_NVCC_DEFS:-no-defs}" | tee "$out/compile-bar.txt" | tail -1
  grep -q "^BAR .* PASS " "$out/compile-bar.txt" || { echo "stack-r3: densegemm-r2 COMPILE BAR FAILED"; exit 1; } ;;
esac
rm -f "$out/rebuilt.sass"

# 6. landing
python3 "$here/tools/landing.py" --include "$INCLUDE" | tee "$out/landing.txt"

# 7. CPU tests
python3 "$here/tests/latchain/test_latchain_cpu.py" --src "$pkg" --served "$out/stack-r2-pkg" > "$out/cpu-latchain.txt" 2>&1 \
  || { tail -30 "$out/cpu-latchain.txt"; echo "stack-r3: latchain CPU test FAILED"; exit 1; }
python3 "$here/tests/latchain/test_latchain_cpu.py" > "$out/cpu-latchain-installed.txt" 2>&1 \
  || { tail -30 "$out/cpu-latchain-installed.txt"; echo "stack-r3: latchain CPU test (installed, part C flag parsing) FAILED"; exit 1; }
python3 "$here/tests/hcfast/test_hcfast_flow_cpu.py" > "$out/cpu-hcfast.txt" 2>&1 \
  || { tail -30 "$out/cpu-hcfast.txt"; echo "stack-r3: hcfast CPU flow test FAILED"; exit 1; }
case " $INCLUDE " in *" mf3 "*)
  python3 "$here/tests/moefast/test_moefast_trace_cpu.py" > "$out/cpu-moefast-trace.txt" 2>&1 \
    || { tail -30 "$out/cpu-moefast-trace.txt"; echo "stack-r3: moefast trace CPU test FAILED"; exit 1; }
  MOEFAST_R3_REQUIRE_PART2=1 python3 "$here/tests/moefast/test_moefast_r3_cpu.py" > "$out/cpu-moefast-r3.txt" 2>&1 \
    || { tail -30 "$out/cpu-moefast-r3.txt"; echo "stack-r3: moefast r3 CPU test FAILED"; exit 1; } ;;
esac
dgv=""; case " $INCLUDE " in *" dg1 "*) dgv=densegemm-r1 ;; *" dg2 "*) dgv=densegemm-r2 ;; esac
if [ -n "$dgv" ]; then
  rm -rf "$out/noguard-pkg" && cp -R "$pkg" "$out/noguard-pkg"
  patch -s -R -p1 --fuzz=0 --batch --no-backup-if-mismatch -d "$out/noguard-pkg" < "$here/series/06-dense-lcguard.patch"
  python3 "$here/tests/$dgv/test_densegemm_cpu.py" > "$out/cpu-densegemm.txt" 2>&1 \
    || { tail -30 "$out/cpu-densegemm.txt"; echo "stack-r3: densegemm CPU test FAILED"; exit 1; }
  python3 "$here/tests/test_dense_lcguard_cpu.py" --tree "$pkg" --untouched "$out/noguard-pkg" > "$out/cpu-lcguard.txt" 2>&1 \
    || { tail -30 "$out/cpu-lcguard.txt"; echo "stack-r3: lcguard CPU test FAILED"; exit 1; }
fi
for f in "$out"/cpu-*.txt; do echo "stack-r3: $(basename "$f"): $(tail -1 "$f")"; done
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "stack-r3 landed: INCLUDE='$INCLUDE' DGV2_NVCC_DEFS='$DGV2_NVCC_DEFS'; served SASS identical; STACK bound PASS; markers ok; CPU tests PASS (all flags default off)"
