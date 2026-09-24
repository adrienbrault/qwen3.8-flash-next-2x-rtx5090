#!/usr/bin/env bash
# rows32 r4 install (run by Dockerfile.box on BASE=tabbyapi:stack-r3, or by hand in a --rm build container without
# --gpus). Host-only patch on the stack-r3 package:
#   0. the base is stack-r3 with densegemm-r2 (hcfast 2, latchain 1, moefast 1|3, densegemm 2 + lcguard) and no rows32
#   1. hash the SASS of every function in the installed exllamav3_ext .so (cuobjdump -sass written to a FILE, then
#      sass_hashes.py; never a shell variable or here-string: R703)
#   2. apply rows32-r4.patch to the installed package (dry run first, --fuzz=0, exit codes checked, no .rej)
#   3. rebuild exllamav3_ext for sm_120 with stack-r3's rebuild-native.py (the common builder + DGV2_NVCC_DEFS, which
#      must equal the base image's local.stack.dgv2_defs label: other defines would change densegemm-r2's SASS)
#   4. SASS identity: every (function, SASS hash) of the base .so must be in the rebuilt .so and no function may be
#      added (host-only patch); counts in $out/sass-identity.txt, anything else fails the install
#   5. landing (landing_rows32.py): re-exec without the image's baked EXL3_* ENV, stack-r3's own landing, torch first,
#      moe_rows32_revision 2, caps and module constants under the flags, strict parsing
#   6. CPU tests: test_rows32_cpu.py on the installed tree (index model, scratch sizes, patch hygiene, flags, source
#      model of the MoE mode per row count and of the lcguard side region); stack-r3's lcguard CPU test on the
#      installed tree (workspace layout with the rows32-sized side region)
# EXL3_DENSE_ROWS32 / EXL3_MOE_COOP_ROWS32 / EXL3_SHARED_EXPERT_ROWS32 are NOT set here: unset = the stack-r3 paths.
# Env: DGV2_NVCC_DEFS, MAX_JOBS (default 4: the gate builds under the GPU lock while the daily keeps serving, and the daily uses host CPU on every token), ROWS32_INSTALL_OUT.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${ROWS32_INSTALL_OUT:-$here/install}
mkdir -p "$out"
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc >/dev/null && command -v cuobjdump >/dev/null
DGV2_NVCC_DEFS=${DGV2_NVCC_DEFS:-}
echo "rows32 r4: DGV2_NVCC_DEFS='$DGV2_NVCC_DEFS' MAX_JOBS=${MAX_JOBS:-4}"

# 0. base = stack-r3 with densegemm-r2, no rows32 (EXL3_* stripped: the image bakes some into its ENV)
env -u EXL3_NONE $(env | sed -n 's/^\(EXL3_[A-Za-z0-9_]*\)=.*/-u \1/p') python3 -c '
import torch, exllamav3_ext as e
assert getattr(e, "hc_mix_v3_revision", None) == 2, "base lacks hcfast r2 (not stack-r3)"
assert getattr(e, "latchain_revision", None) == 1, "base lacks latchain (not stack-r3)"
assert getattr(e, "dense_v2_revision", None) == 2, "base lacks densegemm-r2 (EXL3_DENSE_ROWS32 needs its rows32 twins)"
assert getattr(e, "dense_v2_lc_guard", None) == 1, "base lacks the stack-r3 lcguard"
assert not hasattr(e, "moe_rows32_revision"), "base already carries rows32"
print("rows32 r4: base markers ok (hcfast 2, latchain 1, moefast", e.moe_coop_v3_revision, ", densegemm 2 + lcguard)")'
so=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
echo "rows32 r4: base .so $so"
nice -n 19 cuobjdump -sass "$so" > "$out/base.sass"
python3 "$here/sass_hashes.py" < "$out/base.sass" > "$out/base-sass.txt"
rm -f "$out/base.sass"
echo "rows32 r4: base .so has $(wc -l < "$out/base-sass.txt") function bodies"

# 2. patch
patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$here/rows32-r4.patch"
patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$here/rows32-r4.patch"
if find "$pkg" -name '*.rej' | grep -q .; then echo "rows32 r4: patch left .rej files"; exit 1; fi
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} +

# 3. rebuild
build=$(mktemp -d)
( cd "$build" && DGV2_NVCC_DEFS="$DGV2_NVCC_DEFS" TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-4}" nice -n 19 \
    python3 "$here/rebuild-native.py" build_ext --force --build-lib "$site" ) > "$out/build-native.log" 2>&1 \
  || { tail -40 "$out/build-native.log"; echo "rows32 r4: REBUILD FAILED"; exit 1; }
rm -rf "$build"

# 4. SASS identity
so2=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
nice -n 19 cuobjdump -sass "$so2" > "$out/rebuilt.sass"
python3 "$here/sass_hashes.py" < "$out/rebuilt.sass" > "$out/rebuilt-sass.txt"
rm -f "$out/rebuilt.sass"
python3 "$here/sass_identity.py" "$out/base-sass.txt" "$out/rebuilt-sass.txt" "$out/sass-identity.txt"

# 5. landing (strips the image's EXL3_* keys itself)
include=${STACK_INCLUDE:?STACK_INCLUDE (the base image label local.stack.include) is required}
python3 "$here/landing_rows32.py" --include "$include" | tee "$out/landing.txt"

# 6. CPU tests
python3 "$here/test_rows32_cpu.py" --tree "$pkg" > "$out/cpu-rows32.txt" 2>&1 \
  || { tail -30 "$out/cpu-rows32.txt"; echo "rows32 r4: CPU tests FAILED"; exit 1; }
python3 /opt/stack-r3/tests/test_dense_lcguard_cpu.py --tree "$pkg" > "$out/cpu-lcguard.txt" 2>&1 \
  || { tail -30 "$out/cpu-lcguard.txt"; echo "rows32 r4: lcguard CPU test FAILED on the rows32 tree"; exit 1; }
for f in "$out"/cpu-*.txt; do echo "rows32 r4: $(basename "$f"): $(tail -1 "$f")"; done
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "rows32 r4 landed: patch applied at fuzz 0, exllamav3_ext rebuilt for sm_120 (defines '$DGV2_NVCC_DEFS'), SASS identical, moe_rows32_revision=2, CPU tests PASS (flags default off)"
