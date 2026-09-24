#!/usr/bin/env bash
# moefast r3 install (run by Dockerfile.box on BASE=tabbyapi:stack-r2, or by hand in a --rm build container without
# --gpus). moefast-r3.patch is cumulative over stack-r2 (slotfix-r1 + hcfast-r1 + moefast-r1): it adds r2's kernels
# (EXL3_MOE_COOP_V3=3) and r3's overlap sub-flags (EXL3_SHARED_EXPERT_EARLY, EXL3_SHARED_EXPERT_PRIO, the diagnostic
# EXL3_MOE_COOP_V3_HEAD, the test entry exl3_moe_coop_ev). Steps:
#   1. hash the SASS of every function in the installed (stack-r2) .so: cuobjdump -> FILE -> sass_hashes.py
#   2. apply the patch (dry run first, --fuzz=0)
#   3. rebuild exllamav3_ext for sm_120 into site-packages (rebuild-native.py)
#   4. served-SASS identity (sass_identity.py): every base function unchanged; new = 36 r2 A/B + 1 head, nothing else
#   5. landing check: import torch BEFORE exllamav3_ext; revision 3; entry points; flags default off
# Nothing is enabled here: unset = the served kernels and the served late fork.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${MOEFAST_INSTALL_OUT:-$here/install}
mkdir -p "$out"
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc >/dev/null && command -v cuobjdump >/dev/null
so=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
echo "moefast-r3: base .so $so"
nice -n 19 cuobjdump -sass "$so" > "$out/base.sass"
python3 "$here/sass_hashes.py" "$out/base.sass" > "$out/base-sass.txt"
rm -f "$out/base.sass"
echo "moefast-r3: base .so has $(wc -l < "$out/base-sass.txt") function bodies"
grep -q "exl3_moe_coop_v3_ns" "$out/base-sass.txt" || { echo "moefast-r3: base .so lacks moefast r1 (not stack-r2?)"; exit 1; }
if grep -q "exl3_moe_coop_r2_ns" "$out/base-sass.txt"; then echo "moefast-r3: base .so already has r2 kernels (wrong BASE)"; exit 1; fi

patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$here/moefast-r3.patch"
patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$here/moefast-r3.patch"
if find "$pkg" -name '*.rej' | grep -q .; then echo "moefast-r3: patch left .rej files"; exit 1; fi
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} +

build=$(mktemp -d)
( cd "$build" && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-4}" nice -n 19 \
    python3 "$here/rebuild-native.py" build_ext --force --build-lib "$site" )
rm -rf "$build"

so2=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
nice -n 19 cuobjdump -sass "$so2" > "$out/rebuilt.sass"
python3 "$here/sass_hashes.py" "$out/rebuilt.sass" > "$out/rebuilt-sass.txt"
rm -f "$out/rebuilt.sass"
python3 "$here/sass_identity.py" "$out/base-sass.txt" "$out/rebuilt-sass.txt" | tee "$out/sass-identity.txt"

python3 -c '
import torch, importlib.util as u
print("ext:", u.find_spec("exllamav3_ext").origin)
import exllamav3_ext as e
assert e.moe_coop_v3_revision == 3, "landing marker"
for f in ("exl3_moe_coop", "exl3_moe_coop_ev", "routing_std"):
    assert hasattr(e, f), f
for f in ("start_shared", "shared_early", "shared_prio", "run_bszN"):
    assert hasattr(e.BC_BlockSparseMLP, f), f
import exllamav3.modules.block_sparse_mlp as b
assert hasattr(b.BlockSparseMLP, "_shared_early_ok"), "python early-fork guard"
import exllamav3.modules.hyperconnections as h
print("markers: moefast", e.moe_coop_v3_revision, "hcfast", getattr(h, "_HC_MIX_V3_BUILD", None))'
echo "moefast r3 landed: patch applied over stack-r2, exllamav3_ext rebuilt for sm_120, moe_coop_v3_revision=3, served SASS identical, 36 r2 + 1 head kernels (all flags default off)"
