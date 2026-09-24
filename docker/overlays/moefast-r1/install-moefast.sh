#!/usr/bin/env bash
# moefast r1 install (run by Dockerfile.box, or by hand inside a --rm build container): apply moefast-r1.patch to
# the installed exllamav3 (fuzz 0, dry run first), rebuild exllamav3_ext for sm_120 into site-packages over the
# prebuilt .so (rebuild-native.py, the hcfuse / mixstate builder), then assert the landing marker.
# EXL3_MOE_COOP_V3 is NOT set here: unset = the served V2 kernels.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc
patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$here/moefast-r1.patch"
patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$here/moefast-r1.patch"
if find "$pkg" -name '*.rej' | grep -q .; then echo "moefast: patch left .rej files"; exit 1; fi
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} +
build=$(mktemp -d)
( cd "$build" && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-4}" nice -n 19 \
    python3 "$here/rebuild-native.py" build_ext --force --build-lib "$site" )
rm -rf "$build"
python3 -c 'import torch, importlib.util as u; s = u.find_spec("exllamav3_ext"); print("ext:", s.origin); import exllamav3_ext as e; assert e.moe_coop_v3_revision == 1, "landing marker"; assert hasattr(e, "exl3_moe_coop")'
so=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
n=$(cuobjdump -sass "$so" | grep -c "Function : .*exl3_moe_coop_v3_ns.*exl3_moe_coop_v3_[ab]_kernel" || true)
[ "$n" = 45 ] || { echo "moefast: expected 45 V3 kernel instantiations in the .so, found $n"; exit 1; }
echo "moefast r1 landed: patch applied, exllamav3_ext rebuilt for sm_120, moe_coop_v3_revision=1, $n V3 kernels (default off)"
