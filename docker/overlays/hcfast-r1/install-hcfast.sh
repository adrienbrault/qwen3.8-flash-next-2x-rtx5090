#!/usr/bin/env bash
# hcfast r1 install (run by Dockerfile.box, or by hand inside a --rm build container):
# apply hcfast-r1.patch to the installed exllamav3 (fuzz 0, dry run first), rebuild exllamav3_ext
# for sm_120 into site-packages over the prebuilt .so, then assert the new symbol and the landing
# marker. EXL3_HC_MIX_V3 is NOT set here: unset = the served path.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc
patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$here/hcfast-r1.patch"
patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$here/hcfast-r1.patch"
if find "$pkg" -name '*.rej' | grep -q .; then echo "hcfast: patch left .rej files"; exit 1; fi
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} +
build=$(mktemp -d)
( cd "$build" && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-4}" nice -n 19 \
    python3 "$here/rebuild-native.py" build_ext --force --build-lib "$site" )
rm -rf "$build"
python3 -c 'import torch, importlib.util as u; s = u.find_spec("exllamav3_ext"); print("ext:", s.origin); import exllamav3_ext as e; assert hasattr(e, "gr_mix_v2_int8_v3") and e.hc_mix_v3_revision == 1; assert hasattr(e, "gr_mix_v2_int8_statein") and hasattr(e, "hc_apply") and e.hc_mix_v2_revision == 2'
echo "hcfast r1: patch applied, exllamav3_ext rebuilt for sm_120"
