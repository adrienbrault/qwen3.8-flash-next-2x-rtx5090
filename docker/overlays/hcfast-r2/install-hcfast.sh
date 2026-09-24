#!/usr/bin/env bash
# hcfast r2 install (run by Dockerfile.box, or by hand inside a --rm build container): apply hcfast-r2.patch
# (cumulative over slotfix-r1: r1 + r2) to the installed exllamav3 (fuzz 0, dry run first), rebuild
# exllamav3_ext for sm_120 into site-packages over the prebuilt .so, then assert the symbols and the landing
# markers. The base must NOT already carry hcfast r1 (hc_mix_v3.cu present): r2 is cumulative. A moefast-r1
# base is fine (the patches touch disjoint regions of bindings.cpp; check-composition.sh).
# EXL3_HC_MIX_V3 is NOT set here: unset = the served path.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc
if [ -e "$pkg/exllamav3_ext/hc_mix_v3.cu" ]; then
  echo "hcfast r2: $pkg already has hc_mix_v3.cu (an hcfast r1 base?); r2 is cumulative over slotfix-r1"; exit 1
fi
moefast=0; grep -q moe_coop_v3_revision "$pkg/exllamav3_ext/bindings.cpp" && moefast=1
patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$here/hcfast-r2.patch"
patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$here/hcfast-r2.patch"
if find "$pkg" -name '*.rej' | grep -q .; then echo "hcfast: patch left .rej files"; exit 1; fi
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} +
build=$(mktemp -d)
( cd "$build" && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-4}" nice -n 19 \
    python3 "$here/rebuild-native.py" build_ext --force --build-lib "$site" )
rm -rf "$build"
# torch first: exllamav3_ext links libc10.so, which only torch's import puts on the loader path.
python3 -c 'import torch, importlib.util as u; s = u.find_spec("exllamav3_ext"); print("ext:", s.origin); import exllamav3_ext as e; assert hasattr(e, "gr_mix_v2_int8_v3") and e.hc_mix_v3_revision == 2; assert hasattr(e, "gr_mix_v2_int8_statein") and hasattr(e, "hc_apply") and e.hc_mix_v2_revision == 2'
if [ "$moefast" = 1 ]; then
  python3 -c 'import torch, exllamav3_ext as e; assert e.moe_coop_v3_revision == 1; print("moefast r1 marker kept")'
fi
echo "hcfast r2: patch applied, exllamav3_ext rebuilt for sm_120 (moefast base: $moefast)"
