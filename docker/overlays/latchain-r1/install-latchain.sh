#!/usr/bin/env bash
# latchain r1 install (run by Dockerfile.box, or by hand inside a --rm build container on tabbyapi:stack-r2): apply
# latchain-r1.patch to the installed exllamav3 (fuzz 0, dry run first), rebuild exllamav3_ext for sm_120 into
# site-packages over the prebuilt .so (rebuild-native.py, the hcfast / moefast builder), then check the landing:
# torch imported BEFORE the extension, latchain_revision == 1, TritonKernel.launch_py present, and 8 _rr recurrent
# kernel instantiations in the .so, counted from a SASS dump FILE.
# No EXL3_LC_* flag is set here: unset = the served path (default off).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc
command -v cuobjdump
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# the patch file is stored with its blank context lines as empty lines; restore the leading space (every line of a
# unified-diff hunk starts with ' ', '+', '-' or '\', so an empty line can only be blank context)
sed 's/^$/ /' "$here/latchain-r1.patch" > "$work/latchain-r1.patch"
patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$work/latchain-r1.patch"
patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$work/latchain-r1.patch"
if find "$pkg" -name '*.rej' | grep -q .; then echo "latchain: patch left .rej files"; exit 1; fi
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} +
mkdir -p "$work/build"
( cd "$work/build" && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-4}" nice -n 19 \
    python3 "$here/rebuild-native.py" build_ext --force --build-lib "$site" )
python3 -c 'import torch, importlib.util as u; s = u.find_spec("exllamav3_ext"); print("ext:", s.origin); import exllamav3_ext as e; assert e.latchain_revision == 1, "landing marker"; assert hasattr(e.TritonKernel, "launch_py"), "launch_py binding"; assert e.moe_coop_v3_revision == 1, "stack-r2 moefast marker lost"'
so=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
sass="$work/ext.sass"
cuobjdump -sass "$so" > "$sass"
n=$(grep -c "Function : .*cuda_recurrent_gated_delta_rule_kernel_128_rr" "$sass" || true)
[ "$n" = 8 ] || { echo "latchain: expected 8 _rr recurrent kernel instantiations in the .so, found $n"; exit 1; }
m=$(grep -c "Function : .*cuda_recurrent_gated_delta_rule_kernel_128I" "$sass" || true)
[ "$m" -ge 8 ] || { echo "latchain: served _128 recurrent kernels missing from the .so ($m)"; exit 1; }
echo "latchain r1 landed: patch applied, exllamav3_ext rebuilt for sm_120, latchain_revision=1, $n rr kernels, $m served _128 kernels (default off)"
