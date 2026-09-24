#!/usr/bin/env bash
# densegemm r1 install (run by Dockerfile.box, or by hand inside a --rm build container without --gpus):
#   1. hash the SASS of every function in the installed exllamav3_ext .so (cuobjdump -sass written to a FILE, then
#      sass_hashes.py; never a shell variable or here-string: that broke R703)
#   2. apply densegemm-r1.patch to the installed exllamav3 (dry run first, --fuzz=0, exit codes checked)
#   3. rebuild exllamav3_ext for sm_120 into site-packages over the prebuilt .so (rebuild-native.py)
#   4. served-SASS identity: every (function, SASS hash) of the old .so must be in the new .so, unchanged; the only
#      new functions are the 32 V2 twins (exact mangled-name prefixes); anything else fails the install
#   5. landing check: import torch BEFORE exllamav3_ext; revision marker; flags default off; env parsing
# EXL3_DENSE_V2 / EXL3_DENSE_ROWS32 are NOT set here: unset = the served kernels.
# CPU-only use outside a GPU lock: MAX_JOBS=2 and only while /srv/qwen5090/gpu-queue is empty (OPERATIONS §15).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${DENSEGEMM_INSTALL_OUT:-$here/install}
mkdir -p "$out"
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
command -v nvcc >/dev/null && command -v cuobjdump >/dev/null
so=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
echo "densegemm: base .so $so"
nice -n 19 cuobjdump -sass "$so" > "$out/base.sass"
python3 "$here/sass_hashes.py" < "$out/base.sass" > "$out/base-sass.txt"
echo "densegemm: base .so has $(wc -l < "$out/base-sass.txt") function bodies"

patch -p1 --fuzz=0 --batch --forward --dry-run -d "$pkg" < "$here/densegemm-r1.patch"
patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch -d "$pkg" < "$here/densegemm-r1.patch"
if find "$pkg" -name '*.rej' | grep -q .; then echo "densegemm: patch left .rej files"; exit 1; fi
find "$pkg" -name __pycache__ -type d -prune -exec rm -rf {} +

build=$(mktemp -d)
( cd "$build" && TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS="${MAX_JOBS:-2}" nice -n 19 \
    python3 "$here/rebuild-native.py" build_ext --force --build-lib "$site" ) 2>&1 | tee "$out/build-native.log"
rm -rf "$build"

so2=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
nice -n 19 cuobjdump -sass "$so2" > "$out/rebuilt.sass"
python3 "$here/sass_hashes.py" < "$out/rebuilt.sass" > "$out/rebuilt-sass.txt"
nice -n 19 cuobjdump -res-usage "$so2" > "$out/rebuilt-res-usage.txt" 2>&1 || true
python3 - "$out/base-sass.txt" "$out/rebuilt-sass.txt" <<'PY'
import re, sys
def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            name, h = line.rsplit(" ", 1)
            d.setdefault(name, set()).add(h)
    return d
base, new = load(sys.argv[1]), load(sys.argv[2])
changed = [n for n in base if n not in new or not base[n] <= new[n]]
added = sorted(n for n in new if n not in base)
TWIN = re.compile(r"^(_Z19exl3_gemm_v2_kernel|_Z20exl3_mgemm_v2_kernel|_Z19exl3_gemv_v2_kernel)")
twins = [n for n in added if TWIN.match(n)]
other = [n for n in added if n not in twins]
kinds = {k: sum(1 for n in twins if n.startswith(k)) for k in ("_Z19exl3_gemm_v2", "_Z20exl3_mgemm_v2", "_Z19exl3_gemv_v2")}
print(f"densegemm: served functions {len(base)}; unchanged {len(base) - len(changed)}; changed or missing {len(changed)}; "
      f"new {len(added)} (twins {len(twins)} = {kinds}, other {len(other)})")
for n in changed[:20]:
    print("  CHANGED", n)
for n in other[:20]:
    print("  UNEXPECTED NEW", n)
if changed or other or kinds != {"_Z19exl3_gemm_v2": 12, "_Z20exl3_mgemm_v2": 12, "_Z19exl3_gemv_v2": 8}:
    print("densegemm: SERVED SASS IDENTITY FAILED (expected 0 changed, 0 other, twins 12 gemm + 12 mgemm + 8 gemv)")
    sys.exit(1)
print("densegemm: served SASS identical (whitespace collapsed); 32 twins added")
PY
# spills of the twins (ptxas -v is not in the setuptools log; the resource usage dump is)
grep -A1 -E "exl3_(m?gemm|gemv)_v2_kernel" "$out/rebuilt-res-usage.txt" | grep -oE "REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+" | sort | uniq -c | sed 's/^/densegemm: twin resource usage /' || true

python3 -c '
import torch, importlib.util as u
print("ext:", u.find_spec("exllamav3_ext").origin)
import exllamav3_ext as e
assert e.dense_v2_revision == 1, "dense_v2_revision"
assert tuple(e.dense_v2_modes()) == (0, 0), "flags must default off"
for f in ("dense_v2_set_mode", "dense_v2_launch_counts", "dense_v2_reset_counts", "exl3_gemm", "exl3_mgemm", "exl3_gemv"):
    assert hasattr(e, f), f
print("base markers: moefast", getattr(e, "moe_coop_v3_revision", None))'
EXL3_DENSE_V2=1 EXL3_DENSE_ROWS32=1 python3 -c 'import torch, exllamav3_ext as e; assert tuple(e.dense_v2_modes()) == (1, 1), e.dense_v2_modes()'
EXL3_DENSE_V2=3 python3 -c 'import torch, exllamav3_ext as e; assert tuple(e.dense_v2_modes()) == (3, 0), e.dense_v2_modes()'
if EXL3_DENSE_V2=true python3 -c 'import torch, exllamav3_ext as e; e.dense_v2_modes()' 2>/dev/null; then
  echo "densegemm: EXL3_DENSE_V2=true must be rejected"; exit 1; fi
echo "densegemm r1 landed: patch applied, exllamav3_ext rebuilt for sm_120, dense_v2_revision=1, served SASS identical, 32 twins (default off)"
