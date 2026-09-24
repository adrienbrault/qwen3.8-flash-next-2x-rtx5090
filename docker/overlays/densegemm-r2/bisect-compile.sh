#!/usr/bin/env bash
# densegemm r2 compile bisect: CPU only, no GPU, no daily bounce. Runs inside the BASE image (it has nvcc and cuobjdump):
#   docker run --rm --entrypoint bash -v <this dir>:/w:ro -v <out>:/out tabbyapi:stack-r2 /w/bisect-compile.sh /out
# For each variant it applies a patch to a scratch copy of the installed exllamav3 package, compiles the two twin TUs
# (comp_units/exl3_dense_v2_inst_{gemm,mgemm}.cu) with the real build's nvcc flags plus -Xptxas -v and the variant's
# defines, dumps cuobjdump -res-usage / -sass of the objects, and runs compile_bar.py against the served kernels of the
# base .so (the same shapes). Variants, in preference order after the r1 reference:
#   r1            ref/densegemm-r1.patch, no defines: must FAIL the bar (reproduces R710's stack frames; else the
#                 probe does not measure what the install measured and nothing after it is trusted)
#   r2            densegemm-r2.patch, defaults
#   r2-ai         -DDGV2_LAMBDA_AI=1   (always_inline on the main-loop lambdas)
#   r2-nobpro     -DDGV2_BPRO=0        (served prologue order)
#   r2-nobpro-ai  both
# Output in <out>: per variant <v>-<tu>.{log,res,sass}, bar-<v>.txt, and summary.txt whose last lines are
#   PROBE VALID|INVALID (r1 ...)
#   BISECT-CHOICE <variant> <defines, or "none">   or   BISECT-CHOICE none
#   ROWS32-COMPILE PASS|FAIL <variant>
# Choice: the first r2 variant whose 16-row AND rows32 twins all pass (EXL3_DENSE_ROWS32 = piece 4a, which rows32-r3
# needs); if none, the first whose 16-row twins pass (ROWS32-COMPILE FAIL: the gate reports 4a not ready).
# Exit 0 when a choice exists and the probe is valid, 1 otherwise. JOBS (default 4) compiles run in parallel.
set -uo pipefail
out=${1:?usage: bisect-compile.sh <out dir>}
here=$(cd "$(dirname "$0")" && pwd)
JOBS=${JOBS:-4}
mkdir -p "$out"
command -v nvcc >/dev/null && command -v cuobjdump >/dev/null || { echo "bisect: nvcc / cuobjdump missing"; exit 1; }
site=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
pkg="$site/exllamav3"
so=$(python3 -c "import importlib.util as u; print(u.find_spec('exllamav3_ext').origin)")
python3 -c 'import torch, exllamav3_ext as e; assert not hasattr(e, "dense_v2_revision"), "base image already carries densegemm"' 2>/dev/null \
  || { echo "bisect: $so is not a base image .so (dense_v2_revision present, or import failed)"; exit 1; }
nice -n 19 cuobjdump -res-usage "$so" > "$out/base-res-usage.txt" 2>&1
inc=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
pyinc=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT

VARIANTS="r1 r2 r2-ai r2-nobpro r2-nobpro-ai"
defs_of(){ case $1 in r1|r2) echo "";; r2-ai) echo "-DDGV2_LAMBDA_AI=1";; r2-nobpro) echo "-DDGV2_BPRO=0";;
                      r2-nobpro-ai) echo "-DDGV2_BPRO=0 -DDGV2_LAMBDA_AI=1";; esac; }
patch_of(){ case $1 in r1) echo "$here/ref/densegemm-r1.patch";; *) echo "$here/densegemm-r2.patch";; esac; }

for v in $VARIANTS; do
  t="$scratch/$v"; mkdir -p "$t"; cp -R "$pkg/exllamav3_ext" "$t/"
  patch -p1 --fuzz=0 --batch -s --no-backup-if-mismatch -d "$t" < "$(patch_of $v)" \
    || { echo "bisect: $(patch_of $v) does not apply to the base package"; exit 1; }
done

compile(){ local v=$1 tu=$2 t="$scratch/$1/exllamav3_ext" o="$out/$1-$2"
  ( cd "$t/quant/comp_units" && nice -n 19 nvcc -c "exl3_dense_v2_inst_$tu.cu" -o "$o.o" \
      -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
      --expt-relaxed-constexpr --compiler-options -fPIC -lineinfo -O3 --use_fast_math \
      -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=20012 \
      -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=exllamav3_ext -gencode=arch=compute_120,code=sm_120 -std=c++17 \
      -Xptxas -v $(defs_of "$v") -I"$t" $inc -I"$pyinc" ) > "$o.log" 2>&1 \
    || { echo "COMPILE-FAIL $v $tu"; return 0; }
  cuobjdump -res-usage "$o.o" > "$o.res" 2>&1
  cuobjdump -sass "$o.o" > "$o.sass" 2>&1
  rm -f "$o.o"
  echo "compiled $v $tu"; }
export -f compile defs_of; export scratch out inc pyinc
t0=$(date +%s)
for v in $VARIANTS; do for tu in gemm mgemm; do echo "$v $tu"; done; done \
  | xargs -P "$JOBS" -n 2 bash -c 'compile "$0" "$1"' | tee "$out/compile.txt"
echo "bisect: compiles done in $(( $(date +%s) - t0 )) s"

{
  choice16=none; choice32=none; probe=INVALID
  for v in $VARIANTS; do
    if grep -q "^COMPILE-FAIL $v " "$out/compile.txt"; then
      echo "== $v: COMPILE FAILED ($(defs_of $v))"; grep -m3 -E "error" "$out/$v-gemm.log" "$out/$v-mgemm.log" | sed 's/^/   /'
      continue
    fi
    cat "$out/$v-gemm.res" "$out/$v-mgemm.res" > "$out/$v.res"
    cat "$out/$v-gemm.sass" "$out/$v-mgemm.sass" > "$out/$v.sass"
    echo "== $v ($(defs_of $v))"
    python3 "$here/compile_bar.py" --base "$out/base-res-usage.txt" --twins "$out/$v.res" --sass "$out/$v.sass" --label "$v" \
      | tee "$out/bar-$v.txt"
    rc=${PIPESTATUS[0]}
    grep -hE "bytes stack frame" "$out/$v-gemm.log" "$out/$v-mgemm.log" | sed -E 's/^ +//' | sort | uniq -c \
      | sed 's/^/   ptxas: /' | head -12
    r32=$(grep -oE "ROWS32 (PASS|FAIL)$" "$out/bar-$v.txt" | cut -d' ' -f2)
    if [ "$v" = r1 ]; then
      [ "$rc" = 1 ] && probe=VALID
    elif [ "$rc" = 0 ]; then
      [ "$choice16" = none ] && choice16=$v
      [ "$r32" = PASS ] && [ "$choice32" = none ] && choice32=$v
    fi
  done
  echo "PROBE $probe (r1 must fail the bar, as its R710 build did)"
  choice=$choice32; [ "$choice" = none ] && choice=$choice16
  if [ "$probe" = VALID ] && [ "$choice" != none ]; then
    d=$(defs_of $choice); echo "BISECT-CHOICE $choice ${d:-none}"
    [ "$choice32" != none ] && echo "ROWS32-COMPILE PASS $choice" || echo "ROWS32-COMPILE FAIL $choice"
  else
    echo "BISECT-CHOICE none"
    echo "ROWS32-COMPILE FAIL none"
  fi
} 2>&1 | tee "$out/summary.txt"
chmod -R a+rX "$out" 2>/dev/null || true
grep -q "^BISECT-CHOICE r2" "$out/summary.txt"
