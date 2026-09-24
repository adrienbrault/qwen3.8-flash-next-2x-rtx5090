#!/usr/bin/env bash
# CPU-only single-TU compile of the MoE coop instance TUs with the extension's flags + ptxas -v, and
# SASS dump. Runs inside a --rm tabbyapi:slotfix-r1 container (no --gpus) with the work dir at /w:
#   docker run --rm --entrypoint bash -v /srv/qwen5090/scratch/moefast:/w tabbyapi:slotfix-r1 /w/quick-nvcc.sh <tree> <out> [K...]
# <tree> is a directory holding exllamav3/exllamav3_ext (e.g. /w/base or /w/work).
set -euo pipefail
tree=$1; out=$2; shift 2
ks=${*:-2 3}
inc=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))")
pyinc=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
mkdir -p "$out"
cd "$tree/exllamav3/exllamav3_ext/quant/comp_units"
for k in $ks; do
  nvcc -c exl3_moe_coop_inst_k$k.cu -o "$out/k$k.o" -gencode arch=compute_120,code=sm_120 -std=c++17 \
    -lineinfo -O3 --use_fast_math -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=20012 \
    -Xptxas -v -I"$tree/exllamav3/exllamav3_ext" $inc -I"$pyinc" \
    -DTORCH_EXTENSION_NAME=exllamav3_ext -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    --expt-relaxed-constexpr -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
    > "$out/ptxas-k$k.log" 2>&1 || { cat "$out/ptxas-k$k.log"; exit 1; }
  cuobjdump -sass "$out/k$k.o" > "$out/k$k.sass"
done
# the launcher TU (host dispatch + rot kernel)
cd "$tree/exllamav3/exllamav3_ext/quant"
nvcc -c exl3_moe_coop.cu -o "$out/launcher.o" -gencode arch=compute_120,code=sm_120 -std=c++17 \
  -lineinfo -O3 --use_fast_math -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=20012 \
  -I"$tree/exllamav3/exllamav3_ext" $inc -I"$pyinc" \
  -DTORCH_EXTENSION_NAME=exllamav3_ext -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
  --expt-relaxed-constexpr -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ \
  > "$out/launcher.log" 2>&1 || { cat "$out/launcher.log"; exit 1; }
chmod -R a+rX "$out"
echo "nvcc OK: $out"
