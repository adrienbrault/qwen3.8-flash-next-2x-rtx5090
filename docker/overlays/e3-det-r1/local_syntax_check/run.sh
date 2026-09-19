#!/usr/bin/env bash
# Local (no GPU, no nvcc) compile checks used for e3-det r1 on macOS arm64. NOT part of the image.
#  1. clang CUDA front end (Homebrew LLVM with the NVPTX target) parses and instantiates the whole
#     exl3_moe_prefill_e3.cu for sm_120a, device side and host side, against the real CUDA 12.8
#     headers (from the nvidia-cuda-{runtime,nvcc,cccl}-cu12 wheels) and small stand-ins for the torch /
#     c10 / libcu++ atomic headers (stub/): catches syntax, type and template errors in the kernels,
#     not nvcc-specific issues, register pressure or runtime behaviour.
#  2. clang++ -std=c++20 parses exl3_moe_prefill_e3.cpp and bindings.cpp against the real (CPU) torch
#     headers from the venv (pybind11 m.def instantiation included).
#  3. PTX is emitted for the new kernels; the script checks they contain no atom./red. instructions.
# Usage: run.sh <workspace root>   (expects .venv with torch, .tools/cuda = extracted CUDA headers)
set -euo pipefail
W=${1:?workspace root}
HERE=$(cd "$(dirname "$0")" && pwd)
CL=${CLANG:-/opt/homebrew/opt/llvm/bin/clang++}
TI=$W/.venv/lib/python3.12/site-packages/torch/include
PYI=$("$W/.venv/bin/python" -c "import sysconfig;print(sysconfig.get_paths()['include'])")
X=$(mktemp -d "$W/.tools/extcheck.XXXX")
trap 'rm -rf "$X"' EXIT
cp -R "$W/src/exllamav3/exllamav3_ext/." "$X/"
cp -R "$W/out/e3-det-r1/overlay/exllamav3/exllamav3_ext/." "$X/"
for mode in --cuda-device-only --cuda-host-only; do
  "$CL" -x cuda -std=c++17 --cuda-path="$W/.tools/cuda" --cuda-gpu-arch=sm_120a -nocudalib $mode -fsyntax-only \
    -Wall -Wno-unused-function -Werror -I"$HERE/stub" -include launch_decl.h -I"$X" "$X/quant/exl3_moe_prefill_e3.cu"
  echo "cu $mode: OK"
done
for u in quant/exl3_moe_prefill_e3.cpp bindings.cpp; do
  "$CL" -std=c++20 -fsyntax-only -I"$HERE/stub_cpp" -I"$TI" -I"$TI/torch/csrc/api/include" -I"$PYI" \
    -I"$W/.tools/cuda/include" -I"$X" "$X/$u"
  echo "$u: OK"
done
"$CL" -x cuda -std=c++17 -O3 --cuda-path="$W/.tools/cuda" --cuda-gpu-arch=sm_120a -nocudalib --cuda-device-only -S \
  -o "$X/e3.ptx" -I"$HERE/stub" -include launch_decl.h -I"$X" "$X/quant/exl3_moe_prefill_e3.cu"
python3 - "$X/e3.ptx" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
ents = [m.start() for m in re.finditer(r'\.visible \.entry', s)] + [len(s)]
bad = []
for i in range(len(ents) - 1):
    body = s[ents[i]:ents[i + 1]]
    name = re.search(r'\.entry (\S+?)\(', body).group(1)
    if 'det' in name:
        n = len(re.findall(r'\t(atom|red)\.', body))
        print(f"{name}: atom/red = {n}")
        bad += [name] if n else []
sys.exit(1 if bad else 0)
PY
echo "PTX: DET kernels have no atomics"
