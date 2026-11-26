"""rebase167: rebuild the native extension (exllamav3_ext) against the image's installed Torch ABI, CPU only.

Run inside the image build (Dockerfile), with the integrated exllamav3 package already copied over the
installed one. Usage:

    python3 rebuild-native-167.py build_ext --force --build-lib <site-packages>

Differences from the served recipe (/opt/rebase-dev-r3/rebuild-native.py) and why:

* Sources: upstream v1.6.7 keeps ROCm-only translation units under exllamav3_ext/rocm/, which the served
  recipe's blind rglob would compile on CUDA. The source list comes from upstream's own selector,
  util/cuda_flags.extension_sources(hip=False), loaded by path from the package being built.
* nvcc flags: the served recipe passes '-lineinfo -O3 --use_fast_math -Xcudafe --diag_suppress=177
  -Xcudafe --diag_suppress=20012'. Upstream cuda_flags.cuda_cflags() (9b498413) drops -lineinfo unless
  EXLLAMA_EXT_LINEINFO is set and adds --compress-mode=size unless EXLLAMA_EXT_COMPRESS=0. This script sets
  both env knobs so cuda_cflags() yields exactly the served list, and asserts it: no flag difference, so
  unchanged kernels keep their served SASS. (--compress-mode only packs the fatbin and -lineinfo does not
  change code generation, but neutralising both keeps the SASS/size comparison against served exact.)
* C++ flags: '-Ofast', as served and as upstream setup.py on Linux.
* The served same-stem rename (quant/exl3_moe_prefill_e3.cu -> exl3_moe_prefill_e3__cuda.cu) is already in
  the source tree; a remaining <stem>.cpp + <stem>.cu pair is a hard error here instead of a silent rename.
* DGV2_NVCC_DEFS (served densegemm bisect knob) is still appended when set; the served daily build used ''.
"""
import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SERVED_NVCC_FLAGS = [
    "-lineinfo", "-O3", "--use_fast_math",
    "-Xcudafe", "--diag_suppress=177",
    "-Xcudafe", "--diag_suppress=20012",
]
SERVED_CXX_FLAGS = ["-Ofast"]

site = Path(os.environ.get("REBASE167_SITE") or sysconfig.get_paths()["purelib"])
pkg = site / "exllamav3"
source = pkg / "exllamav3_ext"
assert (source / "libtorch/attention.cpp").is_file(), f"no extension sources under {source}"

spec = importlib.util.spec_from_file_location("cuda_flags_165", pkg / "util" / "cuda_flags.py")
cuda_flags = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cuda_flags)

os.environ["EXLLAMA_EXT_LINEINFO"] = "1"
os.environ["EXLLAMA_EXT_COMPRESS"] = "0"
nvcc_flags = cuda_flags.cuda_cflags(debug = False, hip = False)
assert nvcc_flags == SERVED_NVCC_FLAGS, f"nvcc flags differ from served: {nvcc_flags} != {SERVED_NVCC_FLAGS}"
nvcc_flags = nvcc_flags + os.environ.get("DGV2_NVCC_DEFS", "").split()

sources = cuda_flags.extension_sources(str(source), hip = False)
stems = {}
for p in map(Path, sources):
    key = p.with_suffix("")
    assert key not in stems, f"same-stem sources collide on one ninja object: {stems[key]} and {p}"
    stems[key] = p
assert not any("/rocm/" in s for s in sources), "ROCm sources selected for a CUDA build"

arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
assert arch, "TORCH_CUDA_ARCH_LIST must be set (no GPU at build time)"
print(f"rebase167 native rebuild: {len(sources)} sources, TORCH_CUDA_ARCH_LIST={arch!r}", flush = True)
print("nvcc flags: " + " ".join(nvcc_flags), flush = True)
print("cxx flags: " + " ".join(SERVED_CXX_FLAGS), flush = True)
with open(os.environ.get("REBASE167_SOURCES_OUT", "/dev/null"), "w") as f:
    f.write("\n".join(str(Path(s).relative_to(source)) for s in sources) + "\n")

setup(
    name = "exllamav3-ext-rebase167",
    ext_modules = [CUDAExtension(
        "exllamav3_ext", sources,
        extra_compile_args = {"cxx": SERVED_CXX_FLAGS, "nvcc": nvcc_flags},
    )],
    cmdclass = {"build_ext": BuildExtension},
)
