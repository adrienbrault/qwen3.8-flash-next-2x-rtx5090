"""Build against the installed Torch ABI; run only inside the GPU build environment."""
import os
from pathlib import Path
import sysconfig
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
site = Path(sysconfig.get_paths()['purelib'])
source = site / 'exllamav3/exllamav3_ext'

# Ninja derives the object name from the source path minus extension, so a
# same-directory <stem>.cpp + <stem>.cu pair (host wrapper + CUDA TU, e.g.
# quant/exl3_moe_prefill_e3) collides on one .o and ninja aborts with
# "multiple rules generate". Source filenames are not referenced by anything —
# the wrappers include only .cuh headers — so rename the .cu side in place.
stems = {}
for p in sorted(source.rglob('*')):
    if p.suffix in ('.c', '.cpp', '.cu'):
        key = p.with_suffix('')
        if key in stems:
            cu = p if p.suffix == '.cu' else stems[key]
            if cu.suffix == '.cu':
                tgt = cu.with_name(cu.stem + '__cuda.cu')
                if not tgt.exists():
                    print(f'rename-on-collision: {cu.name} -> {tgt.name}')
                    cu.rename(tgt)
        else:
            stems[key] = p

sources = sorted(str(p) for p in source.rglob('*') if p.suffix in ('.c', '.cpp', '.cu'))
assert sources and (source / 'libtorch/attention.cpp').is_file()
setup(name='qsa-native-rebuild', ext_modules=[CUDAExtension(
    'exllamav3_ext', sources,
    extra_compile_args={'cxx': ['-Ofast'], 'nvcc': [
        '-lineinfo', '-O3', '--use_fast_math', '-Xcudafe', '--diag_suppress=177',
        '-Xcudafe', '--diag_suppress=20012'] + os.environ.get('DGV2_NVCC_DEFS', '').split()})],   # densegemm r2: bisect choice
    cmdclass={'build_ext': BuildExtension})
