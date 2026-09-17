"""Build against the installed Torch ABI; run only inside the GPU build environment."""
from pathlib import Path
import sysconfig
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
site = Path(sysconfig.get_paths()['purelib'])
source = site / 'exllamav3/exllamav3_ext'
sources = sorted(str(p) for p in source.rglob('*') if p.suffix in ('.c', '.cpp', '.cu'))
assert sources and (source / 'libtorch/attention.cpp').is_file()
setup(name='qsa-native-rebuild', ext_modules=[CUDAExtension(
    'exllamav3_ext', sources,
    extra_compile_args={'cxx': ['-Ofast'], 'nvcc': [
        '-lineinfo', '-O3', '--use_fast_math', '-Xcudafe', '--diag_suppress=177',
        '-Xcudafe', '--diag_suppress=20012']})],
    cmdclass={'build_ext': BuildExtension})
