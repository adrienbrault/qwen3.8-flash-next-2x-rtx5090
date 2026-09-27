"""Shared CPU test harness.

Runs the ported package WITHOUT the CUDA extension: the `exllamav3.ext` module is
stubbed with an object that records every attribute access / call, so tests can
assert which extension entry points a flag-gated code path selects. `triton` is
stubbed the same way (the served overlays' Triton kernels never run on CPU).
No packages are installed into the tree; `sys.dont_write_bytecode` is set so
nothing writes __pycache__.
"""
import sys
import types
import os

sys.dont_write_bytecode = True

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PORTED_DIR = os.path.dirname(TESTS_DIR)

# Parent of the package directory, so `import exllamav3` resolves to the ported tree
_pkg_parent = os.path.dirname(PORTED_DIR)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)


class RecordingExt:
    """Stands in for the compiled `exllamav3_ext` module: records calls, returns None."""

    def __init__(self):
        self.calls = []  # (name, args tuple, kwargs)

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None
        # Non-callable attributes (e.g. module constants) are handled by the
        # caller short-circuiting on presence; expose the wrapper for any name.
        return _call

    def called(self, name):
        return [c for c in self.calls if c[0] == name]

    def reset(self):
        self.calls.clear()


_ext = RecordingExt()
_ext_pkg = types.ModuleType("exllamav3.ext")
_ext_pkg.exllamav3_ext = _ext
_ext_pkg.is_precompiled_extension_available = lambda: False
_ext_pkg.exllamav3_ext_obj = _ext
sys.modules["exllamav3.ext"] = _ext_pkg


def _make_triton_stub():
    stub = types.ModuleType("triton")

    class _Fn:
        def __init__(self, fn, **meta):
            self.fn = fn
            self.meta = meta
            self.launched = []

        def __call__(self, *args, **kwargs):
            self.launched.append((args, kwargs))
            return None

    def jit(fn=None, **meta):
        if fn is not None:
            return _Fn(fn, **meta)

        def deco(f):
            return _Fn(f, **meta)
        return deco

    stub.jit = jit
    stub.cdiv = lambda a, b: -(-a // b)
    stub.runtime = types.ModuleType("triton.runtime")
    stub.runtime.driver = types.SimpleNamespace(active=types.SimpleNamespace(
        get_current_target=lambda: types.SimpleNamespace(arch="cpu"),
    ))
    stub.language = types.ModuleType("triton.language")
    stub.language.constexpr = int
    stub.language.float32 = "float32"
    stub.language.int32 = "int32"
    stub.language.int64 = "int64"
    stub.language.int1 = "int1"
    stub.language.bfloat16 = "bfloat16"
    tl = types.ModuleType("triton.language.core")
    tl.constexpr = int
    stub.language.core = tl
    stub.language.core.dtype = "f32"
    sys.modules["triton"] = stub
    sys.modules["triton.runtime"] = stub.runtime
    sys.modules["triton.language"] = stub.language
    sys.modules["triton.language.core"] = stub.language.core
    return stub


_make_triton_stub()


def get_ext():
    return _ext


def clear_env_flags():
    """Remove every EXL3_* selector so a fresh import sees the defaults."""
    for k in list(os.environ):
        if k.startswith("EXL3_"):
            del os.environ[k]
