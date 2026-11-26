"""CPU bootstrap for the TABBY rebase165 tests (no GPU, no network).

install() puts the exllamav3 tree under test (EXL3_TREE, the directory that HOLDS the exllamav3 package) and the
TabbyAPI tree (TABBY_DIR) first on sys.path. With EXL3_STUB_EXT=1 the compiled extension `exllamav3.ext` is replaced
by a recording stub (the stub is the rebase-dev-r3 tests/_runner.py one); triton is stubbed only when it cannot be
imported. Nothing writes __pycache__.
"""
import os
import sys
import types

sys.dont_write_bytecode = True


class RecordingExt:
    """Stands in for the compiled `exllamav3_ext` module: records calls, returns None."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None
        return _call


def _stub_ext():
    ext = RecordingExt()
    mod = types.ModuleType("exllamav3.ext")
    mod.exllamav3_ext = ext
    mod.exllamav3_ext_obj = ext
    mod.is_precompiled_extension_available = lambda: False
    sys.modules["exllamav3.ext"] = mod


def _stub_triton():
    stub = types.ModuleType("triton")

    class _Fn:
        def __init__(self, fn, **meta):
            self.fn = fn
            self.meta = meta

        def __call__(self, *args, **kwargs):
            return None

        def __getitem__(self, grid):
            return self

    def jit(fn=None, **meta):
        if fn is not None:
            return _Fn(fn, **meta)
        return lambda f: _Fn(f, **meta)

    def autotune(*a, **k):
        return lambda f: f

    def heuristics(*a, **k):
        return lambda f: f

    stub.jit = jit
    stub.autotune = autotune
    stub.heuristics = heuristics
    stub.Config = lambda *a, **k: (a, k)
    stub.cdiv = lambda a, b: -(-a // b)
    stub.next_power_of_2 = lambda n: 1 << (max(int(n), 1) - 1).bit_length()
    stub.runtime = types.ModuleType("triton.runtime")
    stub.runtime.driver = types.SimpleNamespace(active=types.SimpleNamespace(
        get_current_target=lambda: types.SimpleNamespace(arch="cpu", backend="cpu"),
    ))
    tl = types.ModuleType("triton.language")
    for name in ("float32", "float16", "bfloat16", "int32", "int64", "int8", "int1", "uint8"):
        setattr(tl, name, name)
    tl.constexpr = int
    tl.core = types.ModuleType("triton.language.core")
    tl.core.constexpr = int
    tl.core.dtype = "f32"
    stub.language = tl
    sys.modules["triton"] = stub
    sys.modules["triton.runtime"] = stub.runtime
    sys.modules["triton.language"] = tl
    sys.modules["triton.language.core"] = tl.core


def install():
    tree = os.environ.get("EXL3_TREE")
    tabby = os.environ.get("TABBY_DIR", "/app")
    for path in (tabby, tree):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    if os.environ.get("EXL3_STUB_EXT") == "1":
        _stub_ext()
    try:
        import triton  # noqa: F401
    except ImportError:
        _stub_triton()
    return tree, tabby


def check_tree(tree):
    """The imported exllamav3 is the one under test (when EXL3_TREE is given)."""
    import exllamav3
    if tree:
        got = os.path.dirname(os.path.dirname(os.path.abspath(exllamav3.__file__)))
        assert got == os.path.abspath(tree), f"imported {exllamav3.__file__}, expected the tree under {tree}"
    return exllamav3
