"""Helpers for the rebase165 PY CPU tests: locate the checkout and import single exllamav3 modules without running
the package __init__ files (which load the compiled extension and the whole model stack).

EXL3_SRC points at the exllamav3 checkout (default: ../../../wt-py relative to this file)."""
import ast
import importlib
import os
import sys
import types
from pathlib import Path

SRC = Path(os.environ.get("EXL3_SRC") or Path(__file__).resolve().parents[3] / "wt-py")
PKG = SRC / "exllamav3"


def text(rel):
    return (SRC / rel).read_text()


def stub_packages(*names):
    """Register exllamav3 and the named subpackages as bare packages (their __init__ does not run)"""
    for name in ("exllamav3",) + tuple(names):
        if name in sys.modules and getattr(sys.modules[name], "_rebase165_stub", False):
            continue
        mod = types.ModuleType(name)
        mod.__path__ = [str(SRC / name.replace(".", "/"))]
        mod._rebase165_stub = True
        sys.modules[name] = mod
        parent, _, leaf = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], leaf, mod)


def stub_module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    parent, _, leaf = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], leaf, mod)
    return mod


def load(name):
    """Import exllamav3.<...> with its parent packages stubbed"""
    parts = name.split(".")
    stub_packages(*[".".join(parts[:i]) for i in range(2, len(parts))])
    return importlib.import_module(name)


def functions(rel, names, cls=None, ns=None):
    """Extract top-level functions (or methods of `cls`) from a checkout file and exec them into a namespace"""
    tree = ast.parse(text(rel))
    body = tree.body
    if cls is not None:
        body = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    found = {n.name: n for n in body if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = set(names) - set(found)
    assert not missing, (rel, missing)
    for n in found.values():
        n.decorator_list = [d for d in n.decorator_list if not (isinstance(d, ast.Name) and d.id == "staticmethod")]
    ns = {} if ns is None else ns
    exec(compile(ast.fix_missing_locations(ast.Module(body=list(found.values()), type_ignores=[])), rel, "exec"), ns)
    return ns


def have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class _Recorder:
    """Stands in for the compiled extension: any attribute is a no-op callable (nothing launches in these tests)"""

    def __getattr__(self, name):
        return lambda *a, **k: None


def import_package():
    """
    Import the whole exllamav3 package from SRC (for tests that need real module classes). EXL3_STUB_EXT=1, or a failing
    real import (no GPU driver in the container), replaces the compiled extension with a no-op recorder; triton is
    stubbed only when it cannot be imported. Returns the package, or raises ImportError (callers skip).
    """
    import importlib
    sys.dont_write_bytecode = True
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if os.environ.get("EXL3_STUB_EXT") != "1":
        try:
            return importlib.import_module("exllamav3")
        except Exception:  # noqa: BLE001 -- fall back to the stub below
            for name in [n for n in sys.modules if n == "exllamav3" or n.startswith("exllamav3.")]:
                del sys.modules[name]
    rec = _Recorder()
    ext = types.ModuleType("exllamav3.ext")
    ext.exllamav3_ext = rec
    ext.exllamav3_ext_obj = rec
    ext.is_precompiled_extension_available = lambda: False
    sys.modules["exllamav3.ext"] = ext
    try:
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        from unittest.mock import MagicMock
        for n in ("triton", "triton.language"):
            sys.modules[n] = MagicMock()
    return importlib.import_module("exllamav3")
