"""A tiny CPU interpreter for the Triton kernels this round touches (tests only).

It runs the kernel sources extracted from the served and patched files, one program at a time, over torch CPU
tensors: pointers are (flat tensor, offset) pairs, ``tl.load``/``tl.store`` index the flat tensor, the arithmetic is
torch's. Only the ``tl`` subset the plane kernels use is implemented. Program order is sequential, so ``store``
records every flat index written per launch and reports a second write to the same element of the same tensor in
one launch (a write race on a GPU).
"""
from __future__ import annotations

import ast
import contextlib
import itertools
import types
from pathlib import Path

import torch


class _Ptr:
    __slots__ = ("base", "off")

    def __init__(self, base: torch.Tensor, off = 0):
        self.base = base
        self.off = off

    def __add__(self, other):
        return _Ptr(self.base, self.off + other)

    __radd__ = __add__


def _index(off):
    if isinstance(off, int):
        return off
    return off.long()


class _State:
    pid = (0, 0, 0)
    writes = {}
    races = []


def _program_id(axis):
    return _State.pid[axis]


def _load(ptr, mask = None, other = None):
    assert mask is None, "masked loads are not used by the plane kernels"
    v = ptr.base[_index(ptr.off)]
    return v.clone()


def _store(ptr, value, mask = None):
    assert mask is None
    idx = _index(ptr.off)
    key = ptr.base.data_ptr()
    seen = _State.writes.setdefault(key, set())
    flat = [idx] if isinstance(idx, int) else idx.reshape(-1).tolist()
    for i in flat:
        if i in seen:
            _State.races.append((key, i))
        seen.add(i)
    if torch.is_tensor(value):
        ptr.base[idx] = value.to(ptr.base.dtype)
    else:
        ptr.base[idx] = value


tl = types.SimpleNamespace(
    constexpr = type("constexpr", (), {}),
    float16 = torch.float16,
    float32 = torch.float32,
    int32 = torch.int32,
    program_id = _program_id,
    arange = lambda a, b: torch.arange(a, b),
    zeros = lambda shape, dtype: torch.zeros(shape, dtype = dtype),
    load = _load,
    store = _store,
    sum = lambda x, axis = 0: torch.sum(x, dim = axis),
    rsqrt = torch.rsqrt,
    cos = torch.cos,
    sin = torch.sin,
)


class Kernel:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            wrap = lambda a: _Ptr(a.reshape(-1)) if torch.is_tensor(a) else a
            for a in list(args) + list(kwargs.values()):
                if torch.is_tensor(a):
                    assert a.is_contiguous(), "kernel arguments must be contiguous"
            wargs = [wrap(a) for a in args]
            wkw = {k: wrap(v) for k, v in kwargs.items()}
            _State.writes = {}
            for pid in itertools.product(*[range(g) for g in grid]):
                _State.pid = tuple(pid) + (0,) * (3 - len(pid))
                self.fn(*wargs, **wkw)
        return launch


def load_functions(path: Path, names: list[str], kernels: dict | None = None) -> dict:
    """Extract top-level functions from a source file and exec them against the fake tl. Functions decorated with
    triton.jit become Kernel objects; plain functions see the kernels (and each other) as globals."""
    src = path.read_text()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node
    missing = set(names) - set(found)
    assert not missing, f"{path}: missing {sorted(missing)}"
    ns = {"tl": tl, "torch": torch, "triton": types.SimpleNamespace(cdiv = lambda a, b: -(-a // b))}
    if kernels:
        ns.update(kernels)
    out = {}
    for name, node in found.items():
        is_kernel = any(
            ast.unparse(d).startswith("triton.jit") for d in node.decorator_list
        )
        node.decorator_list = []
        mod = ast.Module(body = [node], type_ignores = [])
        exec(compile(mod, str(path), "exec"), ns)
        out[name] = Kernel(ns[name]) if is_kernel else ns[name]
        ns[name] = out[name]
    return out


def reset_races():
    _State.races = []


def races():
    return list(_State.races)


def kernel_set(qsa_triton: Path, mla_triton: Path):
    """The K object rawk_scenarios expects: served pair + ring kernels + the eager ring launcher."""
    served = load_functions(qsa_triton, ["_qsa_pool_update_kernel"])
    mla = load_functions(mla_triton, ["_mla_plane_update_kernel"])
    ring = load_functions(qsa_triton, [
        "_qsa_raw_ring_append_kernel", "_qsa_pool_update_ring_bc_kernel", "_qsa_pool_update_ring_kernel",
    ])
    launcher = load_functions(qsa_triton, ["qsa_ring_plane_update"], kernels = ring)
    return types.SimpleNamespace(
        mla = mla["_mla_plane_update_kernel"],
        pool = served["_qsa_pool_update_kernel"],
        ring_append = ring["_qsa_raw_ring_append_kernel"],
        ring_bc_pool = ring["_qsa_pool_update_ring_bc_kernel"],
        ring_plane_update = launcher["qsa_ring_plane_update"],
        ctx = lambda dev: contextlib.nullcontext(),
    )
