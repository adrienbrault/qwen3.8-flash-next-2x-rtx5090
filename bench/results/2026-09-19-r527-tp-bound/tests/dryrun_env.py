"""CPU import environment for the REAL exllamav3 package in src/ (no CUDA, no compiled extension, no triton).

install() must run before anything imports exllamav3. It
  * puts src/ first on sys.path and disables .pyc writing (src/ stays pristine),
  * registers `exllamav3.ext` as a module whose `exllamav3_ext` is ext_sigs.StubExt (only bound names, checked arity),
  * registers inert triton modules (the package imports triton kernels at module import time; nothing here runs them),
  * patches the torch.cuda entry points the package touches on a CPU host (synchronize, streams, events, _sleep,
    device properties) so module code that is written for CUDA can run its Python on CPU tensors.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(WORKSPACE, "src")
EXT_DIR = os.path.join(SRC, "exllamav3", "exllamav3_ext")

_state = {}


class _Rec:
    """Inert object: any attribute / call / index returns another _Rec (for triton's decorators and dtypes)."""

    def __init__(self, name="triton"):
        self._n = name

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Rec(f"{self._n}.{k}")

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        return _Rec(f"{self._n}()")

    def __getitem__(self, k):
        return self

    def __repr__(self):
        return f"<stub {self._n}>"


def _module(name, **attrs):
    m = types.ModuleType(name)

    def ga(k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Rec(f"{name}.{k}")
    m.__getattr__ = ga
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _decorator(*a, **k):
    if len(a) == 1 and callable(a[0]) and not k:
        return a[0]
    return lambda f: f


def install():
    if _state.get("installed"):
        return _state["ext"]
    sys.dont_write_bytecode = True
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    import torch
    # torch's meta/prims paths import dynamo -> inductor, which probe triton when it is importable: import them
    # BEFORE the inert triton stand-ins are registered so torch sees "no triton"
    import torch._dynamo  # noqa: F401
    import torch._inductor.runtime.hints  # noqa: F401
    from ext_sigs import ExtSigs, StubExt

    sigs = ExtSigs(EXT_DIR)
    ext = StubExt(sigs)
    ext_mod = types.ModuleType("exllamav3.ext")
    ext_mod.exllamav3_ext = ext
    sys.modules["exllamav3.ext"] = ext_mod

    class _Compiled:
        asm = {"cubin": b""}

        class metadata:
            name = "stub_kernel"
            num_warps = 4
            shared = 0

    def _np2(n):
        n = int(n)
        return 1 if n <= 1 else 1 << (n - 1).bit_length()

    for n in ("triton", "triton.language", "triton.compiler", "triton.runtime", "triton.language.extra",
              "triton.language.extra.cuda", "triton.language.extra.libdevice", "triton.runtime.jit",
              "triton.backends", "triton.language.core"):
        sys.modules.setdefault(n, _module(n, jit=_decorator, autotune=_decorator, heuristics=_decorator,
                                          Config=_Rec("triton.Config"), cdiv=lambda a, b: (a + b - 1) // b,
                                          next_power_of_2=_np2, compile=lambda *a, **k: _Compiled()))

    # torch.cuda on a CPU host
    class _Stream:
        def __init__(self, *a, **k):
            self.cuda_stream = 0
            self.device = torch.device("cpu")

        def wait_stream(self, *a):
            pass

        def wait_event(self, *a):
            pass

        def synchronize(self):
            pass

        def record_event(self, *a):
            return _Event()

    class _Event:
        def __init__(self, *a, **k):
            pass

        def record(self, *a):
            pass

        def synchronize(self):
            pass

        def wait(self, *a):
            pass

        def query(self):
            return True

        def elapsed_time(self, other):
            return 1.0

    class _Props:
        name = "cpu-dryrun"
        clock_rate = 2_400_000
        multi_processor_count = 170
        total_memory = 32 << 30
        major, minor = 12, 0
        L2_cache_size = 96 << 20

    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.Stream = _Stream
    torch.cuda.Event = _Event
    torch.cuda.current_stream = lambda *a, **k: _Stream()
    torch.cuda.default_stream = lambda *a, **k: _Stream()
    torch.cuda.stream = lambda *a, **k: contextlib.nullcontext()
    torch.cuda.device = lambda *a, **k: contextlib.nullcontext()
    torch.cuda._sleep = lambda *a, **k: None
    torch.cuda.get_device_properties = lambda *a, **k: _Props()
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.current_device = lambda: 0
    torch.cuda.is_available = lambda: True
    _state.update(installed=True, ext=ext, sigs=sigs)
    return ext


def ext():
    return _state["ext"]
