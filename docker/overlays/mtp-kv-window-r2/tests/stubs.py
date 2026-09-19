"""Import stubs for running the real exllamav3 generator/cache/attention modules on a CPU-only machine.

Only the compiled extension (exllamav3.ext) and triton are replaced, by permissive stand-ins that
make the modules importable. No scenario runs a kernel: every call that would launch one goes to a
fake model / recorder in mtp_window_scenarios.py, and a scenario that reached a stub by accident
gets a Stub object back, which fails the first comparison or JSON encoding it meets.
"""
from __future__ import annotations

import sys
import types


class Stub:
    def __init__(self, name: str = "stub"):
        self._name = name

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub(name)

    def __call__(self, *args, **kwargs):
        # Used as a decorator (triton.jit, triton.autotune(...)(fn)): hand the function back
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return Stub(self._name + "()")

    def __getitem__(self, key):
        return Stub(self._name + "[]")

    def __mro_entries__(self, bases):
        return (object,)

    def __repr__(self):
        return f"<Stub {self._name}>"


class StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Stub(name)


TRITON_MODULES = [
    "triton", "triton.language", "triton.compiler", "triton.compiler.compiler", "triton.runtime",
    "triton.runtime.jit", "triton.backends", "triton.backends.compiler", "triton.tools",
    "triton.language.extra", "triton.language.extra.cuda", "triton.language.extra.libdevice",
]


def install(ext_recorder = None) -> None:
    ext = types.ModuleType("exllamav3.ext")
    ext.exllamav3_ext = ext_recorder if ext_recorder is not None else Stub("exllamav3_ext")
    sys.modules["exllamav3.ext"] = ext
    for name in TRITON_MODULES:
        module = StubModule(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules["triton"].language = sys.modules["triton.language"]
