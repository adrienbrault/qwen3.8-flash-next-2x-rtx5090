"""
CPU test harness: imports individual exllamav3 source files without the package's CUDA extension.

Every ``exllamav3.*`` module that is not loaded from a real file is an auto-stub whose attributes are
placeholder classes, so ``from ..x import A, B`` works. Real files are loaded from the overlay (the
deliverable) or from the pristine baseline ``src/exllamav3``. Flags are import-time constants in the real
modules, so each load takes the environment it should see and returns a fresh module object.
"""
from __future__ import annotations
import importlib.util
import os
import pathlib
import sys
import types

# never write .pyc next to the pristine src/ baseline or the overlay
sys.dont_write_bytecode = True

HERE = pathlib.Path(__file__).resolve().parent
DELIV = HERE.parent
OVERLAY = DELIV / "overlay" / "exllamav3"
WORKSPACE = DELIV.parent.parent
BASELINE = WORKSPACE / "src" / "exllamav3"

FLAGS = (
    "EXL3_EMBED_GPU", "EXL3_EMBED_GPU_PRUNED", "EXL3_EMBED_GPU_MAX_MB", "EXL3_MTP_DEVICE_DRAFT",
    "EXL3_MTP_HEAD_N", "EXL3_DRAFT_PINNED_STAGING", "EXL3_BATCH_VERIFY",
)


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {"__init__": lambda self, *a, **k: None})
        setattr(self, name, cls)
        return cls


class _StubFinder:
    """Any exllamav3.* module not loaded from a real file becomes an auto-stub package."""

    @staticmethod
    def find_spec(name, path = None, target = None):
        if name == "exllamav3" or name.startswith("exllamav3."):
            return importlib.util.spec_from_loader(name, _StubFinder, is_package = True)
        return None

    @staticmethod
    def create_module(spec):
        return _Stub(spec.name)

    @staticmethod
    def exec_module(module):
        module.__path__ = []


if not any(f is _StubFinder for f in sys.meta_path):
    sys.meta_path.append(_StubFinder)


def _ensure_pkg(name: str):
    if name in sys.modules:
        return sys.modules[name]
    mod = _Stub(name)
    mod.__path__ = []
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(_ensure_pkg(parent), child, mod)
    return mod


def _load_file(name: str, path: pathlib.Path):
    parent = name.rpartition(".")[0]
    _ensure_pkg(parent)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[parent], name.rpartition(".")[2], mod)
    return mod


def _base_packages():
    for p in ("exllamav3", "exllamav3.util", "exllamav3.model", "exllamav3.modules", "exllamav3.tokenizer",
              "exllamav3.generator", "exllamav3.architecture", "exllamav3.cache", "exllamav3.modules.arch_specific"):
        _ensure_pkg(p)
    # Real torch-only helpers the loaded files depend on
    _load_file("exllamav3.util.device_copy", BASELINE / "util" / "device_copy.py")
    _load_file("exllamav3.util.tensor", BASELINE / "util" / "tensor.py")
    _load_file("exllamav3.modules.module", BASELINE / "modules" / "module.py")
    sys.modules["exllamav3.modules"].Module = sys.modules["exllamav3.modules.module"].Module
    mm = _ensure_pkg("exllamav3.tokenizer.mm_embedding")
    mm.FIRST_MM_EMBEDDING_INDEX = 1 << 30
    cst = _ensure_pkg("exllamav3.constants")
    cst.PAGE_SIZE = 256


def set_env(**flags):
    for k in FLAGS:
        os.environ.pop(k, None)
    for k, v in flags.items():
        os.environ[k] = str(v)


def load_embedding(root: str = "overlay", **flags):
    """Fresh exllamav3.modules.embedding (+ embedding_pruned) from the overlay or the baseline, under flags."""
    set_env(**flags)
    _base_packages()
    base = OVERLAY if root == "overlay" else BASELINE
    if (base / "modules" / "embedding_pruned.py").exists():
        _load_file("exllamav3.modules.embedding_pruned", base / "modules" / "embedding_pruned.py")
    mod = _load_file("exllamav3.modules.embedding", base / "modules" / "embedding.py")
    sys.modules["exllamav3.modules"].Embedding = mod.Embedding
    return mod


def load_pruned_helper():
    _base_packages()
    return _load_file("exllamav3.modules.embedding_pruned", OVERLAY / "modules" / "embedding_pruned.py")


def load_mtp(root: str = "overlay", **flags):
    set_env(**flags)
    _base_packages()
    base = OVERLAY if root == "overlay" else BASELINE
    return _load_file("exllamav3.architecture.qwen4_exp_mtp", base / "architecture" / "qwen4_exp_mtp.py")


def load_generator(root: str = "overlay", **flags):
    set_env(**flags)
    _base_packages()
    util = sys.modules["exllamav3.util"]
    util.cuda_sync_active = lambda: None
    util.profile_opt = lambda *a, **k: (lambda f: f)
    base = OVERLAY if root == "overlay" else BASELINE
    return _load_file("exllamav3.generator.generator", base / "generator" / "generator.py")
