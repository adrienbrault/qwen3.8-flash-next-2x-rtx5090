"""
CPU test harness for the NVMe tier (round 3).

Loads the REAL overlay modules (generator/disk_cache.py, generator/pagetable.py, cache/recurrent.py and, in stacked
mode, recurrent-tip-r1's cache/recurrent_tip.py) over the served src/ tree, without importing the exllamav3 package
__init__ (which needs the CUDA extension). Only leaf modules that pull in the extension are stubbed: exllamav3.ext,
cache.cache (Cache/CacheLayer are only used for type hints by pagetable) and util.memory.malloc_trim.

Layout selection (one per process):
  NVME_STACKED=0 (default)  overlay/ over src/                                  = stack-r4-e3r2 + this round
  NVME_STACKED=1            overlay-tip/ over overlay/ over recurrent-tip-r1 over src/
                                                                              = stack-r4-e3r2 + tip r1 + this round
  NVME_TEST_TREE=<dir>      a single installed exllamav3 directory (e.g. site-packages/exllamav3 in the image)

MiniEngine drives the real PageTable / Sequence / RecurrentCache (and TipPolicyMixin when stacked) with CPU "cache
tensors" whose page contents are a deterministic function of the page's content hash, so every restored page and
checkpoint can be compared bit for bit with what a cold run would have written.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import pathlib
import sys
import time
import types

import torch

HERE = pathlib.Path(__file__).resolve().parent
DELIV = HERE.parent
ROOT = DELIV.parent.parent
SRC = ROOT / "src" / "exllamav3"
OVL = DELIV / "overlay" / "exllamav3"
OVL_TIP = DELIV / "overlay-tip" / "exllamav3"
TIP = ROOT / "ref" / "recurrent-tip-r1" / "overlay" / "exllamav3"
PAGE_SIZE = 256

STACKED = os.environ.get("NVME_STACKED", "0") == "1"
TREE = os.environ.get("NVME_TEST_TREE")


def tree_paths() -> list[pathlib.Path]:
    if TREE:
        return [pathlib.Path(TREE)]
    if STACKED:
        return [OVL_TIP, OVL, TIP, SRC]
    return [OVL, SRC]


class _M:
    pass


M = _M()


def load():
    """Import the modules under test once per process; returns a namespace with them."""
    if getattr(M, "loaded", False):
        return M
    paths = [str(p) for p in tree_paths()]

    def pkg(name, sub):
        mod = types.ModuleType(name)
        mod.__path__ = [str(pathlib.Path(p) / sub) if sub else p for p in paths]
        sys.modules[name] = mod
        return mod

    root = pkg("exllamav3", "")
    pkg("exllamav3.generator", "generator")
    cache_pkg = pkg("exllamav3.cache", "cache")
    util = pkg("exllamav3.util", "util")
    util.profile_opt = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    pkg("exllamav3.tokenizer", "tokenizer")
    mem = types.ModuleType("exllamav3.util.memory")
    mem.malloc_trim = lambda: None
    sys.modules["exllamav3.util.memory"] = mem
    ext = types.ModuleType("exllamav3.ext")
    ext.exllamav3_ext = None
    sys.modules["exllamav3.ext"] = ext
    cc = types.ModuleType("exllamav3.cache.cache")
    cc.Cache = object
    cc.CacheLayer = object
    sys.modules["exllamav3.cache.cache"] = cc

    M.constants = importlib.import_module("exllamav3.constants")
    M.recurrent = importlib.import_module("exllamav3.cache.recurrent")
    cache_pkg.RecurrentCache = M.recurrent.RecurrentCache
    M.pagetable = importlib.import_module("exllamav3.generator.pagetable")
    M.disk_cache = importlib.import_module("exllamav3.generator.disk_cache")
    M.mm = importlib.import_module("exllamav3.tokenizer.mm_embedding")
    M.recurrent_tip = importlib.import_module("exllamav3.cache.recurrent_tip") if (STACKED or TREE) and \
        any((pathlib.Path(p) / "cache" / "recurrent_tip.py").exists() for p in paths) else None
    M.loaded = True
    return M


def module_file(name: str) -> pathlib.Path:
    return pathlib.Path(sys.modules[name].__file__).resolve()


# --------------------------------------------------------------------------------------------------------------
# Fake caches / model
# --------------------------------------------------------------------------------------------------------------

class FakeConfig:
    arch_string = "FakeHybrid"

    def __init__(self, directory):
        self.directory = str(directory)


class FakeModel:
    loaded_tp = False

    def __init__(self, directory):
        self.config = FakeConfig(directory)


class FakePagedLayer:
    """A full-attention cache layer: page-major tensors like CacheLayer_quant (qk, qv int32; sk, sv fp16)."""

    def __init__(self, pages: int, dims: list[tuple[int, torch.dtype]]):
        self.k_bits = 8
        self.v_bits = 8
        self.tensors = [torch.zeros((pages, PAGE_SIZE, d), dtype = dt) for d, dt in dims]

    def get_tensors(self):
        return self.tensors


class FakeRecurrentLayer:
    def __init__(self, n):
        self.n = n

    def get_checkpoint_size(self):
        return self.n


class FakeCache:
    def __init__(self, model_dir, pages: int, layers: int = 2, draft: bool = False):
        self.model = FakeModel(model_dir)
        self.max_num_tokens = pages * PAGE_SIZE
        dims = [(8, torch.int32), (8, torch.int32), (2, torch.float16), (2, torch.float16)]
        if draft:
            dims = [(4, torch.bfloat16), (4, torch.bfloat16)]
        self.layers = {(i, 0): FakePagedLayer(pages, dims) for i in range(layers)}
        self.recurrent_layers = {} if draft else {(i, 0): FakeRecurrentLayer(1000 + i) for i in range(3)}


class FakeGenerator:
    tokenizer = None

    def __init__(self):
        self.recurrent_cache = None


def seeded(h: bytes, salt: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed((int.from_bytes(hashlib.blake2b(h + bytes([salt]), digest_size = 8).digest(), "little")) & ((1 << 63) - 1))
    return g


def page_value(h: bytes, idx: int, shape, dtype) -> torch.Tensor:
    g = seeded(h, idx)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-2**31, 2**31 - 1, shape, generator = g, dtype = torch.int64).to(dtype)
    return torch.randn(shape, generator = g).to(dtype)


def state_value(h: bytes, position: int) -> dict:
    """Deterministic stash dict in GDNState.stash()'s layout: position, checkpoint_size, layer key -> (fp32, bf16)."""
    out = {"position": position, "checkpoint_size": 0}
    size = 0
    for i in range(3):
        g = seeded(h, 100 + i)
        rs = torch.randn((1, 2, 3, 4), generator = g, dtype = torch.float32)
        cs = torch.randn((6, 4), generator = g).to(torch.bfloat16)
        out[(i, 0)] = (rs, cs)
        size += rs.numel() * 4 + cs.numel() * 2
    out["checkpoint_size"] = size
    return out


class StashState:
    """Stand-in for a live recurrent state handed to RecurrentCache.put()."""

    def __init__(self, stashed):
        self.stashed = stashed
        self.position = stashed["position"]
        self.checkpoint_size = stashed["checkpoint_size"]

    def stash(self):
        return self.stashed


def same_stash(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, tuple):
            if len(va) != len(vb):
                return False
            for x, y in zip(va, vb):
                if x.dtype != y.dtype or x.shape != y.shape or not torch.equal(x.view(torch.uint8) if x.dtype != torch.uint8 else x,
                                                                                y.view(torch.uint8) if y.dtype != torch.uint8 else y):
                    return False
        elif va != vb:
            return False
    return True


# --------------------------------------------------------------------------------------------------------------
# MiniEngine: the real page table + recurrent cache + tier, driven like Generator/Job drive them
# --------------------------------------------------------------------------------------------------------------

class MiniEngine:

    def __init__(
        self,
        tier_dir,
        model_dir,
        pages: int = 96,
        cap: int = 256 * 1024**2,
        min_pages: int = 4,
        rc_size: int = 10**9,
        segment: int = 4 * 1024**2,
        staging: int = 8 * 1024**2,
        pump_pages: int = 4,
        tier: bool = True,
        tip_policy: bool = False,
        mtp: bool = True,
        scan: bool = False,
    ):
        m = load()
        self.m = m
        self.cache = FakeCache(model_dir, pages)
        self.draft = FakeCache(model_dir, pages, layers = 1, draft = True)
        self.gen = FakeGenerator()
        self.pt = m.pagetable.PageTable(self.gen, self.cache)
        self.rc = m.recurrent.RecurrentCache(self.cache.model, rc_size)
        self.rc.pagetable = self.pt
        self.gen.recurrent_cache = self.rc
        self.gen.pagetable = self.pt
        self.mtp = mtp
        if tip_policy:
            m.recurrent_tip.install_tip_policy(self.rc)
        self.tier = None
        if tier:
            self.tier = m.disk_cache.DiskPageCache(
                [self.cache, self.draft], str(tier_dir), cap, min_free_pct = 0.0, segment_size = segment,
                staging_size = staging, admission_min_pages = min_pages, require_dedicated_filesystem = False,
                pump_pages = pump_pages, engine = {"test": "engine"}, log_secs = 0, scan_checkpoints = scan,
            )
            self.tier.attach(self.pt, self.rc)
            self.pt.disk_tier = self.tier
            self.rc.disk_tier = self.tier
        self.tensors = [t for c in (self.cache, self.draft) for l in c.layers.values() for t in l.get_tensors()]

    # -- Generator hooks --

    def iterate_start(self):
        if self.tier is not None:
            self.tier.pump()

    def idle(self):
        if self.tier is not None:
            self.tier.on_busy()
            self.tier.on_idle()

    def wait_drained(self, timeout = 20.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.tier.pending_copies() == 0 and self.tier._write_queue.unfinished_tasks == 0:
                return True
            time.sleep(0.005)
        return False

    # -- one request --

    def write_page(self, page, h: bytes):
        for i, t in enumerate(self.tensors):
            t[page.page_index].copy_(page_value(h, i, t.shape[1:], t.dtype))

    def check_page(self, page, h: bytes) -> bool:
        for i, t in enumerate(self.tensors):
            if not torch.equal(t[page.page_index], page_value(h, i, t.shape[1:], t.dtype)):
                return False
        return True

    def run(self, tokens: list[int], checkpoints: list[int] | None = None, pump_after: bool = True) -> dict:
        """Allocate like Job.allocate_pages, verify every reused page and the restored state bit for bit, "prefill"
        the rest (write the deterministic page contents, link and complete pages like job.prefill), stash
        checkpoints (default: the prompt-end checkpoint job.py takes at the last full page), release the pages."""
        m = self.m
        self.iterate_start()
        ids = torch.tensor([tokens], dtype = torch.long)
        seq = m.pagetable.Sequence(ids, ids.clone())
        seq.prepare(False, 16)
        if self.mtp:
            seq.max_cached_pages = max(0, (len(tokens) - 2) // PAGE_SIZE)
        _, cached_pages, _, stashed = seq.allocate_pages(self.pt, self.rc)
        pages = seq.allocated_pages
        prompt_pages = (len(tokens) - 1) // PAGE_SIZE
        bad_pages = [i for i in range(cached_pages) if not self.check_page(pages[i], seq.page_hashes[i])]
        state_ok = None
        if cached_pages:
            expect = state_value(seq.page_hashes[cached_pages - 1], cached_pages * PAGE_SIZE)
            state_ok = stashed is not None and same_stash(stashed, expect)
        for i in range(cached_pages, prompt_pages):
            p = pages[i]
            h = seq.page_hashes[i]
            self.write_page(p, h)
            p.sequence[0].copy_(ids[0, i * PAGE_SIZE:(i + 1) * PAGE_SIZE])
            p.kv_position = PAGE_SIZE
            p.prev_hash = None if i == 0 else pages[i - 1].phash
            assert p.phash == h
        if checkpoints is None:
            checkpoints = [prompt_pages] if prompt_pages else []
        for n in checkpoints:
            h = seq.page_hashes[n - 1]
            self.rc.put(h, StashState(state_value(h, n * PAGE_SIZE)))
        self.pt.deallocate_pages(pages)
        if pump_after:
            self.iterate_start()
        return {
            "cached_pages": cached_pages,
            "bad_pages": bad_pages,
            "state_ok": state_ok,
            "page_hashes": seq.page_hashes,
            "prompt_pages": prompt_pages,
        }

    def close(self):
        if self.tier is not None:
            self.tier.close()


def tokens_for(seed: int, n: int) -> list[int]:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, 150000, (n,), generator = g).tolist()


def model_dir(tmp) -> pathlib.Path:
    d = pathlib.Path(tmp) / "model"
    d.mkdir(parents = True, exist_ok = True)
    (d / "config.json").write_text('{"arch": "fake", "layers": 2}')
    (d / "model.safetensors").write_bytes(b"x" * 200000)
    return d
