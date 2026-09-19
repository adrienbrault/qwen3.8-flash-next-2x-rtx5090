#!/usr/bin/env python3
"""CPU scenario runner for ngram-prefetch-r1: exercises the patched NGramEmbedding end to end
with the real package code and a CPU stand-in for the n-gram extension functions.

Usage: python3 cpu_runtime.py <patched-tree-parent> <output.json>
Env: EXL3_NGRAM_PREFETCH2 / EXL3_NGRAM_TIMING are read by the caller's environment.

What it does:
  1. Builds a small synthetic trellis table (K=2, 6 heads, eos segmentation exercised) and
     loads NGramEmbedding from a fake SafetensorsCollection in trellis_disk mode.
  2. Runs the scenarios, recording per-scenario records:
     - forward == torch reference (compute_ngram_ids + fetch_rows) for prefill and decode
       histories, exercising eos-boundary segmentation;
     - prefetch off / prefetch on / double-deferred PREFETCH2 staging all give identical rows;
     - a queued set that never matches is retired and only costs time (discard semantics);
     - prefetch_ids builds the same history forward() builds and is consumed (hit);
     - the timing instrument records prefill/decode waits and timing_report() aggregates;
     - the flag-off paths stay inert (_timing empty, no deferred entries).
Exit 0 on pass.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

TREE_PARENT = sys.argv[1]
OUT = sys.argv[2]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # fake_ext
sys.path.insert(0, TREE_PARENT)                                  # package root parent

# ---- stub modules (fake precompiled extension + triton) ----------------------------------------
import importlib.machinery
import importlib.util
import fake_ext as fx


def _missing(name):
    if name.startswith("__"):
        raise AttributeError(name)
    def _stub(*a, **k):
        raise RuntimeError(f"stub {name} called")
    return _stub


def _deco(name):
    def d(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def wrap(fn):
            return fn
        return wrap
    return d


def stub_module(name, suffix = ".so", attrfn = _missing):
    m = types.ModuleType(name)
    spec = importlib.util.spec_from_file_location(name, f"/tmp/fake_{name}{suffix}")
    m.__spec__ = spec
    m.__file__ = f"/tmp/fake_{name}{suffix}"
    m.__path__ = [f"/tmp/fake_{name}"]
    m.__getattr__ = attrfn
    sys.modules[name] = m
    return m


ext_stub = stub_module("exllamav3_ext")
ext_stub.__spec__.loader = importlib.machinery.ExtensionFileLoader(
    "exllamav3_ext", "/tmp/fake_exllamav3_ext.cpython-darwin.so")
for fn in ("ngram_hash_cpu", "ngram_gather_cpu", "ngram_dequant"):
    setattr(ext_stub, fn, getattr(fx, fn))

triton = stub_module("triton", suffix = "", attrfn = _deco)
triton_lang = stub_module("triton.language", suffix = "", attrfn = _deco)
triton.language = triton_lang

# ---- import the real (patched) package ----------------------------------------------------------
import torch

from exllamav3.loader.safetensors import DiskTensorHandle


def _fake_open(self):
    if self.fd is None:
        self.fd = fx.register_table(self._store)
    return self.fd
DiskTensorHandle._ensure_open = _fake_open


def _fake_read_rows(self, indices):
    idx = indices.reshape(-1).cpu().to(torch.int64)
    return fx._FD_STORES[self._ensure_open()].index_select(0, idx)
DiskTensorHandle.read_rows = _fake_read_rows

from exllamav3.modules.ngram_embedding import (
    NGramEmbedding, PREFETCH2, PREFETCH_ENABLED, PREFETCH_MIN_TOKENS)
from exllamav3.modules.quant.exl3_lib.ngram_codec import (
    ROW_DIM, mul1_codebook, pack_rows, words_per_row)

# ---- build a small trellis table ---------------------------------------------------------------
torch.manual_seed(7)
NGRAM_SIZE, HEADS_PER_NGRAM = 4, 2
NUM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
K = 2
ROW_WORDS = words_per_row(K)

sizes = []
p = 101
for _ in range(NUM_HEADS):
    sizes.append(p)
    p += 2
    while any(p % d == 0 for d in range(3, int(p ** 0.5) + 1, 2)) or p % 2 == 0:
        p += 2
sizes_t = torch.tensor(sizes, dtype = torch.int64)
offsets_t = torch.tensor([0] + [sum(sizes[:i + 1]) for i in range(NUM_HEADS - 1)],
                         dtype = torch.int64)
mult_t = torch.tensor([7919, 104729, 1299709, 15485863], dtype = torch.int64)
NUM_ROWS = int(sizes_t.sum())
EOS = 151643
VOCAB = 100000

states = torch.randint(0, 1 << K, (NUM_ROWS, ROW_DIM), dtype = torch.int16)
scales = torch.rand(NUM_ROWS).half() * 0.5 + 0.5
packed = pack_rows(states, scales, K).contiguous()
head_bias = (torch.randn(NUM_HEADS, ROW_DIM) * 0.05).half()


class FakeSTC:
    def __init__(self):
        self.tensor_file_map = {}
        self.disk_handles = []

    def has_tensor(self, name):
        return name == "test.trellis"

    def get_tensor(self, name, device, optional = False, **k):
        mapping = {
            "test.head_offsets": offsets_t,
            "test.head_vocab_sizes": sizes_t,
            "test.layer_multipliers": mult_t,
            "test.head_bias": head_bias,
        }
        t = mapping.get(name)
        if t is None and not optional:
            raise KeyError(name)
        return None if t is None else t.clone()

    def get_tensor_handle(self, key):
        h = DiskTensorHandle(key = key, filename = "fake.safetensors", abs_offset = 0,
                             shape = [packed.shape[0], packed.shape[1]], dtype = torch.int16)
        h._store = packed
        return h

    def get_tensor_meta(self, key):
        return {key: {"shape": [packed.shape[0], packed.shape[1]]}}

    def release_file(self, f):
        pass

    def find_stc(self, key):
        return self


class Cfg:
    pass


cfg = Cfg()
cfg.stc = FakeSTC()
cfg.infer_params = types.SimpleNamespace(ngram_stream_from_disk = True)
cfg.layer_map = None

emb = NGramEmbedding(cfg, "test", NGRAM_SIZE, HEADS_PER_NGRAM, NUM_HEADS * ROW_DIM, EOS)
emb.load(torch.device("cpu"))
assert emb.mode == "trellis_disk" and emb.K == K, (emb.mode, emb.K)

records = {}
stats = lambda: dict(emb.prefetch_stats)


def reference(x: torch.Tensor) -> torch.Tensor:
    """fetch_rows-based reference output for the history x (bsz, ctx + seq)."""
    out_len = x.shape[1] - emb.context_len
    ngram_ids = emb.compute_ngram_ids(x, out_len)
    return emb.embed_ids(ngram_ids, torch.half)


def forward(x: torch.Tensor) -> torch.Tensor:
    return emb.forward(x, params = {}, out_dtype = torch.half)


# random histories: prefill-sized (>= PREFETCH_MIN_TOKENS) and decode-sized, with eos in the
# middle to exercise segmentation. Keep ids positive and products below 2^63 (like the real
# table's arithmetic on real token ids and prime multipliers).
torch.manual_seed(11)
ids_pool = torch.randint(3, VOCAB, (4, 512), dtype = torch.int64)
ids_pool[1, 100] = EOS
ids_pool[1, 101] = EOS        # adjacent eos positions
ids_pool[2, 0] = EOS
ids_pool[3, 511] = EOS

# ---- 1. forward == reference, no prefetch --------------------------------------------------------
prefill_hist = ids_pool[:, :256]                     # bsz 4, ctx 3, seq 253
decode_hist = ids_pool[1, 200:204].view(1, 4)        # out_len 1 (stateless position-0 path)
out_ref_p, out_fwd_p = reference(prefill_hist), forward(prefill_hist)
out_ref_d, out_fwd_d = reference(decode_hist), forward(decode_hist)
records["forward_equals_reference"] = {
    "prefill": bool(torch.equal(out_ref_p, out_fwd_p)),
    "decode": bool(torch.equal(out_ref_d, out_fwd_d := forward(decode_hist))),
}
assert records["forward_equals_reference"]["prefill"] and records["forward_equals_reference"]["decode"]

# ---- 2. prefetch on == prefetch off (queued then consumed) ---------------------------------------
hist = ids_pool[:, :512]                             # bsz 4, ctx 3, seq 509
out_plain = forward(hist)
emb.prefetch(hist)                                   # worker stages behind the forward's call
out_with = forward(hist)
records["prefetch_off_vs_on"] = {
    "equal": bool(torch.equal(out_plain, out_with)),
    "hit": emb.prefetch_stats["hit"],
}
assert records["prefetch_off_vs_on"]["equal"], "prefetch must not change results"
assert emb.prefetch_stats["hit"] >= 1 or not PREFETCH_ENABLED, "staged set must be consumed"

# ---- 3. discard: queued history that never matches ------------------------------------------------
hist_wrong = hist.clone()
hist_wrong[0, 0] = (hist_wrong[0, 0] + 1) % 100000
emb.prefetch(hist_wrong)
import time as _t
while emb._pending and emb._pending[0]["pin"] is None:
    _t.sleep(0.01)                                   # deferred acquisition may still be running
out_discard = forward(hist)
records["discard_mismatch"] = {
    "equal": bool(torch.equal(out_plain, out_discard)),
}
assert records["discard_mismatch"]["equal"], "discard must not change results"

# A stale set is not actively retired by a mismatching forward (served semantics: it is only
# evicted by _acquire_pin when the staging sets run out). Queue two more histories so the
# acquire path must retire the stale set, then check both new forwards stay correct.
h1 = ids_pool[:, :256]
h2 = ids_pool[:, 256:512]
emb.prefetch(h1)
emb.prefetch(h2)
out_h1, out_h2 = forward(h1), forward(h2)
records["discard_mismatch"].update({
    "stale_evicted": emb.prefetch_stats["retired"] >= 1,
    "h1_equal": bool(torch.equal(out_h1, reference(h1))),
    "h2_equal": bool(torch.equal(out_h2, reference(h2))),
    "pins_bounded": len(emb._pins) <= 2,
})
assert (records["discard_mismatch"]["stale_evicted"] or not PREFETCH_ENABLED), \
    "stale set must be evicted by the acquire path"
assert records["discard_mismatch"]["h1_equal"] and records["discard_mismatch"]["h2_equal"]
assert records["discard_mismatch"]["pins_bounded"], "staging memory must stay bounded"

# ---- 4. verify-window prefetch_ids: same history forward builds -----------------------------------
# simulate the generator: a verify history = (ctx carried ids) + (1 + window) ids
verify_ids = ids_pool[0, 508:512].view(1, 4)         # 1 + 3 tokens
carried = ids_pool[0:1, 505:508]                     # last ctx ids before the window
verify_history = torch.cat((carried, verify_ids), dim = 1)
emb.prefetch_ids(verify_history)
out_verify = forward(verify_history)
ref_verify = reference(verify_history)
records["verify_prefetch"] = {
    "equal_reference": bool(torch.equal(out_verify, ref_verify)),
    "hit": emb.prefetch_stats["hit"],
}
assert records["verify_prefetch"]["equal_reference"]
assert (records["verify_prefetch"]["hit"] >= 1) or (not PREFETCH2 and not PREFETCH_ENABLED), \
    "verify history must be staged and consumed"

# ---- 5. retired stale sets keep the staging bound under repeated mismatching prefetches ----------
for i in range(6):
    bad = ids_pool[:, :256].clone()
    bad[0, -1] = (bad[0, -1] + 1 + i) % VOCAB      # distinct each round: 6 distinct stale sets
    emb.prefetch(bad)
good = ids_pool[:, :256]
emb.prefetch(good)
out_good = forward(good)
records["pressure"] = {
    "equal": bool(torch.equal(out_good, reference(good))),
    "retired_total": emb.prefetch_stats["retired"],
    "pins_bounded": len(emb._pins) <= 2,
    "pending_small": len(emb._pending) <= 2,
}
assert records["pressure"]["equal"] and records["pressure"]["pins_bounded"]

# ---- 6. timing instrument -------------------------------------------------------------------------
rep = emb.timing_report()
cells = [c for c in rep["exposed_wait"].values() if c]
records["timing_report"] = {
    "forwards_seen": rep["forwards_seen"],
    "has_prefill": rep["exposed_wait"]["prefill"] is not None,
    "has_decode": rep["exposed_wait"]["decode"] is not None,
    "nonneg": all(v >= 0 for c in rep["exposed_wait"].values() if c
                  for v in (c["mean_ms"], c["p50_ms"], c["p90_ms"], c["max_ms"])),
}
print(json.dumps(rep, indent = 1), file = sys.stderr)

# ---- 7. flag-off inertia checks --------------------------------------------------------------------
records["flags"] = {"prefetch2": bool(PREFETCH2)}
timing_active = (os.environ.get("EXL3_NGRAM_TIMING", "0") == "1"
                 or os.environ.get("EXL3_NGRAM_PREFETCH2", "0") == "1")
if not timing_active:
    records["flag_off_inert"] = {"timing_entries": len(emb._timing), "timing_seen": emb._timing_n}
    assert len(emb._timing) == 0 and emb._timing_n == 0

records["staging"] = dict(emb.prefetch_stats)
with open(OUT, "w") as f:
    json.dump({"ok": True, "records": records}, f, indent = 1)
print("all scenarios pass")
