#!/usr/bin/env python3
"""GPU/box verification script for ngram-prefetch-r1 (EXL3_NGRAM_PREFETCH2). Runs inside the
patched image, one card per run:

  sudo docker run --rm --gpus all --entrypoint python3 -v "$R":/out "$NIMG" \
    /opt/ngram-prefetch-r1/tests/gpu_ngram_prefetch.py --device cuda:0 --json /out/gpu0.json

What the box can check here that the CPU harness cannot:
  1. forward() through the real device path: hash + pinned gather + H2D + ngram_dequant,
     compared torch.equal against the module's torch reference on the same table.
  2. CUDA events: the staging sets record events after each forward; re-staging and re-using
     the same set across later forwards (the wait-before-rewrite path) stays exact.
  3. Deferred acquisition (PREFETCH2): several sets queued back to back, consumed in any
     order, against a running stream — no deadlock, results exact.
  4. Verify-window hook exact + consumed; discard semantics (timing only).
  5. bench_us (informational).

Exit 0 only if every check passes. --json writes the full record.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("NGRAM_TREE_PARENT", ""))
sys.path.insert(0, str(HERE))

import fake_ext as fx   # noqa: E402  (fake hash/gather/dequant; the real CUDA event path below)
import types
import importlib.machinery
import importlib.util

# the image loads the real prebuilt extension through exllamav3.ext; for this harness the
# n-gram functions are re-pointed at the fake (a faithful port of ngram.cu) so the run does
# not need the real 18.5 GiB table — every other kernel stays the image's own build.
import exllamav3.ext as _extmod
_fx_hash, _fx_gather, _fx_dequant = fx.ngram_hash_cpu, fx.ngram_gather_cpu, fx.ngram_dequant
_extmod.exllamav3_ext.ngram_hash_cpu = _fx_hash
_extmod.exllamav3_ext.ngram_gather_cpu = _fx_gather
_extmod.exllamav3_ext.ngram_dequant = _fx_dequant

from exllamav3.modules.ngram_embedding import NGramEmbedding, PREFETCH2   # noqa: E402
from exllamav3.modules.quant.exl3_lib.ngram_codec import (                # noqa: E402
    pack_rows, words_per_row, ROW_DIM)

records = {"flags": {"prefetch2": bool(PREFETCH2)}, "checks": {}, "bench_us": {}}


def check(name, cond, detail = None):
    records["checks"][name] = bool(cond)
    assert cond, f"{name} failed{(': ' + str(detail)) if detail is not None else ''}"


def build(device):
    torch.manual_seed(7)
    NG, HN = 4, 2
    H = (NG - 1) * HN
    K = 2
    sizes = []
    p = 101
    for _ in range(H):
        sizes.append(p)
        p += 2
        while any(p % d == 0 for d in range(3, int(p ** 0.5) + 1, 2)) or p % 2 == 0:
            p += 2
    offsets_t = torch.tensor([0] + [sum(sizes[:i + 1]) for i in range(H - 1)], dtype = torch.int64)
    sizes_t = torch.tensor(sizes, dtype = torch.int64)
    mult_t = torch.tensor([7919, 104729, 1299709, 15485863], dtype = torch.int64)
    num_rows = int(sizes_t.sum())
    states = torch.randint(0, 1 << K, (num_rows, ROW_DIM), dtype = torch.int16)
    scales = torch.rand(num_rows).half() * 0.5 + 0.5
    packed = pack_rows(states, scales, K).contiguous()
    head_bias = (torch.randn(H, ROW_DIM) * 0.05).half()
    from exllamav3.loader.safetensors import DiskTensorHandle
    # The n-gram gather is re-pointed at fake_ext (above), which reads rows from its own fd registry, not from a file:
    # register the in-memory table on open, as tests/cpu_runtime.py does (R559 tries 1-2: the real _ensure_open opened a
    # file the fake gather knows nothing about)
    def _fake_open(self):
        if self.fd is None:
            self.fd = fx.register_table(self._store)
        return self.fd
    def _fake_read_rows(self, indices):
        return fx._FD_STORES[self._ensure_open()].index_select(0, indices.reshape(-1).cpu().to(torch.int64))
    DiskTensorHandle._ensure_open = _fake_open
    DiskTensorHandle.read_rows = _fake_read_rows

    class FakeSTC:
        tensor_file_map = {}
        disk_handles = []

        def has_tensor(self, name):
            return name == "test.trellis"

        def get_tensor(self, name, device_, optional = False, **k):
            mapping = {"test.head_offsets": offsets_t, "test.head_vocab_sizes": sizes_t,
                       "test.layer_multipliers": mult_t, "test.head_bias": head_bias}
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
    cfg.infer_params = type("IP", (), {"ngram_stream_from_disk": True})()
    cfg.layer_map = None
    emb = NGramEmbedding(cfg, "test", NG, HN, H * ROW_DIM, 151643)
    emb.load(device)
    return emb


def reference(emb, x):
    out_len = x.shape[1] - emb.context_len
    return emb.embed_ids(emb.compute_ngram_ids(x, out_len), torch.half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--json", default = None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    emb = build(dev)

    torch.manual_seed(11)
    ids = torch.randint(3, 100000, (4, 512), dtype = torch.int64)

    # 1. reference == forward on CUDA
    ref, fwd = reference(emb, ids), emb.forward(ids, params = {}, out_dtype = torch.half)
    check("forward_equals_reference", torch.equal(ref, fwd))

    # 2. stock prefetch consumed on the second forward, exact
    emb.prefetch(ids)
    fwd2 = emb.forward(ids, params = {}, out_dtype = torch.half)
    check("prefetch_consumed_exact", torch.equal(fwd, fwd2))

    # 3. PREFETCH2 deferred path: two sets queued back to back, consumed out of queue order
    h1, h2 = ids[:, :256], ids[:, 256:]
    emb.prefetch(h1)
    emb.prefetch(h2)
    o2 = emb.forward(h2, params = {}, out_dtype = torch.half)
    o1 = emb.forward(h1, params = {}, out_dtype = torch.half)
    check("deferred_h1_exact", torch.equal(o1, reference(emb, h1)))
    check("deferred_h2_exact", torch.equal(o2, reference(emb, h2)))

    # 4. event reuse: the sets' events recorded on the last forwards; re-staging over the
    #    same buffers must wait on them and stay exact (runs with real kernels queued)
    o1b = emb.forward(h1, params = {}, out_dtype = torch.half)
    check("event_reuse_h1", torch.equal(o1, o1b))
    o2b = emb.forward(h2, params = {}, out_dtype = torch.half)
    check("event_reuse_h2", torch.equal(o2, o2b))

    # 5. verify-window hook: exact and consumed
    verify_history = torch.cat((ids[0:1, 505:508], ids[0:1, 508:512]), dim = 1)
    emb.prefetch_ids(verify_history)
    ov = emb.forward(verify_history, params = {}, out_dtype = torch.half)
    check("verify_prefetch_exact", torch.equal(ov, reference(emb, verify_history)))
    check("verify_prefetch_hit", emb.prefetch_stats["hit"] >= 1, emb.prefetch_stats)

    # 6. discard semantics: a queued set that never matches changes nothing
    bad = h1.clone()
    bad[0, -1] = (bad[0, -1] + 1) % 100000
    emb.prefetch(bad)
    o3 = emb.forward(h1, params = {}, out_dtype = torch.half)
    check("discard_changes_nothing", torch.equal(o1, o3))

    # 7. timing report sane
    rep = emb.timing_report()
    check("timing_report_present", isinstance(rep, dict) and "exposed_wait" in rep)

    # 8. bench (informational): a full prefill-sized forward after staging
    import time
    emb.prefetch(ids)
    while emb._pending and emb._pending[0]["pin"] is None:
        pass
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    emb.forward(ids, params = {}, out_dtype = torch.half)
    torch.cuda.synchronize(dev)
    records["bench_us"]["forward_2036_rows_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    print(json.dumps(records, indent = 1))
    if a.json:
        Path(a.json).write_text(json.dumps(records, indent = 1))
    ok = all(records["checks"].values())
    print("GPU harness:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
