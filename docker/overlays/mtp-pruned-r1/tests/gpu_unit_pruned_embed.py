#!/usr/bin/env python3
"""
GPU unit test + lookup microbenchmark for EXL3_EMBED_GPU_PRUNED (operator-run, in the built image, ~1 min, no
server needed; needs ~1.7 GiB free on the mirror device for the full-mirror reference).

Loads ONLY the trunk's Embedding module of the checkpoint to the host (as served: prefer_cpu), then on the
mirror device (default cuda:1, the lm_head card):
  1. pruned mirror rows == host rows [0, N) bit for bit; VRAM added = torch.cuda.memory_allocated delta
  2. Embedding.forward through the pruned branch == through the full-mirror branch (the r4 unpruned path)
     == the host path, bitwise, for: in-set device IDs (idx >= 1 contract) and mixed batches with out-of-set
     verified tokens given their host copy (idx == 0 contract): N-1, N, vocab-1, special-token range
  3. the out-of-set path does not synchronize the host (checked with a pending CUDA sleep kernel)
  4. microbenchmark (CUDA events): per-lookup time for pruned device gather, host fallback upload,
     full-mirror gather and the host path, at 1 / 4 / 16 rows

The module flags are import-time constants that the code reads at call time, so this script flips the module
globals to select each path inside one process.
Exit status 0 = all checks passed. Writes a JSON report with --out.
"""
import argparse
import json
import os
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--model", default = "/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab")
p.add_argument("--device", default = "cuda:1")
p.add_argument("--head-n", type = int, default = int(os.environ.get("EXL3_MTP_HEAD_N", "65536")))
p.add_argument("--iters", type = int, default = 2000)
p.add_argument("--out", default = None)
args = p.parse_args()

from exllamav3 import Config, Model
import exllamav3.modules.embedding as emb_mod
import exllamav3.modules.embedding_pruned as ep
from _refs import raw_rows, forward_reference, describe   # tests/ is on sys.path (script directory)

dev = torch.device(args.device)
report = {"device": str(dev), "checks": {}, "bench_us": {}}
fails = []


def check(name, ok, detail = None):
    report["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f"  {detail}"))
    if not ok:
        fails.append(name)


from _refs import bits_equal


config = Config.from_directory(args.model)
model = Model.from_config(config)
embed = next(m for m in model.modules if isinstance(m, emb_mod.Embedding))
embed.load(torch.device("cpu"))
w = embed.embedding.weight
V, D = w.shape
# n_cols exactly as qwen4_exp_mtp._pruned_head computes it (whole 128-column tiles of the head)
N = min(args.head_n, V) // 128 * 128
report.update({"table": [V, D], "dtype": str(w.dtype), "n_keep": N})
print(f"embedding table {V} x {D} {w.dtype} on {w.device}; pruned rows {N}; mirror device {dev}")
check("table on host", w.device.type == "cpu")

keep = torch.arange(N, dtype = torch.long)


def set_mode(pruned: bool, gpu: bool = True):
    emb_mod._EMBED_GPU = gpu
    emb_mod._EMBED_GPU_PRUNED = pruned


def fwd(ids, host_ids = None, pruned = False):
    params = {}
    if pruned:
        params["embed_pruned"] = True
        if host_ids is not None:
            params["embed_pruned_host_ids"] = host_ids
    with torch.inference_mode():
        return embed.forward(ids, params, out_dtype = torch.half)


# 1. build + VRAM
set_mode(pruned = True)
torch.cuda.synchronize(dev)
a0 = torch.cuda.memory_allocated(dev)
f0 = torch.cuda.mem_get_info(dev)[0]
pm = embed.prepare_pruned_mirror(dev, keep)
torch.cuda.synchronize(dev)
a1 = torch.cuda.memory_allocated(dev)
f1 = torch.cuda.mem_get_info(dev)[0]
check("pruned mirror built", pm is not None and pm.prefix and pm.remap is None)
check("mirror rows exact", bits_equal(pm.mirror, w[:N]))
report["vram_pruned_bytes"] = a1 - a0
report["vram_pruned_free_delta_bytes"] = f0 - f1
check("VRAM added = N*D*2 bytes", a1 - a0 == N * D * w.element_size(), f"{(a1 - a0) / 2**20:.1f} MiB allocated, "
      f"{(f0 - f1) / 2**20:.1f} MiB free-delta")
check("full mirror refused in pruned mode", embed.can_embed_device_ids(dev) is False)

# 2. identity: pruned vs full-mirror vs host path
g = torch.Generator().manual_seed(20260919)
special = torch.arange(max(N, V - 400), V)                     # special-token range at the end of the vocab
cases = {
    "in_set_1": torch.randint(0, N, (1, 1), generator = g),
    "in_set_4": torch.randint(0, N, (4, 1), generator = g),
    "in_set_16": torch.randint(0, N, (16, 1), generator = g),
    "boundary": torch.tensor([[N - 1], [0], [N // 2], [1]]),
    "oos_boundary": torch.tensor([[N - 1], [N], [V - 1], [N + 1]]),
    "oos_special": special[torch.randint(0, special.numel(), (4, 1), generator = g)],
    "oos_mixed_16": torch.cat([torch.randint(0, N, (12, 1), generator = g), torch.randint(N, V, (4, 1), generator = g)]),
}
for name, h in cases.items():
    d = h.to(dev)
    set_mode(pruned = False)
    ref_full = fwd(d)                                       # r4 full-mirror path (builds the 1.27 GB mirror once)
    ref_host = fwd(h).to(dev)                               # host path (flags-off chain)
    set_mode(pruned = True)
    before = dict(ep.stats)
    in_set = bool((h < N).all())
    got = fwd(d, None if in_set else h, pruned = True)
    got_hinted = fwd(d, h, pruned = True)
    took_host = ep.stats["host_calls"] - before["host_calls"]
    check(f"{name}: pruned == full mirror", bits_equal(got, ref_full))
    check(f"{name}: pruned(hinted) == full mirror", bits_equal(got_hinted, ref_full))
    check(f"{name}: full mirror == host path", bits_equal(ref_full, ref_host))
    check(f"{name}: host fallback used iff out-of-set", took_host == (0 if in_set else 2), f"host_calls +{took_host}")
full_bytes = embed._gpu_mirror.numel() * embed._gpu_mirror.element_size() if embed._gpu_mirror is not None else 0
report["vram_full_mirror_bytes"] = full_bytes
embed._gpu_mirror = None
embed._gpu_mirror_device = None
torch.cuda.empty_cache()

# 3. out-of-set path under ~100 ms of queued device work.
# R522 (round 1 of this script) compared Embedding.forward's output (fp16: to2(x, out_dtype=half)) with the raw
# table rows (the table dtype, bf16 when the checkpoint stores it so) and failed on dtype alone. Each level is now
# compared with its own reference: raw rows via PrunedEmbeddingMirror.lookup, forward output via the full-mirror
# forward (device cast) and, informationally, via the host-cast forward_reference.
set_mode(pruned = True)
hA = torch.tensor([[N + 3], [5]])
hB = torch.tensor([[V - 1], [N]])                           # same shape as hA -> same pinned staging buffer
dA, dB = hA.to(dev), hB.to(dev)
set_mode(pruned = False)
ref_fwd_A = fwd(dA)                                        # full-mirror forward on dev (the r4 unpruned path)
set_mode(pruned = True)
torch.cuda.synchronize(dev)
# Same grad mode as the served path: Embedding.forward runs under the generator's inference_mode (fwd() above
# uses it too). R522 rerun: without it the round-1b section crashed in gather_host's index_select(out=) on the
# requires_grad table (fixed in embedding_pruned.stage_rows; see test_lookup_with_requires_grad_parameter).
with torch.inference_mode(), torch.cuda.device(dev):
    torch.cuda._sleep(200_000_000)                          # ~0.1 s of queued GPU work on dev's current stream
    t0 = time.perf_counter()
    rawA = pm.lookup(w, dA, hA)                             # raw rows: pinned staging -> async H2D behind the sleep
    t_host = time.perf_counter() - t0
    # Buffer-reuse race probe: the next out-of-set lookup of the same shape rewrites the same pinned buffer. It must
    # first wait for rawA's upload (still queued behind the sleep), or rawA would receive hB's rows.
    t1 = time.perf_counter()
    rawB = pm.lookup(w, dB, hB)
    t_reuse = time.perf_counter() - t1
    outA = fwd(dA, hA, pruned = True)                       # forward level: same path + to2(half)
    torch.cuda.synchronize(dev)
check("oos lookup does not wait for the device", t_host < 0.02, f"{t_host * 1e3:.2f} ms host time with ~100 ms queued")
check("oos staging reuse waits for the previous upload", t_reuse > 0.02,
      f"{t_reuse * 1e3:.2f} ms host time for the second lookup (expected ~ the queued sleep)")
check("oos raw rows exact after async upload (A)", bits_equal(rawA, raw_rows(w, hA)), describe(rawA, raw_rows(w, hA)))
check("oos raw rows exact after buffer reuse (B)", bits_equal(rawB, raw_rows(w, hB)), describe(rawB, raw_rows(w, hB)))
check("oos forward output == full-mirror forward", bits_equal(outA, ref_fwd_A), describe(outA, ref_fwd_A))
fr = forward_reference(w, hA)
report["checks"]["info: oos forward == host-cast reference"] = {"ok": bits_equal(outA, fr), "detail": describe(outA, fr)}
print(f"INFO oos forward == host-cast reference: {bits_equal(outA, fr)}  {describe(outA, fr)}")
embed._gpu_mirror = None
embed._gpu_mirror_device = None
torch.cuda.empty_cache()

# 4. microbenchmark
def bench(label, fn, iters):
    for _ in range(20):
        fn()
    torch.cuda.synchronize(dev)
    s, e = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    t0 = time.perf_counter()
    s.record(torch.cuda.current_stream(dev))
    for _ in range(iters):
        fn()
    e.record(torch.cuda.current_stream(dev))
    torch.cuda.synchronize(dev)
    wall = (time.perf_counter() - t0) / iters * 1e6
    gpu = s.elapsed_time(e) / iters * 1e3
    report["bench_us"][label] = {"wall": round(wall, 2), "gpu": round(gpu, 2)}
    print(f"  {label:34s} wall {wall:8.2f} us/lookup   gpu {gpu:8.2f} us/lookup")


print("microbenchmark (per lookup, includes the half cast of Embedding.forward)")
for rows in (1, 4, 16):
    hi = torch.randint(0, N, (rows, 1), generator = g)
    ho = hi.clone()
    ho[0, 0] = V - 1
    di, do = hi.to(dev), ho.to(dev)
    set_mode(pruned = True)
    bench(f"pruned device gather r{rows}", lambda: fwd(di, None, pruned = True), args.iters)
    bench(f"pruned idx0 in-set (host check) r{rows}", lambda: fwd(di, hi, pruned = True), args.iters)
    bench(f"pruned idx0 out-of-set upload r{rows}", lambda: fwd(do, ho, pruned = True), args.iters)
    set_mode(pruned = False)
    bench(f"full mirror gather r{rows}", lambda: fwd(di), args.iters)
    bench(f"host path + upload r{rows}", lambda: fwd(hi).to(dev), args.iters // 4)
embed._gpu_mirror = None
torch.cuda.empty_cache()

report["stats"] = dict(ep.stats)
report["ok"] = not fails
if args.out:
    with open(args.out, "w") as f:
        json.dump(report, f, indent = 1)
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(0 if not fails else 1)
