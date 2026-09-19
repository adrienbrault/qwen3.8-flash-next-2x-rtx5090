#!/usr/bin/env python3
"""TP=2 bounding probe, part 2a: NCCL all-reduce latency between the two cards (2 processes).

    NCCL_P2P_LEVEL=SYS NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/results/nccl.%h.%p.log \
      python3 -m torch.distributed.run --standalone --nproc_per_node=2 tp_bound_ar.py --out /results/tp_bound_ar.json

Payload = the tensor TP=2 reduces in this model: the sublayer output, rows x hidden 2560 (x 4 = 10240 for the variant
that reduces the four hyper-connection streams), fp32 (the modules' out_dtype) and fp16/bf16, rows 1/4/8/16
(c1 draft block / c1 d3 verify and c4 draft block / c4 d1 / c4 d3 verify). Eager and inside a CUDA graph (the served
decode runs as graphs). The ref/ar_bench.py (R184) method: warm on the current stream, time `iters` back-to-back calls,
median of repeats, group max of the per-rank medians.

Validation: every cell first reduces a pattern (rank r holds (i % 17) + 3 r, exact in every dtype) and checks the sum
exactly, eager and once after a graph replay; timed calls run on zeros (in-place reduction of zeros stays finite).
Transport: after the run rank 0 parses the NCCL debug file(s) and requires every channel to connect "via P2P";
"via SHM" or "via NET" -> the run FAILS (exit 3): P2P is the premise of the TP budget.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="1,4,8,16")
    ap.add_argument("--widths", default="2560,10240")
    ap.add_argument("--dtypes", default="fp32,fp16,bf16")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--graph-iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default="tp_bound_ar.json")
    ap.add_argument("--nccl-log-glob", default=None, help="default: derived from NCCL_DEBUG_FILE")
    return ap.parse_args(argv)


def transport_lines(paths):
    lines = []
    for p in paths:
        try:
            with open(p, errors="replace") as f:
                lines += [ln.strip() for ln in f if " via " in ln and "Channel" in ln]
        except OSError:
            pass
    return lines


def judge_transport(lines):
    """-> (ok, summary). ok only if there is at least one channel line and every one is P2P."""
    kinds = {}
    for ln in lines:
        m = re.search(r" via (\S+)", ln)
        if m:
            k = m.group(1).split("/")[0]
            kinds[k] = kinds.get(k, 0) + 1
    ok = bool(kinds) and set(kinds) == {"P2P"}
    return ok, kinds


def _nccl_version(torch):
    try:
        v = torch.cuda.nccl.version()
        return ".".join(str(x) for x in v) if isinstance(v, tuple) else str(v)
    except Exception as e:  # noqa: BLE001
        return repr(e)


def pattern(torch, n, rank, dtype, device):
    return ((torch.arange(n, device=device) % 17) + 3 * rank).to(dtype)


def main(argv=None) -> int:
    a = parse_args(argv)
    import torch
    import torch.distributed as dist
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    try:
        dist.init_process_group("nccl", device_id=dev)
    except TypeError:            # torch < 2.3 has no device_id
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        print("FATAL: expected 2 ranks", file=sys.stderr)
        return 2

    def gmax(v):
        t = torch.tensor([v], dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def gmin(v):
        t = torch.tensor([v], dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        return float(t.item())

    rows = [int(x) for x in a.rows.split(",")]
    widths = [int(x) for x in a.widths.split(",")]
    dts = a.dtypes.split(",")
    results = []
    t0 = time.time()
    for dname in dts:
        dtype = getattr(torch, DTYPES[dname])
        for w in widths:
            for r in rows:
                n = r * w
                # --- validation (eager) ---
                x = pattern(torch, n, rank, dtype, dev).view(r, w)
                dist.all_reduce(x)
                torch.cuda.synchronize()
                exp = (2 * (torch.arange(n, device=dev) % 17) + 3).to(dtype).view(r, w)
                ok_eager = bool(torch.equal(x, exp))
                # --- eager timing on zeros ---
                z = torch.zeros((r, w), dtype=dtype, device=dev)
                for _ in range(20):
                    dist.all_reduce(z)
                torch.cuda.synchronize()
                samples = []
                for _ in range(a.repeats):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    dist.barrier()
                    torch.cuda.synchronize()
                    s.record()
                    for _ in range(a.iters):
                        dist.all_reduce(z)
                    e.record()
                    torch.cuda.synchronize()
                    samples.append(s.elapsed_time(e) * 1000.0 / a.iters)
                med = statistics.median(samples)
                results.append({"method": "nccl", "mode": "eager", "rows": r, "width": w, "dtype": dname,
                                "median_us": round(gmax(med), 3), "min_us": round(gmin(min(samples)), 3),
                                "max_us": round(gmax(max(samples)), 3), "valid": bool(gmin(float(ok_eager)) > 0.5)})
                # --- graph timing (+ replay validation) ---
                rec = {"method": "nccl", "mode": "graph", "rows": r, "width": w, "dtype": dname, "valid": False}
                try:
                    xg = pattern(torch, n, rank, dtype, dev).view(r, w)
                    g1 = torch.cuda.CUDAGraph()
                    for _ in range(3):
                        dist.all_reduce(z)
                    torch.cuda.synchronize()
                    with torch.cuda.graph(g1):
                        dist.all_reduce(xg)
                    xg.copy_(pattern(torch, n, rank, dtype, dev).view(r, w))
                    torch.cuda.synchronize()
                    dist.barrier()
                    g1.replay()
                    torch.cuda.synchronize()
                    ok_graph = bool(torch.equal(xg, exp))
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for _ in range(a.graph_iters):
                            dist.all_reduce(z)
                    torch.cuda.synchronize()
                    dist.barrier()
                    g.replay()
                    torch.cuda.synchronize()
                    gs = []
                    for _ in range(a.repeats):
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        dist.barrier()
                        torch.cuda.synchronize()
                        s.record()
                        g.replay()
                        e.record()
                        torch.cuda.synchronize()
                        gs.append(s.elapsed_time(e) * 1000.0 / a.graph_iters)
                    gm = statistics.median(gs)
                    rec.update({"median_us": round(gmax(gm), 3), "min_us": round(gmin(min(gs)), 3),
                                "max_us": round(gmax(max(gs)), 3), "valid": bool(gmin(float(ok_graph)) > 0.5)})
                    del g, g1
                except Exception as ex:  # noqa: BLE001
                    rec["error"] = repr(ex)
                results.append(rec)
                if rank == 0:
                    print(f"nccl {dname} {r}x{w}: eager {results[-2]['median_us']} us valid {results[-2]['valid']}; "
                          f"graph {rec.get('median_us')} us valid {rec['valid']}", flush=True)
    dist.barrier()
    ok = True
    out = {"results": results, "world": world, "elapsed_s": round(time.time() - t0, 1),
           "torch": torch.__version__, "nccl": _nccl_version(torch),
           "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
    if rank == 0:
        pat = a.nccl_log_glob
        if pat is None and os.environ.get("NCCL_DEBUG_FILE"):
            pat = re.sub(r"%[hp]", "*", os.environ["NCCL_DEBUG_FILE"])
        paths = sorted(glob.glob(pat)) if pat else []
        lines = transport_lines(paths)
        t_ok, kinds = judge_transport(lines)
        out["transport"] = {"ok": t_ok, "kinds": kinds, "files": paths, "sample": lines[:6]}
        bad = [x for x in results if not x["valid"] and x["mode"] == "eager"]
        out["invalid_eager_cells"] = len(bad)
        ok = t_ok and not bad
        out["ok"] = ok
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"transport {kinds} ok={t_ok}; invalid eager cells {len(bad)}; wrote {a.out}", flush=True)
        if not t_ok:
            print("FATAL: NCCL did not connect the two cards via P2P (or no NCCL_DEBUG channel lines were found) -- "
                  "the TP budget assumes P2P", file=sys.stderr)
    okt = torch.tensor([1.0 if ok else 0.0], device=dev)
    dist.broadcast(okt, 0)
    dist.destroy_process_group()
    if okt.item() < 0.5:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
