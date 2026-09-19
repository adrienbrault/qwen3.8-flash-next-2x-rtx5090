#!/usr/bin/env python3
"""TP=2 bounding probe, part 2b: P2P gate + a single-process peer-copy + add all-reduce (one process, both cards).

1. Gate (exit 3 on failure, before anything else runs): two CUDA devices, cudaDeviceCanAccessPeer both ways, and
   EXACT content of peer copies 4 KB .. 32 MB in both directions (R129 found a driver state in which the API reported
   peer access while copies returned garbage). Peer-copy bandwidth at 32 MB is reported (R129: 28.5 GB/s).
2. Peer-copy + add all-reduce ("one-shot push"): each card pulls the other card's partial (rows x width) over P2P into a
   local buffer and adds it to its own; the two cards synchronize on each other's events every iteration (the exchange
   cannot run ahead of the peer's partial), so the per-call time is the latency a TP step would see. Eager only (the
   cross-device event wait cannot live inside one CUDA graph). Validated exactly on small-integer patterns.
   Same payload grid as tp_bound_ar.py. This is what a hand-written P2P all-reduce in exllamav3 would do with IPC
   handles in two processes; exllamav3 itself has no P2P collective (its native TP backend reduces through host shared
   memory, model_tp_backend.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="1,4,8,16")
    ap.add_argument("--widths", default="2560,10240")
    ap.add_argument("--dtypes", default="fp32,fp16")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", default="tp_bound_p2p.json")
    return ap.parse_args(argv)


def check_copies(torch, d0, d1, sizes):
    res = []
    for nbytes in sizes:
        n = nbytes // 4
        for src_d, dst_d in ((d0, d1), (d1, d0)):
            src = (torch.arange(n, dtype=torch.int32, device=src_d) * 2654435761 % 2147483647).to(torch.int32)
            dst = torch.full((n,), -1, dtype=torch.int32, device=dst_d)
            dst.copy_(src)
            torch.cuda.synchronize(src_d)
            torch.cuda.synchronize(dst_d)
            ok = bool(torch.equal(dst.cpu(), src.cpu()))
            res.append({"bytes": nbytes, "src": str(src_d), "dst": str(dst_d), "exact": ok})
    return res


def copy_bandwidth(torch, d0, d1, nbytes=32 << 20, reps=20):
    out = {}
    for src_d, dst_d in ((d0, d1), (d1, d0)):
        src = torch.empty(nbytes, dtype=torch.uint8, device=src_d)
        dst = torch.empty(nbytes, dtype=torch.uint8, device=dst_d)
        dst.copy_(src)
        torch.cuda.synchronize(src_d)
        torch.cuda.synchronize(dst_d)
        with torch.cuda.device(dst_d):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(reps):
                dst.copy_(src, non_blocking=True)
            e.record()
            e.synchronize()
            ms = s.elapsed_time(e) / reps
        out[f"{src_d.index}->{dst_d.index}"] = {"GBps": round(nbytes / ms / 1e6, 2), "ms": round(ms, 4)}
    return out


class PeerAR:
    """One-shot push all-reduce of x0 (cuda:0) and x1 (cuda:1), in place, synchronized per call."""

    def __init__(self, torch, x0, x1):
        self.t = torch
        self.x = [x0, x1]
        self.buf = [torch.empty_like(x0), torch.empty_like(x1)]
        self.dev = [x0.device, x1.device]
        self.st = [torch.cuda.Stream(device=self.dev[0]), torch.cuda.Stream(device=self.dev[1])]
        E = torch.cuda.Event
        self.ready = [E(), E()]       # partial ready (before the exchange)
        self.done = [E(), E()]        # this side finished reading the peer (peer may overwrite)

    def __call__(self):
        t = self.t
        # both sides: mark partial ready
        for i in (0, 1):
            with t.cuda.device(self.dev[i]), t.cuda.stream(self.st[i]):
                self.ready[i].record(self.st[i])
        for i in (0, 1):
            j = 1 - i
            with t.cuda.device(self.dev[i]), t.cuda.stream(self.st[i]):
                self.st[i].wait_event(self.ready[j])
                self.buf[i].copy_(self.x[j], non_blocking=True)      # P2P read of the peer's partial
                self.done[i].record(self.st[i])
        for i in (0, 1):
            j = 1 - i
            with t.cuda.device(self.dev[i]), t.cuda.stream(self.st[i]):
                self.st[i].wait_event(self.done[j])                  # peer has read my partial: safe to overwrite
                self.x[i].add_(self.buf[i])


def time_peer_ar(torch, ar, iters, repeats):
    samples = []
    d0 = ar.dev[0]
    for _ in range(repeats):
        torch.cuda.synchronize(ar.dev[0])
        torch.cuda.synchronize(ar.dev[1])
        with torch.cuda.device(d0):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(ar.st[0])
        for _ in range(iters):
            ar()
        with torch.cuda.device(d0):
            e.record(ar.st[0])
        torch.cuda.synchronize(ar.dev[1])
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / iters)
    return samples


def main(argv=None) -> int:
    a = parse_args(argv)
    import torch
    out = {"ok": False, "results": []}
    n = torch.cuda.device_count()
    out["device_count"] = n
    if n < 2:
        print("FATAL: two CUDA devices required", file=sys.stderr)
        _write(a.out, out)
        return 3
    d0, d1 = torch.device("cuda", 0), torch.device("cuda", 1)
    out["devices"] = [torch.cuda.get_device_name(i) for i in (0, 1)]
    acc = {"0->1": bool(torch.cuda.can_device_access_peer(0, 1)), "1->0": bool(torch.cuda.can_device_access_peer(1, 0))}
    out["can_access_peer"] = acc
    if not all(acc.values()):
        print(f"FATAL: P2P not available: {acc} (TP budget premise; is the P2P driver loaded?)", file=sys.stderr)
        _write(a.out, out)
        return 3
    copies = check_copies(torch, d0, d1, [4 << 10, 64 << 10, 1 << 20, 32 << 20])
    out["copies"] = copies
    if not all(c["exact"] for c in copies):
        print(f"FATAL: peer copies are not exact: {[c for c in copies if not c['exact']]}", file=sys.stderr)
        _write(a.out, out)
        return 3
    out["bandwidth"] = copy_bandwidth(torch, d0, d1)
    print(f"P2P ok: {acc}; copies exact; bandwidth {out['bandwidth']}", flush=True)
    if min(v["GBps"] for v in out["bandwidth"].values()) < 15.0:
        print("WARN: peer-copy bandwidth below 15 GB/s (R129 measured 28.5): the copies may be host-staged",
              file=sys.stderr)
        out["bandwidth_warning"] = True

    t0 = time.time()
    for dname in a.dtypes.split(","):
        dtype = getattr(torch, DTYPES[dname])
        for w in [int(x) for x in a.widths.split(",")]:
            for r in [int(x) for x in a.rows.split(",")]:
                nel = r * w
                base = torch.arange(nel) % 17
                x0 = base.to(dtype).to(d0).view(r, w)
                x1 = (base + 3).to(dtype).to(d1).view(r, w)
                ar = PeerAR(torch, x0, x1)
                ar()
                torch.cuda.synchronize(d0)
                torch.cuda.synchronize(d1)
                exp = (2 * base + 3).to(dtype).view(r, w)
                valid = bool(torch.equal(x0.cpu(), exp) and torch.equal(x1.cpu(), exp))
                x0.zero_()
                x1.zero_()
                for _ in range(10):
                    ar()
                samples = time_peer_ar(torch, ar, a.iters, a.repeats)
                rec = {"method": "peer_copy_add", "mode": "eager", "rows": r, "width": w, "dtype": dname,
                       "median_us": round(statistics.median(samples), 3), "min_us": round(min(samples), 3),
                       "max_us": round(max(samples), 3), "valid": valid}
                out["results"].append(rec)
                print(f"peer_copy_add {dname} {r}x{w}: {rec['median_us']} us valid {valid}", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["ok"] = all(x["valid"] for x in out["results"])
    _write(a.out, out)
    if not out["ok"]:
        print("FATAL: peer-copy all-reduce produced a wrong sum", file=sys.stderr)
        return 3
    return 0


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
