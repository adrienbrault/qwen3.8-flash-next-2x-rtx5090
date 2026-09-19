#!/usr/bin/env python3
"""Shared-expert join stall from a Kineto / Chrome trace of the served decode (CPU-only).

With EXL3_SHARED_EXPERT_OVERLAP=1 every MoE layer runs (blocksparse_mlp.cpp:102-127, exl3_moe_coop.cu:143-177):

    main stream : [rot] -> stage A (exl3_moe_coop_*_a_kernel) -> wait(shared_done) -> stage B (..._b_kernel)
    side stream : shared expert graph (exl3_mgemm_kernel gate+up, silu, exl3_gemm_kernel / int8 GEMV down)

The only way the shared expert still costs wall time is if its side-stream chain ends after stage A, so stage B
waits on the event (the "join stall"), or if it slows stage A by sharing SMs/DRAM (not visible in one trace: the
GPU bound probe measures it). This tool pairs every A with the next B on the same device+stream, finds the kernels
that ran on other streams of that device between the previous B's end and this B's start (the shared chain), and
reports per pair:

    exposed = max(0, min(shared_end, B_start) - A_end)    time B could not start because the shared chain was late

Summed over the trace and divided by --steps it is the per-step ceiling of any shared-expert speedup that leaves the
overlap in place (a fused or faster shared kernel only shortens the side stream, so it can recover at most this).

Usage:  python3 join_stall.py trace.json [--steps 64] [--json out.json]
Accepts torch.profiler chrome exports (traceEvents with ph "X", cat "kernel", args.device / args.stream).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict

A_RE = re.compile(r"exl3_moe_coop\w*_a_kernel")
B_RE = re.compile(r"exl3_moe_coop\w*_b_kernel")
# Kernels of the shared-expert graph (mlp.cpp:14-91): gate+up exl3_mgemm_kernel, act_mul_kernel_* (silu),
# down exl3_gemm_kernel / exl3_gemv_kernel / exl3_gemv_int8_{sq,coop}_kernel. Other side-stream work is ignored.
SHARED_RE = re.compile(r"exl3_mgemm_kernel|exl3_gemm_kernel|exl3_gemv\w*_kernel|act_mul_kernel")


def load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    events = doc["traceEvents"] if isinstance(doc, dict) else doc
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if str(e.get("cat", "")).lower() != "kernel":
            continue
        args = e.get("args") or {}
        dev = args.get("device", e.get("pid"))
        stream = args.get("stream", e.get("tid"))
        out.append({
            "name": e.get("name", ""),
            "ts": float(e["ts"]),
            "end": float(e["ts"]) + float(e.get("dur", 0.0)),
            "dev": dev,
            "stream": stream,
        })
    out.sort(key=lambda k: k["ts"])
    return out


def analyze(kernels: list[dict], shared_re: re.Pattern = SHARED_RE) -> dict:
    by_dev: dict = defaultdict(list)
    for k in kernels:
        by_dev[k["dev"]].append(k)
    pairs = []
    for dev, ks in by_dev.items():
        # pair each stage A with the next stage B on the same stream
        open_a: dict = {}
        prev_b_end: dict = {}
        for k in ks:
            if A_RE.search(k["name"]):
                open_a[k["stream"]] = k
            elif B_RE.search(k["name"]) and k["stream"] in open_a:
                a = open_a.pop(k["stream"])
                lo = prev_b_end.get(k["stream"], a["ts"] - 1e9)
                side = [s for s in ks
                        if s["stream"] != k["stream"] and s["ts"] >= lo and s["ts"] < k["ts"]
                        and shared_re.search(s["name"])]
                sh_end = max((s["end"] for s in side), default=None)
                sh_busy = sum(s["end"] - s["ts"] for s in side)
                exposed = 0.0
                if sh_end is not None:
                    exposed = max(0.0, min(sh_end, k["ts"]) - a["end"])
                pairs.append({
                    "dev": dev,
                    "stream": k["stream"],
                    "a_us": a["end"] - a["ts"],
                    "b_us": k["end"] - k["ts"],
                    "gap_ab_us": k["ts"] - a["end"],
                    "side_kernels": len(side),
                    "side_busy_us": sh_busy,
                    "exposed_us": exposed,
                })
                prev_b_end[k["stream"]] = k["end"]
    return {"pairs": pairs}


def summarize(res: dict, steps: int | None) -> dict:
    pairs = res["pairs"]
    n = len(pairs)
    tot_exp = sum(p["exposed_us"] for p in pairs)
    tot_side = sum(p["side_busy_us"] for p in pairs)
    with_side = sum(1 for p in pairs if p["side_kernels"])
    stalled = sum(1 for p in pairs if p["exposed_us"] > 0.5)
    s = {
        "moe_calls": n,
        "calls_with_side_stream_work": with_side,
        "calls_stalled_gt_0p5us": stalled,
        "exposed_total_us": round(tot_exp, 3),
        "side_busy_total_us": round(tot_side, 3),
        "exposed_per_call_us": round(tot_exp / n, 3) if n else None,
    }
    if steps:
        s["exposed_ms_per_step"] = round(tot_exp / steps / 1000.0, 4)
        s["side_busy_ms_per_step"] = round(tot_side / steps / 1000.0, 4)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--steps", type=int, default=None, help="decode steps captured (to normalize per step)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = analyze(load_events(a.trace))
    summ = summarize(res, a.steps)
    print(json.dumps(summ, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summ, **res}, f, indent=1)
    if summ["moe_calls"] == 0:
        print("no exl3_moe_coop stage A/B pairs found: was the trace taken on the served (coop) decode path?",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
