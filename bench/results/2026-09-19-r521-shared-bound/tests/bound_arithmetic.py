#!/usr/bin/env python3
"""Arithmetic behind impl-status.md section 2: how much of the shared expert is still on the critical path after
R490's side-stream overlap, from numbers already recorded in ref/FINDINGS.md and ref/decode-overhead-survey.md.
Every input is quoted with its source; test_cpu.py pins the outputs so the document cannot drift from them.
"""
from __future__ import annotations

LAYERS = 48

# survey §4.1 / §4.2 (R464 traces, 3.05 pack, post-R460 stack, overlap OFF): shared-expert kernel ms per step
SHARED_MS = {"c1d3": 1.138, "c4d3": 1.222}
# R465 un-profiled wall ms per step (FINDINGS R465 table, 1k ctx), same stack
WALL_MS = {"c1d3": 14.81, "c4d3": 23.49}
# survey: "~147 MB" K=5 shared payload per step (48 layers)
SHARED_MB_PER_STEP = 147.0
PEAK_TBPS = 1.79

# R490 one-layer bench (FINDINGS:5309), microseconds per MoE layer, OFF -> ON overlap
R490_LAYER_US = {1: [(71.3, 61.9)], 4: [(170.1, 142.8)], 16: [(310.4, 279.9), (263.6, 241.0)]}

# R490 fn_bench greedy (identical tokens, so tok/s ratio == step-time ratio), 4 runs per arm
FN = {
    "code c1": ([212.6, 217.0, 215.1, 216.7], [218.4, 220.8, 220.1, 218.6]),
    "prose c1": ([166.1, 172.4, 167.4, 171.9], [179.9, 180.1, 179.7, 178.9]),
    "code c4": ([428.4, 440.6, 432.2, 444.2], [448.2, 460.9, 459.8, 460.4]),
    "prose c4": ([420.2, 433.8, 420.6, 437.2], [446.7, 453.5, 450.3, 453.4]),
}
# R490 multiprompt paired-by-seed gains (sampled, n = 24 per kind)
MULTI = {"code c1": 6.3, "prose c1": 2.7, "code c4": 5.5, "prose c4": 6.9}


def per_layer_us(shape: str) -> float:
    return SHARED_MS[shape] * 1000.0 / LAYERS


def ceiling_gain_pct(shape: str) -> float:
    """Decode speedup if the whole shared bucket left the step (step time -> wall - shared)."""
    w = WALL_MS[shape]
    return (w / (w - SHARED_MS[shape]) - 1.0) * 100.0


def fn_gain_pct(kind: str) -> float:
    off, on = FN[kind]
    return (sum(on) / len(on)) / (sum(off) / len(off)) * 100.0 - 100.0


def realized_ms(shape: str, gain_pct: float) -> float:
    """ms/step removed by the overlap if the step sped up by gain_pct."""
    w = WALL_MS[shape]
    return w - w / (1.0 + gain_pct / 100.0)


def report() -> dict:
    r = {}
    for s in SHARED_MS:
        r[f"shared_us_per_layer_{s}"] = round(per_layer_us(s), 2)
        r[f"ceiling_gain_pct_{s}"] = round(ceiling_gain_pct(s), 2)
    r["payload_mb_per_layer"] = round(SHARED_MB_PER_STEP / LAYERS, 3)
    r["payload_floor_us_per_layer"] = round(SHARED_MB_PER_STEP / LAYERS * 1e-3 / PEAK_TBPS * 1e3, 2)
    r["payload_rate_gbps_c1d3"] = round(SHARED_MB_PER_STEP / SHARED_MS["c1d3"], 1)
    for rows, pairs in R490_LAYER_US.items():
        r[f"r490_saving_us_rows{rows}"] = [round(off - on, 1) for off, on in pairs]
    for k in FN:
        r[f"fn_gain_pct_{k.replace(' ', '_')}"] = round(fn_gain_pct(k), 2)
    c1 = [fn_gain_pct("code c1"), fn_gain_pct("prose c1"), MULTI["code c1"], MULTI["prose c1"]]
    c4 = [fn_gain_pct("code c4"), fn_gain_pct("prose c4"), MULTI["code c4"], MULTI["prose c4"]]
    g1, g4 = sum(c1) / 4, sum(c4) / 4
    r["mean_gain_pct_c1"] = round(g1, 2)
    r["mean_gain_pct_c4"] = round(g4, 2)
    r["realized_ms_c1d3"] = round(realized_ms("c1d3", g1), 3)
    r["realized_ms_c4d3"] = round(realized_ms("c4d3", g4), 3)
    r["residual_ms_c1d3_point"] = round(SHARED_MS["c1d3"] - realized_ms("c1d3", g1), 3)
    r["residual_ms_c4d3_point"] = round(SHARED_MS["c4d3"] - realized_ms("c4d3", g4), 3)
    # microbench bound: the smallest per-layer saving at 16 rows vs the standalone shared cost at c4 d3
    r["residual_ms_c4d3_microbench_bound"] = round(
        max(0.0, per_layer_us("c4d3") - min(off - on for off, on in R490_LAYER_US[16])) * LAYERS / 1000.0, 3)
    r["residual_ms_c1d3_microbench_bound"] = round(
        max(0.0, per_layer_us("c1d3") - min(off - on for off, on in R490_LAYER_US[4])) * LAYERS / 1000.0, 3)
    return r


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=1))
