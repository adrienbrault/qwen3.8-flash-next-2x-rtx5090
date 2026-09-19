#!/usr/bin/env python3
"""CPU/offline fp16-vs-int8 mixer error audit at the served Qwen dimensions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F


HIDDEN = 2560
HC_COUNT = 4
RANK = 320
TRUNK_SITES = 96


def quantize(fn_h: torch.Tensor, up_h: torch.Tensor):
    fn_f = fn_h.float()
    fn_s = fn_f.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    fn_q = torch.round(fn_f / fn_s[:, None]).clamp_(-128, 127).to(torch.int8)

    upx = up_h.view(HC_COUNT, HIDDEN // 4, 4, RANK).permute(0, 1, 3, 2).contiguous()
    upx_f = upx.float()
    up_s = upx_f.abs().amax(dim=2).clamp_min(1e-8) / 127.0
    up_q = torch.round(upx_f / up_s[:, :, None, :]).clamp_(-128, 127).to(torch.int8)
    return fn_q, fn_s, up_q, up_s


def dequantize(fn_q, fn_s, up_q, up_s):
    fn = fn_q.float() * fn_s[:, None]
    upx = up_q.float() * up_s[:, :, None, :]
    up = upx.permute(0, 1, 3, 2).contiguous().view(HC_COUNT * HIDDEN, RANK)
    return fn, up


def mix(streams, fn, up, w_h, use_combine=True):
    rows = streams.shape[0]
    m = RANK + (HC_COUNT if use_combine else 0)
    fn = fn[:m].view(m, HC_COUNT, HIDDEN)
    dots = torch.einsum("rhd,mhd->rmh", streams, fn.float())
    rmr = torch.rsqrt(streams.square().mean(dim=-1) + 1e-6)
    state = (dots * rmr[:, None, :]).sum(dim=-1) / HC_COUNT
    rank_state = F.silu(state[:, :RANK])
    gates = rank_state @ up.float().T
    mixed = (
        torch.sigmoid(gates.view(rows, HC_COUNT, HIDDEN))
        * streams
        * rmr[:, :, None]
        * w_h.float().view(1, HC_COUNT, HIDDEN)
    ).mean(dim=1).half()
    post = 2.0 * torch.sigmoid(state[:, RANK:]) if use_combine else None
    return post, mixed


def update_error(acc, name, ref, got):
    err = (got.float() - ref.float()).abs()
    rel = err / ref.float().abs().clamp_min(1e-6)
    acc[name]["max_abs"] = max(acc[name]["max_abs"], float(err.max()))
    acc[name]["max_rel"] = max(acc[name]["max_rel"], float(rel.max()))
    acc[name]["mean_abs_sum"] += float(err.mean())


def storage_bytes(site_count: int, final_count: int):
    def one(use_combine):
        m = RANK + (HC_COUNT if use_combine else 0)
        fp16 = 2 * (m * HC_COUNT * HIDDEN + HC_COUNT * HIDDEN * RANK)
        int8 = (
            m * HC_COUNT * HIDDEN + 4 * m
            + HC_COUNT * HIDDEN * RANK + 4 * HC_COUNT * HIDDEN
        )
        return fp16, int8

    sf, si = one(True)
    ff, fi = one(False)
    return {
        "fp16_bytes": site_count * sf + final_count * ff,
        "int8_bytes": site_count * si + final_count * fi,
        "freed_bytes": site_count * (sf - si) + final_count * (ff - fi),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=TRUNK_SITES)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    errors = {
        "mixed": {"max_abs": 0.0, "max_rel": 0.0, "mean_abs_sum": 0.0},
        "post": {"max_abs": 0.0, "max_rel": 0.0, "mean_abs_sum": 0.0},
    }
    started = time.perf_counter()
    for _ in range(args.sites):
        streams = torch.randn(args.rows, HC_COUNT, HIDDEN, generator=gen, dtype=torch.float)
        fn_h = (torch.randn(RANK + HC_COUNT, HC_COUNT * HIDDEN, generator=gen) * 0.02).half()
        up_h = (torch.randn(HC_COUNT * HIDDEN, RANK, generator=gen) * 0.02).half()
        w_h = (1.0 + torch.randn(HC_COUNT * HIDDEN, generator=gen) * 0.02).half()
        fn_q, fn_s, up_q, up_s = quantize(fn_h, up_h)
        fn_dq, up_dq = dequantize(fn_q, fn_s, up_q, up_s)
        post_ref, mixed_ref = mix(streams, fn_h, up_h, w_h)
        post_i8, mixed_i8 = mix(streams, fn_dq, up_dq, w_h)
        update_error(errors, "mixed", mixed_ref, mixed_i8)
        update_error(errors, "post", post_ref, post_i8)

    for values in errors.values():
        values["mean_abs"] = values.pop("mean_abs_sum") / args.sites
    result = {
        "synthetic": {
            "hidden": HIDDEN,
            "hc_count": HC_COUNT,
            "rank": RANK,
            "sites": args.sites,
            "rows": args.rows,
            "seed": args.seed,
        },
        "errors": errors,
        "storage_96_site_mixers": storage_bytes(96, 0),
        "storage_target_96_plus_final": storage_bytes(96, 1),
        "storage_target_plus_mtp_100_mixers": storage_bytes(98, 2),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_tested": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
