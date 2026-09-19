#!/usr/bin/env python3
"""Bitwise round-6 kernel checks for one GPU; run once per card.

Every candidate launch is compared with torch.equal against the served kernel:

- GDN B/A (EXL3_GDN_BA_WARP1): the served eight-warp reference is forced by padding the
  independent rows until the served grid reaches the SM count, so the selector's
  under-filled predicate falls back for the reference call and fires for the candidate.
- hc_apply (EXL3_HC_APPLY_WARP1): same padding trick; half/float y, comb/no-comb.
- V2 state (EXL3_GR_STATE_REGRID): served gr_mix_v2 vs gr_mix_v2_regrid (fp16 weights) and
  served gr_mix_v2_int8 vs gr_mix_v2_int8_regrid (int8 weights quantized exactly as
  GatedResidual._quantize_v2_int8 does), site form (post) and final form (no post),
  half and float `mixed`. All four outputs (dots, state, post, mixed) must be equal.

With --launch-check (default on) a torch.profiler pass records the kernel names of one
candidate call per family and fails if the round-6 kernel did not actually launch.

--dry-run prints the planned cases without importing torch (CPU tests use it).
Exit status is nonzero on any mismatch, non-finite output, or missing launch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys


H = 4
D = 2560
LR = 320
N_BA = 96
BA_WARPS = 8
HC_THREADS = 256
SELECTORS = ("EXL3_GDN_BA_WARP1", "EXL3_HC_APPLY_WARP1")


# ---- host-side geometry, identical to the served launch arithmetic ---------------------

def served_ba_blocks(rows: int) -> int:
    return ((N_BA + BA_WARPS - 1) // BA_WARPS) * rows


def served_hc_blocks(rows: int, threads: int = HC_THREADS) -> int:
    chunks = min(32, max(1, 256 // rows))
    chunk_cols = ((D // chunks + 4 * threads - 1) // (4 * threads)) * (4 * threads)
    return ((D + chunk_cols - 1) // chunk_cols) * rows


def first_rows_at_or_above(sm_count: int, blocks_fn) -> int:
    rows = 1
    while blocks_fn(rows) < sm_count:
        rows += 1
    return rows


def regrid_blocks(rows: int, with_post: bool) -> int:
    m = LR + (H if with_post else 0)
    return ((m + 31) // 32) * rows


def plan(rows: list[int], sm_count: int = 170) -> dict:
    return {
        "rows": rows,
        "assumed_sm_count": sm_count,
        "gdn_ba": [
            {"rows": r, "bias": b, "served_blocks": served_ba_blocks(r),
             "warp1_selected": served_ba_blocks(r) < sm_count}
            for r in rows for b in (False, True)
        ],
        "hc_apply": [
            {"rows": r, "y_dtype": y, "comb": c, "served_blocks": served_hc_blocks(r),
             "warp1_selected": served_hc_blocks(r) < sm_count,
             "warp1_blocks": served_hc_blocks(r, 32)}
            for r in rows for y in ("half", "float") for c in (False, True)
        ],
        "gr_state": [
            {"weights": w, "rows": r, "post": p, "mixed": m, "regrid_blocks": regrid_blocks(r, p)}
            for w in ("fp16", "int8") for r in rows for p in (False, True) for m in ("half", "float")
        ],
    }


# ---- GPU checks ------------------------------------------------------------------------

class Checker:
    def __init__(self):
        self.mismatches: list[dict] = []

    def equal(self, torch, left, right, case: dict) -> bool:
        ok = True
        if not (torch.isfinite(left).all().item() and torch.isfinite(right).all().item()):
            self.mismatches.append({**case, "reason": "non-finite"})
            ok = False
        elif not torch.equal(left, right):
            diff = (left.float() - right.float()).abs()
            self.mismatches.append({
                **case, "reason": "not equal",
                "n_diff": int((diff != 0).sum().item()), "max_abs": float(diff.max().item()),
            })
            ok = False
        return ok


def quantize_v2_int8(torch, fn_h, upx_h):
    """Copy of GatedResidual._quantize_v2_int8 (served hyperconnections.py:325-337)."""
    qmax = 127.0
    fn = fn_h.float()
    fn_s = (fn.abs().amax(dim=1).clamp_min(1e-8) / qmax).contiguous()
    fn_q = torch.round(fn / fn_s[:, None]).clamp_(-128, 127).to(torch.int8).contiguous()
    upx = upx_h.float()
    up_scale = (upx.abs().amax(dim=2).clamp_min(1e-8) / qmax).contiguous()
    upx_q = torch.round(upx / up_scale[:, :, None, :]).clamp_(-128, 127).to(torch.int8).contiguous()
    upx_s = up_scale.reshape(H, D).contiguous()
    return fn_q, fn_s, upx_q, upx_s


def alloc_gr(torch, rows, with_post, mixed_dtype, device):
    m = LR + (H if with_post else 0)
    # Poison the workspaces so an unwritten element cannot compare equal by accident.
    return (
        torch.full((rows, m + 1, H), float("nan"), dtype=torch.float, device=device),
        torch.full((rows, LR + H), float("nan"), dtype=torch.float, device=device),
        torch.full((rows, H), float("nan"), dtype=torch.float, device=device) if with_post else None,
        torch.full((rows, D), float("nan"), dtype=mixed_dtype, device=device),
    )


def run_gr(torch, fn, streams, weights, with_post, mixed_dtype):
    dots, state, post, mixed = alloc_gr(torch, streams.shape[0], with_post, mixed_dtype, streams.device)
    fn(streams, *weights, 1e-6, dots, state, post, mixed)
    return dots, state, post, mixed


def kernel_names(torch, call) -> list[str]:
    from torch.profiler import profile, ProfilerActivity
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    return sorted({evt.key for evt in prof.key_averages()})


def time_call(torch, call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(values)


def run_gpu(args) -> int:
    # The C++ selectors are process-lifetime statics read at the first launch of their
    # kernel, so setting them here (before any extension call) is sufficient.
    for key in SELECTORS:
        os.environ[key] = "1"
    import torch
    from exllamav3.ext import exllamav3_ext as ext

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (12, 0):
        raise SystemExit(f"round-6 re-grids require SM120, got {props.major}.{props.minor}")
    for name in ("gr_mix_v2_regrid", "gr_mix_v2_int8_regrid", "gr_mix_v2_int8", "exl3_moe_prefill_e3_det"):
        if not hasattr(ext, name):
            raise SystemExit(f"extension lacks {name}")

    sm = props.multi_processor_count
    rows_tested = list(range(1, args.max_rows + 1))
    result = {
        "device": str(device), "gpu": props.name, "capability": [props.major, props.minor],
        "sm_count": sm, "plan": plan(rows_tested, sm), "gdn_ba": [], "hc_apply": [], "gr_state": [],
        "launch_check": {}, "timing_us": {},
    }
    check = Checker()
    gen = torch.Generator(device=device).manual_seed(6006)

    with torch.inference_mode():
        # GDN B/A, native split-GDN shape (N, K) = (96, 2560).
        w_ba = (torch.randn((N_BA, D), generator=gen, device=device, dtype=torch.half) * 0.05).contiguous()
        b_ba = (torch.randn((N_BA,), generator=gen, device=device, dtype=torch.half) * 0.05).contiguous()
        ba_ref_rows = first_rows_at_or_above(sm, served_ba_blocks)
        for rows in rows_tested:
            for use_bias in (False, True):
                padded = max(rows, ba_ref_rows)
                x_pad = (torch.randn((padded, D), generator=gen, device=device, dtype=torch.half) * 0.05).contiguous()
                candidate = torch.full((rows, N_BA), float("nan"), dtype=torch.float, device=device)
                served = torch.full((padded, N_BA), float("nan"), dtype=torch.float, device=device)
                bias = b_ba if use_bias else None
                ext.gdn_ba_gemv(x_pad[:rows].contiguous(), w_ba, bias, candidate)
                ext.gdn_ba_gemv(x_pad, w_ba, bias, served)
                torch.cuda.synchronize(device)
                case = {"family": "gdn_ba", "rows": rows, "bias": use_bias,
                        "warp1_selected": served_ba_blocks(rows) < sm}
                result["gdn_ba"].append({**case, "equal": check.equal(torch, candidate, served[:rows], case)})

        # hc_apply, row-local and in place.
        hc_ref_rows = first_rows_at_or_above(sm, served_hc_blocks)
        for rows in rows_tested:
            for y_dtype in (torch.half, torch.float):
                for with_comb in (False, True):
                    padded = max(rows, hc_ref_rows)
                    x_pad = torch.randn((padded, H, D), generator=gen, device=device, dtype=torch.float)
                    y_pad = torch.randn((padded, D), generator=gen, device=device, dtype=y_dtype)
                    post_pad = torch.randn((padded, H), generator=gen, device=device, dtype=torch.float)
                    comb_pad = torch.randn((padded, H, H), generator=gen, device=device, dtype=torch.float) \
                        if with_comb else None
                    candidate = x_pad[:rows].clone()
                    served = x_pad.clone()
                    ext.hc_apply(candidate, y_pad[:rows].contiguous(), post_pad[:rows].contiguous(),
                                 comb_pad[:rows].contiguous() if with_comb else None)
                    ext.hc_apply(served, y_pad, post_pad, comb_pad)
                    torch.cuda.synchronize(device)
                    case = {"family": "hc_apply", "rows": rows, "y_dtype": str(y_dtype), "comb": with_comb,
                            "warp1_selected": served_hc_blocks(rows) < sm}
                    result["hc_apply"].append({**case, "equal": check.equal(torch, candidate, served[:rows], case)})

        # V2 state: fp16 and int8 weight variants, served entry point vs re-gridded entry point.
        streams_all = (torch.randn((args.max_rows, H, D), generator=gen, device=device) * 0.05).contiguous()
        norm_w = (1.0 + torch.randn((H * D,), generator=gen, device=device, dtype=torch.half) * 0.05).contiguous()
        up = (torch.randn((H * D, LR), generator=gen, device=device, dtype=torch.half) * 0.05).contiguous()
        upx = up.view(H, D // 4, 4, LR).permute(0, 1, 3, 2).contiguous()
        variants = {}
        for with_post in (False, True):
            m = LR + (H if with_post else 0)
            fn_h = (torch.randn((m, H * D), generator=gen, device=device, dtype=torch.half) * 0.05).contiguous()
            fn_q, fn_s, upx_q, upx_s = quantize_v2_int8(torch, fn_h, upx)
            variants[("fp16", with_post)] = (ext.gr_mix_v2, ext.gr_mix_v2_regrid, (fn_h, upx, norm_w))
            variants[("int8", with_post)] = (ext.gr_mix_v2_int8, ext.gr_mix_v2_int8_regrid,
                                             (fn_q, fn_s, upx_q, upx_s, norm_w))
        for (weights_kind, with_post), (served_fn, regrid_fn, weights) in variants.items():
            for rows in rows_tested:
                streams = streams_all[:rows].contiguous()
                for mixed_dtype in (torch.half, torch.float):
                    served = run_gr(torch, served_fn, streams, weights, with_post, mixed_dtype)
                    candidate = run_gr(torch, regrid_fn, streams, weights, with_post, mixed_dtype)
                    torch.cuda.synchronize(device)
                    ok = True
                    for label, left, right in zip(("dots", "state", "post", "mixed"), served, candidate):
                        if left is None:
                            continue
                        case = {"family": "gr_state", "weights": weights_kind, "rows": rows, "post": with_post,
                                "mixed": str(mixed_dtype), "tensor": label}
                        ok = check.equal(torch, left, right, case) and ok
                    result["gr_state"].append({"weights": weights_kind, "rows": rows, "post": with_post,
                                               "mixed": str(mixed_dtype), "equal": ok})

        # Launch check: the equality above is vacuous if the candidate fell back to the served kernel.
        if args.launch_check:
            x1 = (torch.randn((4, D), generator=gen, device=device, dtype=torch.half) * 0.05).contiguous()
            y1 = torch.empty((4, N_BA), dtype=torch.float, device=device)
            xs = torch.randn((4, H, D), generator=gen, device=device)
            ys = torch.randn((4, D), generator=gen, device=device, dtype=torch.half)
            ps = torch.randn((4, H), generator=gen, device=device)
            s4 = streams_all[:4].contiguous()
            probes = {
                "gdn_ba_rows4": (lambda: ext.gdn_ba_gemv(x1, w_ba, None, y1), r"gdn_ba_gemv_kernel<1>"),
                "hc_apply_rows4": (lambda: ext.hc_apply(xs, ys, ps, None), r"hc_apply_kernel<.*\b32>"),
                "gr_state_fp16_rows4": (lambda: run_gr(torch, variants[("fp16", True)][1], s4,
                                                       variants[("fp16", True)][2], True, torch.half),
                                        r"gr_v2_state_regrid_kernel"),
                "gr_state_int8_rows4": (lambda: run_gr(torch, variants[("int8", True)][1], s4,
                                                       variants[("int8", True)][2], True, torch.half),
                                        r"gr_v2_state_regrid_kernel"),
                "gr_state_int8_served_rows4": (lambda: run_gr(torch, variants[("int8", True)][0], s4,
                                                              variants[("int8", True)][2], True, torch.half),
                                               r"gr_v2_state_kernel"),
            }
            for name, (call, pattern) in probes.items():
                names = kernel_names(torch, call)
                found = any(re.search(pattern, n) for n in names)
                result["launch_check"][name] = {"expected": pattern, "found": found, "kernels": names}
                if not found:
                    check.mismatches.append({"family": "launch_check", "probe": name, "reason": "kernel not launched"})

        if args.iterations > 0:
            for (weights_kind, with_post), (served_fn, regrid_fn, weights) in variants.items():
                for rows in (1, 4, 16):
                    if rows > args.max_rows:
                        continue
                    streams = streams_all[:rows].contiguous()
                    ws = alloc_gr(torch, rows, with_post, torch.half, device)
                    key = f"{weights_kind}_{'site' if with_post else 'final'}_r{rows}"
                    result["timing_us"][key] = {
                        "served": time_call(torch, lambda: served_fn(streams, *weights, 1e-6, *ws),
                                            args.warmup, args.iterations),
                        "regrid": time_call(torch, lambda: regrid_fn(streams, *weights, 1e-6, *ws),
                                            args.warmup, args.iterations),
                    }

    result["mismatches"] = check.mismatches
    result["pass"] = not check.mismatches
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "device": result["device"], "gpu": result["gpu"], "rows": [1, args.max_rows],
        "cases": len(result["gdn_ba"]) + len(result["hc_apply"]) + len(result["gr_state"]),
        "mismatches": len(check.mismatches), "pass": result["pass"],
    }))
    return 0 if result["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100,
                        help="timing iterations for the V2 mixer entry points; 0 skips timing")
    parser.add_argument("--no-launch-check", dest="launch_check", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without importing torch")
    args = parser.parse_args(argv)
    if not 1 <= args.max_rows <= 32:
        parser.error("--max-rows must be in 1..32 (gr_mix_v2 domain)")
    if args.dry_run:
        payload = {"dry_run": True, "selectors_set": list(SELECTORS) + ["(GR_STATE_REGRID via entry points)"],
                   "plan": plan(list(range(1, args.max_rows + 1)))}
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({"dry_run": True, "cases": sum(len(v) for k, v in payload["plan"].items()
                                                        if isinstance(v, list) and k != "rows")}))
        return 0
    return run_gpu(args)


if __name__ == "__main__":
    sys.exit(main())
