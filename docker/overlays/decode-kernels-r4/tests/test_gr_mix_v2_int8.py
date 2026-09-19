#!/usr/bin/env python3
"""Box-only CUDA parity/error/timing harness for gr_mix_v2_int8."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from exllamav3.ext import exllamav3_ext as ext

    torch.manual_seed(1234)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    H, D, LR = 4, 2560, 320

    def quantize(fn_h, up_h):
        fn_f = fn_h.float()
        fn_s = fn_f.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        fn_q = torch.round(fn_f / fn_s[:, None]).clamp_(-128, 127).to(torch.int8).contiguous()
        upx = up_h.view(H, D // 4, 4, LR).permute(0, 1, 3, 2).contiguous()
        up_f = upx.float()
        up_s4 = up_f.abs().amax(dim=2).clamp_min(1e-8) / 127.0
        up_q = torch.round(up_f / up_s4[:, :, None, :]).clamp_(-128, 127).to(torch.int8).contiguous()
        return fn_q, fn_s.contiguous(), up_q, up_s4.reshape(H, D).contiguous()

    def torch_ref(streams, fn_q, fn_s, up_q, up_s, w_h, combine):
        m = LR + (H if combine else 0)
        fn = (fn_q.float() * fn_s[:, None]).view(m, H, D)
        dots = torch.einsum("rhd,mhd->rmh", streams, fn)
        rmr = torch.rsqrt(streams.square().mean(-1) + 1e-6)
        state = (dots * rmr[:, None, :]).sum(-1) / H
        t = F.silu(state[:, :LR])
        upx = up_q.float() * up_s.view(H, D // 4, 1, 4)
        up = upx.permute(0, 1, 3, 2).contiguous().view(H * D, LR)
        gates = t @ up.T
        mixed = (
            torch.sigmoid(gates.view(-1, H, D)) * streams * rmr[:, :, None]
            * w_h.float().view(1, H, D)
        ).mean(1).half()
        post = 2 * torch.sigmoid(state[:, LR:]) if combine else None
        return post, mixed

    def alloc(rows, combine):
        m = LR + (H if combine else 0)
        streams = torch.randn(rows, H, D, device=device, dtype=torch.float)
        fn_h = (torch.randn(m, H * D, device=device) * 0.02).half()
        up_h = (torch.randn(H * D, LR, device=device) * 0.02).half()
        upx_h = up_h.view(H, D // 4, 4, LR).permute(0, 1, 3, 2).contiguous()
        w_h = (1 + torch.randn(H * D, device=device) * 0.02).half()
        fn_q, fn_s, up_q, up_s = quantize(fn_h, up_h)
        dots = torch.empty(rows, m + 1, H, device=device, dtype=torch.float)
        state = torch.empty(rows, LR + H, device=device, dtype=torch.float)
        post = torch.empty(rows, H, device=device, dtype=torch.float) if combine else None
        mixed = torch.empty(rows, D, device=device, dtype=torch.half)
        return streams, fn_h, upx_h, fn_q, fn_s, up_q, up_s, w_h, dots, state, post, mixed

    def timed(call):
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(args.iterations):
            a = torch.cuda.Event(enable_timing = True)
            b = torch.cuda.Event(enable_timing = True)
            a.record(); call(); b.record(); b.synchronize()
            samples.append(a.elapsed_time(b) * 1000.0)
        return statistics.median(samples)

    results = []
    for combine in (True, False):
        for rows in args.rows:
            values = alloc(rows, combine)
            streams, fn_h, upx_h, fn_q, fn_s, up_q, up_s, w_h, dots, state, post, mixed = values
            m = fn_h.shape[0]
            dots_fp = torch.empty_like(dots)
            state_fp = torch.empty_like(state)
            post_fp = torch.empty_like(post) if post is not None else None
            mixed_fp = torch.empty_like(mixed)

            def fp16_call():
                ext.gr_mix_v2(streams, fn_h, upx_h, w_h, 1e-6,
                              dots_fp, state_fp, post_fp, mixed_fp)

            def int8_call():
                ext.gr_mix_v2_int8(streams, fn_q, fn_s, up_q, up_s, w_h, 1e-6,
                                   dots, state, post, mixed)

            fp16_call(); int8_call(); torch.cuda.synchronize(device)
            post_ref, mixed_ref = torch_ref(streams, fn_q, fn_s, up_q, up_s, w_h, combine)
            torch.testing.assert_close(mixed, mixed_ref, rtol=5e-3, atol=5e-3)
            if combine:
                torch.testing.assert_close(post, post_ref, rtol=5e-3, atol=5e-3)
            delta = (mixed.float() - mixed_fp.float()).abs()
            results.append({
                "rows": rows,
                "combine": combine,
                "mixed_fp16_int8_max_abs": float(delta.max()),
                "mixed_fp16_int8_max_rel": float((delta / mixed_fp.float().abs().clamp_min(1e-6)).max()),
                "fp16_median_us": timed(fp16_call),
                "int8_median_us": timed(int8_call),
            })

    output = {"device": str(device), "results": results}
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
