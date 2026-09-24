#!/usr/bin/env python3
"""hcfast r1 GPU parity test: the served int8 state-in-up mixer vs gr_mix_v2_int8_v3, bitwise.

Reference (the served launches, EXL3_HC_MIX_V2=1 / MIN_R=1 / INT8=1 / GR_STATE_IN_UP=1):
    ext.gr_mix_v2_int8_statein(streams, fn_q, fn_s, up_q, up_s, w, eps, dots, state, post, mixed)
Candidate:
    ext.gr_mix_v2_int8_v3(..., mode, dots_bmax, up_bmax)
      mode 3 = both launches V3 (what EXL3_HC_MIX_V3=1 serves), 1 = V3 dots only, 2 = V3 up only,
      0 = the copied served kernels; caps 8/4/2/1 retile rows per CTA, 0 = one-wave auto tiling.

Every output (dots, state, post, mixed) must be storage-bit equal. Outputs start from different
sentinel bit patterns in the two arms, so an element one side never writes fails too.

Sections (all must pass; exit 0 only if they do):
  1. real HC weights (attention + MLP sites of layers 12 and 40, the final mixer), rows 1..32,
     mixed half and fp32, 13 mode/cap configurations; the returned V3 mask must be the requested one
  2. Python integration: GatedResidual._mix with MIX_V3 on vs off (post and mixed), rows 1..32
  3. fuzz: 200 random seeds (rows 1..32, site, magnitude 1e-2..3e2, configuration, mixed dtype)
  4. CUDA graph capture of the candidate at R = 1, 4, 8, 16 + 20 replays on fresh inputs vs eager reference;
     4b. in a fresh process, the auto tile (caps 0/0) captured with its occupancy query never run before
     (the first auto call happens inside the capture, as it would in a served graph), 5 replays each at R = 4, 16
  5. fallback: a 16-byte-misaligned dots workspace must decline V3 up (mask bit 1 clear) and stay exact

Run (one GPU, never the daily's card while it serves):
  docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro --entrypoint python3 \\
    tabbyapi:hcfast-r1 /opt/hcfast-r1/test_hcfast_parity.py \\
    --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab
--synthetic uses random weights of the checkpoint's shapes instead (explicit, never a fallback).
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

SERVED_FLAGS = {
    "EXL3_HC_MIX_V2": "1",
    "EXL3_HC_MIX_V2_MIN_R": "1",
    "EXL3_HC_MIX_V2_INT8": "1",
    "EXL3_GR_STATE_REGRID": "1",
    "EXL3_GR_STATE_IN_UP": "1",
    "EXL3_HC_APPLY_WARP1": "1",
}
for k, v in SERVED_FLAGS.items():
    os.environ[k] = v
for k in ("EXL3_HC_MIX_V3", "EXL3_HC_MIX_V3_DOTS_B", "EXL3_HC_MIX_V3_UP_B"):
    os.environ.pop(k, None)

ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 24, 32)
LAYERS = (12, 40)
# (mode, dots_bmax, up_bmax)
CONFIGS = ((3, 8, 8), (1, 8, 8), (2, 8, 8), (0, 8, 8), (3, 4, 8), (3, 2, 8), (3, 1, 8),
           (3, 8, 4), (3, 8, 2), (3, 8, 1), (0, 1, 1), (3, 0, 0), (0, 0, 0))
FAILURES = []


def arguments():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default = "/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab")
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--synthetic", action = "store_true")
    ap.add_argument("--fuzz-seeds", type = int, default = 200)
    ap.add_argument("--graph-replays", type = int, default = 20)
    ap.add_argument("--json", type = Path, help = "optional summary output")
    ap.add_argument("--auto-capture-child", action = "store_true", help = argparse.SUPPRESS)
    a = ap.parse_args()
    assert a.fuzz_seeds >= 200, "at least 200 fuzz seeds"
    return a


def main():
    args = arguments()
    if not __debug__:
        raise RuntimeError("do not run with python -O")
    import torch
    import exllamav3
    import exllamav3.modules.hyperconnections as hcm
    from exllamav3.modules.hyperconnections import GatedResidual
    from exllamav3.ext import exllamav3_ext as ext

    assert torch.cuda.is_available(), "CUDA required"
    assert hcm._HC_MIX_V3_BUILD == "r1", hcm._HC_MIX_V3_BUILD
    assert getattr(ext, "hc_mix_v3_revision", None) == 1, "extension without gr_mix_v2_int8_v3"
    assert GatedResidual.MIX_V2 and GatedResidual.MIX_V2_MIN_R == 1 and GatedResidual.MIX_V2_INT8
    assert GatedResidual.STATE_IN_UP and GatedResidual.MIX_V3 is False
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    print(f"exllamav3 {exllamav3.__file__}")
    print(f"torch {torch.__version__} CUDA {torch.version.cuda} {torch.cuda.get_device_name(dev)}")

    cfg_json = json.loads((Path(args.model) / "config.json").read_text())
    tc = cfg_json.get("text_config", cfg_json)
    H, D, LR, eps = tc["hc_count"], tc["hidden_size"], tc["hc_lowrank"], tc["rms_norm_eps"]
    assert H == 4
    v3_shape = D == 2560 and LR == 320
    print(f"H={H} D={D} LR={LR}; V3 kernels apply to this shape: {v3_shape}")

    # ---- sites ---------------------------------------------------------------------------------
    sites = {}
    if args.synthetic:
        gen = torch.Generator(device = dev).manual_seed(614)
        for name, combine in (("syn.attn", True), ("syn.mlp", True), ("syn.final", False)):
            m = GatedResidual(None, name, H, D, eps, use_combine = combine)
            m.norm_w_raw = torch.randn(H * D, device = dev, generator = gen) * 0.05
            down = (torch.randn(LR, H * D, device = dev, generator = gen) / (H * D) ** 0.5).half()
            up = (torch.randn(H * D, LR, device = dev, generator = gen) / LR ** 0.5).half()
            inject = (torch.randn(H, H * D, device = dev, generator = gen) / (H * D) ** 0.5).half() if combine else None
            m._prepare(down, up, inject)
            sites[name] = m
        print("weights: SYNTHETIC (explicit --synthetic)")
    else:
        from exllamav3.model.config import Config
        cfg = Config.from_directory(args.model)
        keys = [(f"model.language_model.layers.{l}.{s}_hyper_connection", True)
                for l in LAYERS for s in ("attn", "mlp")]
        keys.append(("model.language_model.hyper_connection_mixer", False))
        for key, combine in keys:
            m = GatedResidual(cfg, key, H, D, eps, use_combine = combine)
            m.load(dev)
            sites[key] = m
        print(f"weights: REAL, {len(sites)} sites from {args.model}")
    for m in sites.values():
        assert m.fn_q is not None and m.upx_q is not None and m.rank == LR

    def bits_equal(a, b):
        return a.shape == b.shape and a.dtype == b.dtype and \
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))

    def check(label, pairs):
        bad = []
        for name, a, b in pairs:
            if a is None or b is None:
                if not (a is None and b is None):
                    bad.append(f"{name}: one side None")
                continue
            if not bits_equal(a, b):
                af, bf = a.float(), b.float()
                n = int((af.view(-1).view(torch.int32) != bf.view(-1).view(torch.int32)).sum()) \
                    if af.shape == bf.shape else -1
                d = (af - bf).abs().nan_to_num(float("inf")).max().item() if af.shape == bf.shape else float("nan")
                bad.append(f"{name}: {n} elements differ, max |diff| {d:.3e}")
        if bad:
            FAILURES.append(f"{label}: " + "; ".join(bad))
            print(f"  FAIL {label}: " + "; ".join(bad))
        return not bad

    def buffers(m, R, mixed_dtype, sentinel_bits):
        M = m.fn_q.shape[0]
        def fill(shape, dtype):
            t = torch.empty(shape, dtype = dtype, device = dev)
            if dtype == torch.half:
                t.view(torch.int16).fill_(sentinel_bits & 0x7FFF)
            else:
                t.view(torch.int32).fill_(sentinel_bits)
            return t
        dots = fill((R, M + 1, H), torch.float)
        state = fill((R, LR + H), torch.float)
        post = fill((R, H), torch.float) if m.use_combine else None
        mixed = fill((R, D), mixed_dtype)
        return dots, state, post, mixed

    def run_ref(m, s3, mixed_dtype):
        dots, state, post, mixed = buffers(m, s3.shape[0], mixed_dtype, 0x7FC00001)
        ext.gr_mix_v2_int8_statein(s3, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                   dots, state, post, mixed)
        return dots, state, post, mixed

    def run_v3(m, s3, mixed_dtype, cfgt, dots_ws = None):
        mode, bd, bu = cfgt
        dots, state, post, mixed = buffers(m, s3.shape[0], mixed_dtype, 0x7F800003)
        if dots_ws is not None:
            dots = dots_ws
        ran = ext.gr_mix_v2_int8_v3(s3, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                    dots, state, post, mixed, mode, bd, bu)
        return (dots, state, post, mixed), ran

    def compare(label, m, s3, mixed_dtype, cfgt):
        ref = run_ref(m, s3, mixed_dtype)
        got, ran = run_v3(m, s3, mixed_dtype, cfgt)
        torch.cuda.synchronize(dev)
        ok = check(label, zip(("dots", "state", "post", "mixed"), ref, got))
        expect = (cfgt[0] & 3) if v3_shape else 0
        if ran != expect:
            FAILURES.append(f"{label}: V3 mask {ran}, expected {expect}")
            print(f"  FAIL {label}: V3 mask {ran}, expected {expect}")
            ok = False
        return ok

    def streams_for(R, seed, scale = 4.0):
        g = torch.Generator(device = dev).manual_seed(seed)
        return (torch.randn((R, H, D), device = dev, generator = g) * scale).contiguous()

    summary = {}

    if args.auto_capture_child:
        # Section 4b body, run in a fresh process by the parent. Kernel modules are loaded eagerly at fixed
        # caps first (lazy loading is not what this checks); the auto path's occupancy query and cache are
        # first touched inside the capture.
        m = sites[list(sites)[0]]
        for bb in (1, 2, 4, 8):
            s_w = streams_for(8, 77)
            run_v3(m, s_w, torch.half, (3, bb, bb))
        torch.cuda.synchronize(dev)
        n = n_ok = 0
        for R in (4, 16):
            s_static = streams_for(R, 7000 + R)
            dots, state, post, mixed = buffers(m, R, torch.half, 0x7F800003)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ext.gr_mix_v2_int8_v3(s_static, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                      dots, state, post, mixed, 3, 0, 0)
            for rep in range(5):
                s_static.copy_(streams_for(R, 7100 + 10 * R + rep))
                for t in (dots, state, mixed) + ((post,) if post is not None else ()):
                    t.view(torch.uint8).fill_(0x5A)
                graph.replay()
                ref = run_ref(m, s_static, torch.half)
                torch.cuda.synchronize(dev)
                n += 1
                n_ok += check(f"[4b] auto R={R} replay {rep}", zip(("dots", "state", "post", "mixed"), ref, (dots, state, post, mixed)))
            del graph
        print(f"AUTO-CAPTURE {n_ok}/{n}")
        return 0 if n_ok == n else 1

    # ---- 1. real weights, all configurations ----------------------------------------------------
    print("\n[1] kernel parity: rows x sites x mixed dtype x configurations")
    n = n_ok = 0
    for key, m in sites.items():
        for R in ROWS:
            s3 = streams_for(R, 1000 + R)
            for mixed_dtype in (torch.half, torch.float):
                for cfgt in CONFIGS:
                    n += 1
                    n_ok += compare(f"[1] {key} R={R} {str(mixed_dtype)[6:]} cfg={cfgt}", m, s3, mixed_dtype, cfgt)
    print(f"  {n_ok}/{n} identical")
    summary["kernel"] = [n_ok, n]

    # ---- 2. Python integration ----------------------------------------------------------------
    print("\n[2] GatedResidual._mix, MIX_V3 on vs off")
    n = n_ok = 0
    for key, m in sites.items():
        for R in ROWS:
            s = streams_for(R, 2000 + R).view(1, R, H, D)
            GatedResidual.MIX_V3 = False
            p0, x0 = m._mix(s, cached = False)
            GatedResidual.MIX_V3 = True
            p1, x1 = m._mix(s, cached = False)
            GatedResidual.MIX_V3 = False
            torch.cuda.synchronize(dev)
            n += 1
            n_ok += check(f"[2] {key} R={R}", (("post", p0, p1), ("mixed", x0, x1)))
    print(f"  {n_ok}/{n} identical")
    summary["integration"] = [n_ok, n]

    # ---- 3. fuzz --------------------------------------------------------------------------------
    print(f"\n[3] fuzz, {args.fuzz_seeds} seeds")
    rng = random.Random(20260924)
    keys = list(sites)
    n = n_ok = 0
    for seed in range(args.fuzz_seeds):
        R = rng.randint(1, 32)
        key = rng.choice(keys)
        scale = 10 ** rng.uniform(-2, 2.5)
        cfgt = rng.choice(CONFIGS)
        mixed_dtype = rng.choice((torch.half, torch.float))
        s3 = streams_for(R, 30000 + seed, scale)
        n += 1
        n_ok += compare(f"[3] seed={seed} {key} R={R} scale={scale:.3g} cfg={cfgt}", sites[key], s3, mixed_dtype, cfgt)
    print(f"  {n_ok}/{n} identical")
    summary["fuzz"] = [n_ok, n]

    # ---- 4. CUDA graph capture -----------------------------------------------------------------
    print(f"\n[4] graph capture + {args.graph_replays} replays")
    n = n_ok = 0
    m = sites[keys[0]]
    for R in (1, 4, 8, 16):
        s_static = streams_for(R, 4000 + R)
        dots, state, post, mixed = buffers(m, R, torch.half, 0x7F800003)
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):     # warm the launch path outside capture
            ext.gr_mix_v2_int8_v3(s_static, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                  dots, state, post, mixed, 3, 8, 8)
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ext.gr_mix_v2_int8_v3(s_static, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                  dots, state, post, mixed, 3, 8, 8)
        for rep in range(args.graph_replays):
            s_static.copy_(streams_for(R, 5000 + 100 * R + rep, 10 ** random.Random(rep).uniform(-1, 2)))
            for t in (dots, state, mixed) + ((post,) if post is not None else ()):
                t.view(torch.uint8).fill_(0xA5)
            graph.replay()
            ref = run_ref(m, s_static, torch.half)
            torch.cuda.synchronize(dev)
            n += 1
            n_ok += check(f"[4] R={R} replay {rep}", zip(("dots", "state", "post", "mixed"), ref, (dots, state, post, mixed)))
        del graph
    print(f"  {n_ok}/{n} identical")
    summary["graph"] = [n_ok, n]

    print("\n[4b] auto tile captured cold, fresh process")
    import subprocess
    cmd = [sys.executable, os.path.abspath(__file__), "--auto-capture-child", "--model", args.model,
           "--device", args.device] + (["--synthetic"] if args.synthetic else [])
    child = subprocess.run(cmd, capture_output = True, text = True)
    tail = [l for l in child.stdout.splitlines() if l.startswith(("AUTO-CAPTURE", "  FAIL"))]
    for l in tail:
        print("  " + l.strip())
    if child.returncode != 0 or not any(l.startswith("AUTO-CAPTURE") for l in tail):
        FAILURES.append(f"[4b] auto capture child rc={child.returncode}: " + (tail[-1] if tail else child.stderr[-400:]))
        print(f"  FAIL [4b] rc={child.returncode}; stderr tail: {child.stderr[-400:]}")
    summary["auto_capture"] = child.returncode == 0

    # ---- 5. fallback on a misaligned dots workspace ---------------------------------------------
    print("\n[5] misaligned dots workspace -> V3 up declines, outputs exact")
    n = n_ok = 0
    for R in (1, 4, 16):
        mm = sites[keys[0]]
        M = mm.fn_q.shape[0]
        flat = torch.empty((R * (M + 1) * H + 1,), dtype = torch.float, device = dev)
        dws = flat[1:].view(R, M + 1, H)
        assert dws.data_ptr() % 16 != 0
        s3 = streams_for(R, 6000 + R)
        ref = run_ref(mm, s3, torch.half)
        got, ran = run_v3(mm, s3, torch.half, (3, 8, 8), dots_ws = dws)
        torch.cuda.synchronize(dev)
        n += 1
        ok = check(f"[5] R={R}", zip(("dots", "state", "post", "mixed"), ref, got))
        expect = 1 if v3_shape else 0
        if ran != expect:
            FAILURES.append(f"[5] R={R}: mask {ran}, expected {expect}")
            ok = False
        n_ok += ok
    print(f"  {n_ok}/{n} identical")
    summary["fallback"] = [n_ok, n]

    print()
    if FAILURES:
        print(f"PARITY FAILED: {len(FAILURES)} failures; first: {FAILURES[0]}")
    else:
        print("PARITY PASS")
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "failures": FAILURES[:200],
                                         "synthetic": args.synthetic}, indent = 2) + "\n")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
