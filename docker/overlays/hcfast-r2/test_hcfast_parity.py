#!/usr/bin/env python3
"""hcfast r2 GPU parity test: the served int8 state-in-up mixer vs gr_mix_v2_int8_v3, bitwise.

Reference (the served launches, EXL3_HC_MIX_V2=1 / MIN_R=1 / INT8=1 / GR_STATE_IN_UP=1):
    ext.gr_mix_v2_int8_statein(streams, fn_q, fn_s, up_q, up_s, w, eps, dots, state, post, mixed)
Candidate:
    ext.gr_mix_v2_int8_v3(..., mode, dots_bmax, up_bmax, dots_j, dots_pf, up_q)
      r1 (EXL3_HC_MIX_V3=1): mode 3 = both launches V3, 1 = V3 dots only, 2 = V3 up only, 0 = the copied
        served kernels; caps 8/4/2/1 retile rows per CTA, 0 = one-wave auto tiling.
      r2 (EXL3_HC_MIX_V3=2): mode 15 = both launches r2 (hcv4), 31 = the same with programmatic dependent
        launch; dots_j 4/8 fn rows per CTA, dots_pf 0/1/2 stream iterations prefetched, up_q 2/4 column
        quads per CTA; every (B, J, PF) dots instantiation and every (B, Q) up instantiation is covered.

Every output (dots, state, post, mixed) must be storage-bit equal. Outputs start from different
sentinel bit patterns in the two arms, so an element one side never writes fails too.

Sections (all must pass; exit 0 only if they do):
  1. real HC weights (attention + MLP sites of layers 12 and 40, the final mixer), rows 1..32,
     mixed half and fp32, all r1 and r2 configurations; the returned mask must be the expected one
  2. Python integration: GatedResidual._mix at level 1, level 2 (defaults) and level 2 with per-R tables
     and PDL, vs off (post and mixed), rows 1..32
  3. fuzz: 200 random seeds (rows 1..32, site, magnitude 1e-2..3e2, configuration, mixed dtype)
  4. CUDA graph capture of the candidate (r1, r2, r2 + PDL configurations) at R = 1, 4, 8, 16 + 20 replays on
     fresh inputs vs eager reference. Each captured graph is: a kernel that WRITES the streams (so the PDL
     dots launch has a real producer to wait for), mixer A, mixer B sharing A's dots workspace (B's dots
     writes it while A's up is the kernel it follows: the write-after-read case), with separate outputs;
     4b. in a fresh process, the auto tile (caps 0/0) captured with its occupancy query never run before
     (the first auto call happens inside the capture, as it would in a served graph), 5 replays each at R = 4, 16
  5. fallback: a 16-byte-misaligned dots workspace must decline V3 / r2 up (mask bits 1 and 3 clear) and stay
     exact
  6. PDL (mode 31 and the L2 table + PDL integration variant) through sections 1-5, in a fresh process. A launch
     or capture error on the platform is reported as "pdl: unsupported" in the JSON (the gate then runs P0 with
     --no-pdl) and does not fail the non-PDL verdict; a PDL output mismatch fails the test.

Run (one GPU, never the daily's card while it serves):
  docker run --rm --gpus '"device=0"' -v /srv/qwen5090/models:/models:ro --entrypoint python3 \\
    tabbyapi:hcfast-r2 /opt/hcfast-r2/test_hcfast_parity.py \\
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
for k in ("EXL3_HC_MIX_V3", "EXL3_HC_MIX_V3_DOTS_B", "EXL3_HC_MIX_V3_UP_B", "EXL3_HC_MIX_V3_DOTS_J",
          "EXL3_HC_MIX_V3_DOTS_PF", "EXL3_HC_MIX_V3_UP_Q", "EXL3_HC_MIX_V3_PDL"):
    os.environ.pop(k, None)

ROWS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 24, 32)
LAYERS = (12, 40)
# (mode, dots_bmax, up_bmax, dots_j, dots_pf, up_q)
R1_CONFIGS = tuple(c + (4, 0, 2) for c in (
    (3, 8, 8), (1, 8, 8), (2, 8, 8), (0, 8, 8), (3, 4, 8), (3, 2, 8), (3, 1, 8),
    (3, 8, 4), (3, 8, 2), (3, 8, 1), (0, 1, 1), (3, 0, 0), (0, 0, 0)))
R2_CONFIGS = (
    # every dots instantiation (B = 8 clamps to J = 4, PF = 0; B = 4 to PF <= 1, PF = 0 at J = 8)
    tuple((15, b, 8, j, pf, 2) for b in (1, 2, 4, 8) for j in (4, 8) for pf in (0, 1, 2))
    # every up instantiation (B = 8, Q = 2 is r1's kernel)
    + tuple((15, 2, b, 4, 0, q) for b in (1, 2, 4, 8) for q in (2, 4))
    # PDL, and the mixed selections (one launch r2, the other V3 / served copy)
    + ((31, 2, 8, 4, 0, 2), (31, 2, 8, 8, 1, 4), (31, 1, 4, 4, 2, 4), (31, 8, 8, 4, 0, 2), (31, 4, 2, 8, 0, 2),
       (31, 2, 8, 4, 2, 2), (7, 2, 8, 8, 1, 2), (11, 2, 8, 4, 0, 4), (12, 2, 4, 8, 2, 4), (28, 1, 8, 8, 0, 4),
       (15, 0, 0, 4, 0, 2)))
CONFIGS = R1_CONFIGS + R2_CONFIGS
GRAPH_CONFIGS = ((3, 8, 8, 4, 0, 2), (3, 2, 8, 4, 0, 2), (15, 2, 8, 4, 0, 2), (15, 2, 8, 8, 1, 4),
                 (31, 2, 8, 4, 0, 2), (31, 2, 8, 8, 1, 4), (31, 1, 4, 4, 2, 4), (31, 8, 8, 4, 0, 2),
                 (31, 4, 2, 8, 1, 4))
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
    ap.add_argument("--pdl-child", action = "store_true", help = argparse.SUPPRESS)
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
    assert hcm._HC_MIX_V3_BUILD == "r2", hcm._HC_MIX_V3_BUILD
    assert getattr(ext, "hc_mix_v3_revision", None) == 2, "extension without the r2 gr_mix_v2_int8_v3"
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
        dots, state, post, mixed = buffers(m, s3.shape[0], mixed_dtype, 0x7F800003)
        if dots_ws is not None:
            dots = dots_ws
        ran = ext.gr_mix_v2_int8_v3(s3, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                    dots, state, post, mixed, *cfgt)
        return (dots, state, post, mixed), ran

    def tile_rows(R, cap):
        served = 1 if R == 1 else (2 if R == 2 else (4 if R <= 4 else 8))
        cap = 8 if cap >= 8 else (4 if cap >= 4 else (2 if cap >= 2 else 1))
        return min(served, cap)

    def expected_mask(cfgt, mixed_dtype, R, aligned = True):
        mode, _, bu, _, _, uq = cfgt
        if not v3_shape:
            return 0
        if not mode & 12:
            return (mode & 1) | ((mode & 2) if aligned else 0)
        dots_v4 = bool(mode & 4)
        up_v4 = bool(mode & 8) and aligned and mixed_dtype == torch.half
        up_real = up_v4 and not (tile_rows(R, bu if bu else 8) == 8 and uq < 4)   # B = 8, Q = 2 runs r1's up
        up_bits = (8 if up_real else 2) if up_v4 else ((mode & 2) if aligned else 0)
        return ((4 if dots_v4 else mode & 1) | up_bits
                | (16 if (mode & 16) and (dots_v4 or up_real) else 0))

    def compare(label, m, s3, mixed_dtype, cfgt):
        ref = run_ref(m, s3, mixed_dtype)
        got, ran = run_v3(m, s3, mixed_dtype, cfgt)
        torch.cuda.synchronize(dev)
        ok = check(label, zip(("dots", "state", "post", "mixed"), ref, got))
        expect = expected_mask(cfgt, mixed_dtype, s3.shape[0])
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
            run_v3(m, s_w, torch.half, (3, bb, bb, 4, 0, 2))
        torch.cuda.synchronize(dev)
        n = n_ok = 0
        for R in (4, 16):
            s_static = streams_for(R, 7000 + R)
            dots, state, post, mixed = buffers(m, R, torch.half, 0x7F800003)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ext.gr_mix_v2_int8_v3(s_static, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                      dots, state, post, mixed, 3, 0, 0, 4, 0, 2)
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

    PDL = lambda cs: tuple(c for c in cs if c[0] & 16)
    NOPDL = lambda cs: tuple(c for c in cs if not c[0] & 16)

    def sec1(configs, tag):
        print(f"\n[1{tag}] kernel parity: rows x sites x mixed dtype x {len(configs)} configurations")
        n = n_ok = 0
        for key, m in sites.items():
            for R in ROWS:
                s3 = streams_for(R, 1000 + R)
                for mixed_dtype in (torch.half, torch.float):
                    for cfgt in configs:
                        n += 1
                        n_ok += compare(f"[1{tag}] {key} R={R} {str(mixed_dtype)[6:]} cfg={cfgt}", m, s3, mixed_dtype, cfgt)
        print(f"  {n_ok}/{n} identical")
        return [n_ok, n]

    G = GatedResidual
    saved = (G.MIX_V3_DOTS_B, G.MIX_V3_UP_B, G.MIX_V3_DOTS_J, G.MIX_V3_DOTS_PF, G.MIX_V3_UP_Q, G.MIX_V3_PDL)
    def set_level(level, tables):
        G.MIX_V3_LEVEL, G.MIX_V3 = level, level > 0
        (G.MIX_V3_DOTS_B, G.MIX_V3_UP_B, G.MIX_V3_DOTS_J, G.MIX_V3_DOTS_PF, G.MIX_V3_UP_Q,
         G.MIX_V3_PDL) = tables
    t33 = lambda f: tuple(f(r) for r in range(33))
    VARIANTS = (
        ("L1", 1, saved),
        ("L2-default", 2, (t33(lambda r: 2), saved[1], saved[2], t33(lambda r: 1), t33(lambda r: 4), saved[5])),
        ("L2-table", 2, (t33(lambda r: 4 if r <= 4 else 2), t33(lambda r: 2 if r <= 4 else 8),
                         t33(lambda r: 4 if r <= 4 else 8), t33(lambda r: 2 if r <= 4 else 1),
                         t33(lambda r: 2 if r <= 4 else 4), False)))
    PDL_VARIANTS = (("L2-table-pdl", 2, VARIANTS[2][2][:5] + (True,)),)

    def sec2(variants, tag):
        print(f"\n[2{tag}] GatedResidual._mix, " + " / ".join(v[0] for v in variants) + " vs off")
        n = n_ok = 0
        for key, m in sites.items():
            for R in ROWS:
                s = streams_for(R, 2000 + R).view(1, R, H, D)
                set_level(0, saved)
                p0, x0 = m._mix(s, cached = False)
                for vtag, level, tables in variants:
                    set_level(level, tables)
                    p1, x1 = m._mix(s, cached = False)
                    set_level(0, saved)
                    torch.cuda.synchronize(dev)
                    n += 1
                    n_ok += check(f"[2{tag}] {vtag} {key} R={R}", (("post", p0, p1), ("mixed", x0, x1)))
        print(f"  {n_ok}/{n} identical")
        return [n_ok, n]

    def sec3(configs, seeds, tag):
        print(f"\n[3{tag}] fuzz, {seeds} seeds")
        rng = random.Random(20260924 + len(tag))
        keys = list(sites)
        n = n_ok = 0
        for seed in range(seeds):
            R = rng.randint(1, 32)
            key = rng.choice(keys)
            scale = 10 ** rng.uniform(-2, 2.5)
            cfgt = rng.choice(configs)
            mixed_dtype = rng.choice((torch.half, torch.float))
            s3 = streams_for(R, 30000 + seed, scale)
            n += 1
            n_ok += compare(f"[3{tag}] seed={seed} {key} R={R} scale={scale:.3g} cfg={cfgt}", sites[key], s3, mixed_dtype, cfgt)
        print(f"  {n_ok}/{n} identical")
        return [n_ok, n]

    def sec4(configs, tag):
        print(f"\n[4{tag}] graph capture + {args.graph_replays} replays: producer kernel, mixer A, mixer B (shared dots)")
        n = n_ok = 0
        keys = list(sites)
        m = sites[keys[0]]
        mb = sites[keys[1]]
        for cfgt in configs:
            for R in (1, 4, 8, 16):
                src_a = streams_for(R, 4000 + R)
                src_b = streams_for(R, 4500 + R)
                s_a = torch.empty_like(src_a)
                s_b = torch.empty_like(src_b)
                dots, state_a, post_a, mixed_a = buffers(m, R, torch.half, 0x7F800003)
                _, state_b, post_b, mixed_b = buffers(mb, R, torch.half, 0x7F800003)
                assert mb.fn_q.shape == m.fn_q.shape
                def body():
                    torch.mul(src_a, 1.0, out = s_a)            # the producer a PDL dots A must wait for
                    torch.mul(src_b, 1.0, out = s_b)
                    ext.gr_mix_v2_int8_v3(s_a, m.fn_q, m.fn_s, m.upx_q, m.upx_s, m.w_h, m.rms_eps,
                                          dots, state_a, post_a, mixed_a, *cfgt)
                    ext.gr_mix_v2_int8_v3(s_b, mb.fn_q, mb.fn_s, mb.upx_q, mb.upx_s, mb.w_h, mb.rms_eps,
                                          dots, state_b, post_b, mixed_b, *cfgt)
                side = torch.cuda.Stream(dev)
                side.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(side):     # warm the launch path outside capture
                    body()
                torch.cuda.current_stream(dev).wait_stream(side)
                torch.cuda.synchronize(dev)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    body()
                for rep in range(args.graph_replays):
                    sc = 10 ** random.Random(rep).uniform(-1, 2)
                    src_a.copy_(streams_for(R, 5000 + 100 * R + rep, sc))
                    src_b.copy_(streams_for(R, 5500 + 100 * R + rep, sc))
                    for t in (dots, state_a, mixed_a, state_b, mixed_b) + tuple(p for p in (post_a, post_b) if p is not None):
                        t.view(torch.uint8).fill_(0xA5)
                    graph.replay()
                    ref_a = run_ref(m, src_a, torch.half)
                    ref_b = run_ref(mb, src_b, torch.half)
                    torch.cuda.synchronize(dev)
                    n += 1
                    n_ok += check(f"[4{tag}] cfg={cfgt} R={R} replay {rep}",
                                  [("state_a", ref_a[1], state_a), ("post_a", ref_a[2], post_a), ("mixed_a", ref_a[3], mixed_a),
                                   ("dots_b", ref_b[0], dots), ("state_b", ref_b[1], state_b), ("post_b", ref_b[2], post_b),
                                   ("mixed_b", ref_b[3], mixed_b)])
                del graph
        print(f"  {n_ok}/{n} identical")
        return [n_ok, n]

    def sec5(configs, tag):
        print(f"\n[5{tag}] misaligned dots workspace -> V3 / r2 up declines, outputs exact")
        n = n_ok = 0
        keys = list(sites)
        for cfgt in configs:
            for R in (1, 4, 16):
                mm = sites[keys[0]]
                M = mm.fn_q.shape[0]
                flat = torch.empty((R * (M + 1) * H + 1,), dtype = torch.float, device = dev)
                dws = flat[1:].view(R, M + 1, H)
                assert dws.data_ptr() % 16 != 0
                s3 = streams_for(R, 6000 + R)
                ref = run_ref(mm, s3, torch.half)
                got, ran = run_v3(mm, s3, torch.half, cfgt, dots_ws = dws)
                torch.cuda.synchronize(dev)
                n += 1
                ok = check(f"[5{tag}] cfg={cfgt} R={R}", zip(("dots", "state", "post", "mixed"), ref, got))
                expect = expected_mask(cfgt, torch.half, R, aligned = False)
                if ran != expect:
                    FAILURES.append(f"[5{tag}] cfg={cfgt} R={R}: mask {ran}, expected {expect}")
                    ok = False
                n_ok += ok
        print(f"  {n_ok}/{n} identical")
        return [n_ok, n]

    if args.pdl_child:
        # Section 6 body, in a fresh process: every PDL configuration through sections 1-5. A launch or capture
        # error before any PDL work succeeded means the platform does not take the attribute (exit 3, reported
        # as unsupported by the parent, not as a parity failure); a mismatch is a failure (exit 1).
        first = PDL(CONFIGS)[0]
        try:
            m0 = sites[list(sites)[0]]
            run_v3(m0, streams_for(4, 1), torch.half, first)
            torch.cuda.synchronize(dev)
            g0 = torch.cuda.CUDAGraph()
            s0 = streams_for(4, 2)
            with torch.cuda.graph(g0):
                run_v3(m0, s0, torch.half, first)
            g0.replay()
            torch.cuda.synchronize(dev)
            del g0
        except RuntimeError as e:
            print(f"PDL-UNSUPPORTED: {str(e).splitlines()[0][:300]}")
            return 3
        res = [sec1(PDL(CONFIGS), "p"), sec2(PDL_VARIANTS, "p"), sec3(PDL(CONFIGS), 100, "p"),
               sec4(PDL(GRAPH_CONFIGS), "p"), sec5(((31, 2, 8, 8, 1, 4), (31, 2, 8, 4, 1, 2)), "p")]
        ok = sum(r[0] for r in res); tot = sum(r[1] for r in res)
        print(f"PDL {ok}/{tot}")
        return 0 if ok == tot and not FAILURES else 1

    summary["kernel"] = sec1(NOPDL(CONFIGS), "")
    summary["integration"] = sec2(VARIANTS, "")
    summary["fuzz"] = sec3(NOPDL(CONFIGS), args.fuzz_seeds, "")
    summary["graph"] = sec4(NOPDL(GRAPH_CONFIGS), "")

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

    summary["fallback"] = sec5(((3, 8, 8, 4, 0, 2), (15, 2, 8, 4, 0, 2), (15, 2, 8, 8, 1, 4)), "")

    print("\n[6] PDL configurations (mode bit 16), fresh process: sections 1-5 on every PDL configuration")
    cmd = [sys.executable, os.path.abspath(__file__), "--pdl-child", "--model", args.model,
           "--device", args.device, "--graph-replays", str(args.graph_replays)] + (["--synthetic"] if args.synthetic else [])
    child = subprocess.run(cmd, capture_output = True, text = True)
    lines = [l for l in child.stdout.splitlines() if l.startswith(("PDL", "  FAIL")) or "identical" in l]
    for l in lines[-12:]:
        print("  " + l.strip())
    if child.returncode == 0:
        summary["pdl"] = "ok"
    elif child.returncode == 3:
        summary["pdl"] = "unsupported: " + next((l for l in lines if l.startswith("PDL-UNSUPPORTED")), "?")
        print("  PDL not supported on this platform (not a parity failure; the gate runs P0 with --no-pdl)")
    else:
        summary["pdl"] = f"FAIL rc={child.returncode}"
        FAILURES.append(f"[6] PDL child rc={child.returncode}: " + (lines[-1] if lines else child.stderr[-400:]))
        print(f"  FAIL [6] rc={child.returncode}; stderr tail: {child.stderr[-400:]}")

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
