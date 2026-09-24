#!/usr/bin/env python3
"""R682 stage D0: decode rate of the SERVED EXL3 kernels, in-process in the served image.

Why: the K<=4 EXL3 GEMV/MoE kernels reach 18-50 % of DRAM bandwidth in the R680 traces while the K=6 lm_head
reaches 86 % (out-roofline/BYTES-MODEL.md, last section). This driver separates the three candidate causes
(decode-instruction bound / single-wave-occupancy bound / fixed per-call latency) per bit width K and row count.

Entry points, exactly as the model calls them (src/exllamav3, the installed package of tabbyapi:slotfix-r1):
  gemm  : ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, mcg=False, mul1=True, 0)
          = BC_LinearEXL3::run (libtorch/linear.cpp:34-45), what LinearEXL3.forward -> bc.run_alloc does for
          rows <= 144. Dispatch inside (quant/exl3_gemm.cu:176-282): rows<=2 -> exl3_gemv_int8_sq (K<=6 on
          Blackwell), else the QTIP exl3_gemv kernel (K 2..4, rows<=8, shape heuristic), else the autotuned
          cooperative exl3_gemm kernel. Served users: GDN out_proj, attn o_proj, index_qk, lm_head (K=6).
  mgemm : ext.exl3_mgemm(...) in SLICED mode = GatedDeltaNet.project_qkvz_sliced (modules/gated_delta_net.py:
          940-970) / BC_GatedDeltaNetSplit (libtorch/gated_delta_net.cpp:236-243): in_proj_qkv + in_proj_z as
          8 slices of 2048 columns in one launch. Served user: GDN in_proj (the trace's mgemm<4> (20,1,8)).
  moe   : ext.exl3_moe_coop(...) = the same exl3_moe_coop_run -> exl3_moe_coop_launch that
          BC_BlockSparseMLP::run_bszN calls (libtorch/blocksparse_mlp.cpp:79-128, quant/exl3_moe_coop.cu),
          honouring EXL3_MOE_COOP_V2 per launch. Difference to the served call: no shared-expert output is
          merged (sh_out=None), so the side-stream shared expert is not in the timing.

Weights: REAL tensors from the served checkpoint where the served model has that K for that path (dense K=4:
GDN in_proj_qkv/z and out_proj of all 36 GDN layers; lm_head K=6; routed experts K=2 layer 20, K=3 layer 3,
K=4 the MTP layer's experts), and SYNTHETIC random trellis bits for the other K. Any 16-bit window of an EXL3
trellis is a valid code (mul1 codebook = hash multiply + byte sum, no data-dependent control flow), so decode
cost does not depend on the bit values; every real cell has a synthetic twin of identical shape and rotation
count, and HOW-TO-READ accepts synthetic numbers only if the twins agree within 5 %.

L2: 96 MiB. Every timed cell rotates over >= 4x L2 of distinct weights (copies for dense; a walk over a random
permutation of the 512 experts for MoE, so consecutive calls touch disjoint experts), so the timings are DRAM
timings, not L2 timings. ncu (--mode ncu) flushes caches itself (--cache-control all) and uses one copy.

Timing: CUDA events, GPU queue pre-filled behind torch.cuda._sleep so host launch latency is not measured.
  us_each  = median of per-call event pairs (200 reps after 20 warmup) = kernel time of the call
  us_batch = one event pair around 200 back-to-back calls / 200 = throughput incl. inter-kernel gaps
Bytes counted = exactly what the weights occupy: in*out*K/8 + suh (2*in) + svh (2*out) per matrix
(BYTES-MODEL.md "How EXL3 trellis bits become bytes"); activations are excluded and listed separately.

Modes: --mode time (events, writes cells.jsonl), --mode ncu (one call per cell inside cudaProfilerStart/Stop,
NVTX-labelled, run under ncu --profile-from-start off), --mode dryrun (CPU only: key/shape checks + plan).
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

DRAM_PEAK = 1792.128e9          # B/s, TARGET_INFO_GPU.memoryBandwidth in the R680 sqlite
TOPK = 10
N_EXPERTS = 512
HIDDEN = 2560
MOE_I = 640
MAX_BSZN = 16                   # modules/block_sparse_mlp.py:57, libtorch/mlp.h:11
GDN_LAYERS = [i for i in range(48) if i % 4 != 3]   # full_attention_interval 4 -> 3,7,..,47 are attention
EXPERT_LAYERS = {2: "model.language_model.layers.20.mlp", 3: "model.language_model.layers.3.mlp",
                 4: "mtp.layers.0.mlp"}
LM_HEAD = "lm_head"


def log(*a):
    print("[d0]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ tensors
class Checkpoint:
    """Real tensors by key via safetensors (model.safetensors.index.json), one handle per file."""

    def __init__(self, model_dir, device):
        self.dir = Path(model_dir)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.device = device
        self.handles = {}

    def get(self, key):
        from safetensors import safe_open
        f = self.map[key]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.dir / f), framework="pt", device=str(self.device))
        return self.handles[f].get_tensor(key)

    def linear(self, prefix):
        tr, suh, svh = (self.get(f"{prefix}.{s}") for s in ("trellis", "suh", "svh"))
        assert f"{prefix}.mul1" in self.map, f"{prefix}: no mul1 codebook tensor; the kernels here run mul1"
        assert tr.dtype.itemsize == 2 and tr.dim() == 3 and suh.shape[0] == tr.shape[0] * 16 \
            and svh.shape[0] == tr.shape[1] * 16, (prefix, tuple(tr.shape), tuple(suh.shape), tuple(svh.shape))
        return tr.contiguous(), suh.contiguous(), svh.contiguous()


def lin_bytes(k, n, K):
    return k * n * K // 8 + 2 * k + 2 * n


def synth_linear(torch, k, n, K, device, copies=None):
    """Random-bit trellis (k/16, n/16, 16K) int16, suh = +-1, svh = +-0.01 (keeps fp16 outputs finite)."""
    shape = (k // 16, n // 16, 16 * K) if copies is None else (copies, k // 16, n // 16, 16 * K)
    tr = torch.randint(-32768, 32767, shape, dtype=torch.int16, device=device)
    lead = () if copies is None else (copies,)
    suh = (torch.randint(0, 2, lead + (k,), device=device) * 2 - 1).half()
    svh = ((torch.randint(0, 2, lead + (n,), device=device) * 2 - 1) * 0.01).half()
    return tr, suh, svh


def rot_copies(bytes_one, l2, lo=2, hi=400):
    return max(lo, min(hi, math.ceil(4 * l2 / bytes_one)))


# ------------------------------------------------------------------------------------------------ timing
class Timer:
    def __init__(self, torch, reps, warm):
        self.torch, self.reps, self.warm = torch, reps, warm
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * reps + 2)]
        s = torch.cuda.current_stream()
        for e in self.ev:              # first record initialises the event outside every measurement
            e.record(s)
        torch.cuda.synchronize()
        # GPU sleep long enough that the host enqueues all reps before the GPU reaches them
        self.sleep_cycles = int(1.0e8)
        a, b = self.ev[0], self.ev[1]
        a.record(); torch.cuda._sleep(self.sleep_cycles); b.record(); torch.cuda.synchronize()
        self.sleep_ms = a.elapsed_time(b)

    def run(self, fns):
        t, n, R = self.torch, len(fns), self.reps
        for i in range(self.warm):
            fns[i % n]()
        t.cuda.synchronize()
        ev = self.ev
        t.cuda._sleep(self.sleep_cycles)
        h0 = time.perf_counter()
        for i in range(R):
            ev[2 * i].record(); fns[i % n](); ev[2 * i + 1].record()
        h1 = time.perf_counter()
        t.cuda.synchronize()
        each = sorted(ev[2 * i].elapsed_time(ev[2 * i + 1]) * 1e3 for i in range(R))
        b0, b1 = ev[2 * R], ev[2 * R + 1]
        t.cuda._sleep(self.sleep_cycles)
        h2 = time.perf_counter()
        b0.record()
        for i in range(R):
            fns[(i + self.warm) % n]()
        b1.record()
        h3 = time.perf_counter()
        t.cuda.synchronize()
        host_ms = max(h1 - h0, h3 - h2) * 1e3
        return {
            "us_each": each[R // 2], "us_each_mean": sum(each) / R,
            "us_each_p10": each[R // 10], "us_each_p90": each[(9 * R) // 10],
            "us_batch": b0.elapsed_time(b1) * 1e3 / R,
            "host_enqueue_ms": host_ms, "sleep_ms": self.sleep_ms,
            "host_starved": host_ms > 0.9 * self.sleep_ms,
            "rotation": n, "reps": R,
        }


# ------------------------------------------------------------------------------------------------ cells
class Cells:
    """Builds each cell's callables. Each builder returns (meta, [fn per rotation copy]) or None."""

    def __init__(self, torch, ext, device, l2, ckpt, ncu):
        self.torch, self.ext, self.dev, self.l2, self.ckpt, self.ncu = torch, ext, device, l2, ckpt, ncu
        t = torch
        # MoE scratch, sized exactly like BlockSparseMLP.load (bszn_rows = MAX_BSZN * top-k), so
        # exl3_moe_coop_prepare derives the served slots_max / rows_max / counter layout
        S = MAX_BSZN * TOPK
        self.moe_scr = dict(
            had_g=t.empty((S, HIDDEN), dtype=t.half, device=device),
            had_u=t.empty((S, HIDDEN), dtype=t.half, device=device),
            gu_g=t.empty((S, MOE_I), dtype=t.half, device=device),
            gu_u=t.empty((S, MOE_I), dtype=t.half, device=device),
            act=t.empty((S, MOE_I), dtype=t.half, device=device),
            d_out=t.empty((S, HIDDEN), dtype=t.float, device=device),
            ctr=t.zeros((S * (MOE_I // 128) + MAX_BSZN * (HIDDEN // 128) + 2 * S + 3,), dtype=t.int, device=device),
            out=t.empty((MAX_BSZN, HIDDEN), dtype=t.float, device=device),
        )

    # ---- dense exl3_gemm (BC_LinearEXL3::run)
    def gemm(self, mats, rows, force_sms=0):
        """mats: list of (trellis, suh, svh) with identical shapes, one per rotation copy."""
        t, ext = self.torch, self.ext
        k = mats[0][0].shape[0] * 16
        n = mats[0][0].shape[1] * 16
        x = (t.randn((rows, k), device=self.dev) * 0.5).half()
        y = t.empty((rows, n), dtype=t.half, device=self.dev)
        xh = t.empty_like(x)
        tags = []

        def mk(tr, suh, svh):
            def f():
                tags.append(ext.exl3_gemm(x, tr, y, suh, xh, svh, -1, False, True, force_sms))
                if len(tags) > 4:
                    del tags[:-1]
            return f
        return [mk(*m) for m in mats], tags

    # ---- sliced exl3_mgemm (GatedDeltaNet.project_qkvz_sliced)
    def mgemm(self, sources_per_copy, rows, force_sms=0):
        """sources_per_copy: list over rotation copies of [(trellis, suh, svh), ...] (one entry per source
        matrix, same k and K). Mirrors SlicedMultiLinear.__init__ + project_qkvz_sliced exactly."""
        t, ext = self.torch, self.ext
        srcs0 = sources_per_copy[0]
        K = srcs0[0][0].shape[-1] // 16
        k = srcs0[0][0].shape[0] * 16
        widths = [s[0].shape[1] * 16 for s in srcs0]
        width = math.gcd(*widths)
        assert width % 128 == 0 and width >= 256, widths
        x3 = (t.randn((1, rows, k), device=self.dev) * 0.5).half()
        xh = t.empty((len(srcs0), rows, k), dtype=t.half, device=self.dev)
        outs = [t.empty((rows, w), dtype=t.float, device=self.dev) for w in widths]
        fns = []
        tags = []
        for srcs in sources_per_copy:
            tp, sp, tg, off, st, hs = [], [], [], [], [], []
            for i, (tr, suh, svh) in enumerate(srcs):
                n_i = tr.shape[1] * 16
                for n0 in range(0, n_i, width):
                    tp.append(tr.data_ptr() + (n0 // 16) * 16 * K * tr.element_size())
                    sp.append(svh.data_ptr() + n0 * svh.element_size())
                    tg.append(i); off.append(n0); st.append(n_i); hs.append(i)
            S = len(tg)
            ptr_tr = t.tensor(tp, dtype=t.long, device=self.dev)
            ptr_svh = t.tensor(sp, dtype=t.long, device=self.dev)
            ptr_suh = t.tensor([s[1].data_ptr() for s in srcs], dtype=t.long, device=self.dev)
            size_n = t.full((S,), width, dtype=t.int32, device=self.dev)
            n_stride = t.tensor(st, dtype=t.int32, device=self.dev)
            had_src = t.tensor(hs, dtype=t.int32, device=self.dev)
            c_ptrs = t.tensor([outs[tg[j]].data_ptr() + off[j] * 4 for j in range(S)], dtype=t.long, device=self.dev)
            carrier = t.zeros((S, 1, width), dtype=t.float, device=self.dev).expand(S, rows, width)
            nsrc = len(srcs)

            def f(ptr_tr=ptr_tr, ptr_suh=ptr_suh, ptr_svh=ptr_svh, carrier=carrier, size_n=size_n,
                  c_ptrs=c_ptrs, n_stride=n_stride, had_src=had_src, nsrc=nsrc):
                tags.append(ext.exl3_mgemm(x3, ptr_tr, carrier, ptr_suh, xh, ptr_svh, None, None, K, -1,
                                           False, True, -1, -1, force_sms, 1, size_n, c_ptrs, n_stride,
                                           had_src, nsrc))
                if len(tags) > 4:
                    del tags[:-1]
            fns.append(f)
        return fns, tags

    # ---- routed MoE coop (BC_BlockSparseMLP::run_bszN minus the shared expert)
    def moe_tables(self, experts):
        """experts: list of 512 dicts {gate|up|down: (trellis, suh, svh)}; returns pointer tables."""
        t = self.torch
        tab = {}
        for p in ("gate", "up", "down"):
            for j, s in enumerate(("trellis", "suh", "svh")):
                tab[f"{p}_{s}"] = t.tensor([e[p][j].data_ptr() for e in experts], dtype=t.long, device=self.dev)
        K = experts[0]["up"][0].shape[-1] // 16
        Kd = experts[0]["down"][0].shape[-1] // 16
        eb = sum(x.numel() * x.element_size() for p in ("gate", "up", "down") for x in experts[0][p])
        return tab, K, Kd, eb

    def moe(self, tab, K, Kd, rows, D, seed=0):
        """D distinct experts per call (D >= 10, D <= 10*rows): pool of D experts from a walk over a random
        permutation of the 512; row r takes pool[(10r + j) % D], so every row's 10 picks are distinct and the
        union is exactly D. 64 precomputed routings; consecutive calls touch disjoint experts."""
        t, ext = self.torch, self.ext
        assert TOPK <= D <= TOPK * rows and rows <= MAX_BSZN
        rng = random.Random(1000 * rows + D + seed)
        perm = list(range(N_EXPERTS))
        rng.shuffle(perm)
        sets = []
        nsets = 1 if self.ncu else 64
        for c in range(nsets):
            pool = [perm[(c * D + i) % N_EXPERTS] for i in range(D)]
            rng.shuffle(pool)
            sel = [[pool[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(rows)]
            sets.append(t.tensor(sel, dtype=t.long, device=self.dev))
        w = t.rand((rows, TOPK), device=self.dev) + 0.1
        rw = (w / w.sum(-1, keepdim=True)).half().contiguous()
        x = (t.randn((rows, HIDDEN), device=self.dev) * 0.5).half().contiguous()
        s = self.moe_scr

        def mk(sel):
            def f():
                ext.exl3_moe_coop(x, sel, rw, -1, -1, HIDDEN,
                                  tab["gate_trellis"], tab["gate_suh"], tab["gate_svh"],
                                  tab["up_trellis"], tab["up_suh"], tab["up_svh"],
                                  tab["down_trellis"], tab["down_suh"], tab["down_svh"],
                                  None, None, None, K, K, Kd, False, True, 0, 0.0, True,
                                  s["had_g"], s["had_u"], s["gu_g"], s["gu_u"], s["act"], s["d_out"],
                                  s["ctr"], s["out"], None, None)
            return f
        return [mk(sel) for sel in sets]


# ------------------------------------------------------------------------------------------------ plan
GEMM_N = [2560, 5120, 10240, 20480, 40960, 65536]
KS = [2, 3, 4, 6]
ROWS = [1, 4, 16]
MGEMM_SLICES = [1, 2, 4, 8, 16, 32]          # single source of 2048*s columns = s slices of 2048
MOE_D = {1: [10], 2: [20], 4: [10, 20, 28, 40], 8: [40, 80], 16: [10, 20, 40, 77, 160]}
NCU_GEMM_N = [10240, 65536]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["time", "ncu", "dryrun"], default="time")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--variant", default="served", help="label only; the env is set by the caller")
    ap.add_argument("--parts", default="gemm,mgemm,moe,real",
                    help="comma list of gemm (synthetic sweeps + SM sweep), mgemm, moe, real (anchors)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    parts = set(a.parts.split(","))
    env = {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "EXLLAMA", "CUDA_"))}

    if a.mode == "dryrun":
        return dryrun(a, parts)

    import torch
    from exllamav3.ext import exllamav3_ext as ext
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    l2 = getattr(props, "L2_cache_size", 96 * 1024 * 1024) or 96 * 1024 * 1024
    nsm = props.multi_processor_count
    meta = {"variant": a.variant, "mode": a.mode, "parts": sorted(parts), "device": str(props), "sms": nsm,
            "l2_bytes": l2, "dram_peak": DRAM_PEAK, "env": env, "torch": torch.__version__,
            "int8_gemv_max_k": ext.exl3_gemv_int8_max_k(0)}
    (a.out / f"meta-{a.mode}-{a.variant}.json").write_text(json.dumps(meta, indent=1))
    log("meta", json.dumps({k: meta[k] for k in ("variant", "mode", "sms", "l2_bytes", "int8_gemv_max_k")}))

    ncu = a.mode == "ncu"
    ckpt = Checkpoint(a.model, dev)
    C = Cells(torch, ext, dev, l2, ckpt, ncu)
    timer = None if ncu else Timer(torch, a.reps, a.warm)
    out_path = a.out / (f"ncu-cells.jsonl" if ncu else f"cells-{a.variant}.jsonl")
    out_f = out_path.open("w")
    counter = {"i": 0}

    def emit(rec):
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

    def measure(label, fns, tags, rec, n_kernels):
        rec = dict(rec, label=label, variant=a.variant)
        rec["weights"] = rec.get("weights")
        try:
            if ncu:
                for i in range(3):
                    fns[0]()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push(f"d0/{label}")
                fns[0]()
                torch.cuda.nvtx.range_pop()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                rec.update(ncu_index=counter["i"], expected_kernels=n_kernels)
                counter["i"] += 1
            else:
                rec.update(timer.run(fns))
                us = rec["us_each"]
                rec["GBps"] = rec["bytes"] / (us * 1e-6) / 1e9
                rec["pct_dram"] = 100 * rec["bytes"] / (us * 1e-6) / DRAM_PEAK
                rec["Tw_per_s"] = rec["weights"] / (us * 1e-6) / 1e12 if rec["weights"] else None
                rec["GBps_batch"] = rec["bytes"] / (rec["us_batch"] * 1e-6) / 1e9
            rec["path_tag"] = tags[-1] if tags else None
            rec["ok"] = True
        except Exception as e:  # noqa: BLE001 - a cell that cannot launch is data, not a crash
            torch.cuda.synchronize()
            rec.update(ok=False, error=repr(e)[:300])
        emit(rec)
        if not ncu and rec.get("ok"):
            log(f"{label:52s} {rec['us_each']:9.2f} us  batch {rec['us_batch']:9.2f}  "
                f"{rec['GBps']:7.0f} GB/s {rec['pct_dram']:5.1f}%  tag {rec['path_tag']}"
                + ("  HOST-STARVED" if rec["host_starved"] else ""))
        elif not rec.get("ok"):
            log(f"{label:52s} FAILED {rec.get('error')}")

    def free():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ================================================================== dense exl3_gemm, synthetic sweep
    if "gemm" in parts:
        n_list = NCU_GEMM_N if ncu else GEMM_N
        for K in KS:
            for n in n_list:
                b = lin_bytes(HIDDEN, n, K)
                cp = 1 if ncu else rot_copies(b, l2)
                tr, suh, svh = synth_linear(torch, HIDDEN, n, K, dev, copies=cp)
                mats = [(tr[i], suh[i], svh[i]) for i in range(cp)]
                for rows in ROWS:
                    fns, tags = C.gemm(mats, rows)
                    base = {"family": "gemm", "source": "synth", "K": K, "rows": rows, "k": HIDDEN, "n": n,
                            "bytes": b, "weights": HIDDEN * n, "act_bytes": 2 * rows * (HIDDEN + n)}
                    measure(f"gemm/synth/K{K}/r{rows}/n{n}", fns, tags, base, 1)
                    # multi-wave / grid knob: force_num_sms (bypasses gemv + autotune -> select_exl3_gemm_kernel)
                    if rows > 1 and n in NCU_GEMM_N and not ncu:
                        for frac in (1.0, 0.75, 0.5, 0.25):
                            sms = max(1, int(round(nsm * frac)))
                            fns, tags = C.gemm(mats, rows, force_sms=sms)
                            measure(f"gemm/synth/K{K}/r{rows}/n{n}/sms{sms}", fns, tags,
                                    dict(base, force_sms=sms), 1)
                del tr, suh, svh, mats
                free()

    # ================================================================== sliced mgemm, synthetic
    if "mgemm" in parts:
        for K in KS:
            # served structure: qkv 10240 + z 6144 -> 8 slices of 2048
            b = lin_bytes(HIDDEN, 10240, K) + lin_bytes(HIDDEN, 6144, K)
            cp = 1 if ncu else rot_copies(b, l2)
            q = synth_linear(torch, HIDDEN, 10240, K, dev, copies=cp)
            z = synth_linear(torch, HIDDEN, 6144, K, dev, copies=cp)
            srcs = [[(q[0][i], q[1][i], q[2][i]), (z[0][i], z[1][i], z[2][i])] for i in range(cp)]
            for rows in ROWS:
                fns, tags = C.mgemm(srcs, rows)
                measure(f"mgemm/synth/K{K}/r{rows}/qkvz", fns, tags,
                        {"family": "mgemm", "source": "synth", "K": K, "rows": rows, "k": HIDDEN, "n": 16384,
                         "slices": 8, "bytes": b, "weights": HIDDEN * 16384,
                         "act_bytes": 2 * rows * HIDDEN + 4 * rows * 16384}, 1)
            del q, z, srcs
            free()
            if ncu:
                continue
            for s in MGEMM_SLICES:   # intercept sweep: one source, s slices of 2048
                n = 2048 * s
                b = lin_bytes(HIDDEN, n, K)
                cp = rot_copies(b, l2)
                tr, suh, svh = synth_linear(torch, HIDDEN, n, K, dev, copies=cp)
                srcs = [[(tr[i], suh[i], svh[i])] for i in range(cp)]
                for rows in ROWS:
                    fns, tags = C.mgemm(srcs, rows)
                    measure(f"mgemm/synth/K{K}/r{rows}/s{s}", fns, tags,
                            {"family": "mgemm", "source": "synth", "K": K, "rows": rows, "k": HIDDEN, "n": n,
                             "slices": s, "bytes": b, "weights": HIDDEN * n,
                             "act_bytes": 2 * rows * HIDDEN + 4 * rows * n}, 1)
                del tr, suh, svh, srcs
                free()

    # ================================================================== MoE coop, synthetic
    def moe_cells(tab, K, Kd, eb, source):
        for rows, ds in MOE_D.items():
            if ncu and rows not in ROWS:
                continue
            for D in ds:
                if ncu and D != TOPK * rows:
                    continue
                fns = C.moe(tab, K, Kd, rows, D)
                measure(f"moe/{source}/K{K}/r{rows}/D{D}", fns, [],
                        {"family": "moe", "source": source, "K": K, "Kd": Kd, "rows": rows, "D": D,
                         "slots": rows * TOPK, "expert_bytes": eb, "bytes": D * eb,
                         "weights": D * 3 * HIDDEN * MOE_I, "act_bytes": 2 * rows * HIDDEN + 4 * rows * HIDDEN},
                        2 if rows == 1 else 3)

    if "moe" in parts:
        for K in KS:
            g = synth_linear(torch, HIDDEN, MOE_I, K, dev, copies=N_EXPERTS)
            u = synth_linear(torch, HIDDEN, MOE_I, K, dev, copies=N_EXPERTS)
            d = synth_linear(torch, MOE_I, HIDDEN, K, dev, copies=N_EXPERTS)
            experts = [{"gate": (g[0][e], g[1][e], g[2][e]), "up": (u[0][e], u[1][e], u[2][e]),
                        "down": (d[0][e], d[1][e], d[2][e])} for e in range(N_EXPERTS)]
            tab, K_, Kd, eb = C.moe_tables(experts)
            moe_cells(tab, K_, Kd, eb, "synth")
            del g, u, d, experts, tab
            free()

    # ================================================================== real anchors
    if "real" in parts:
        # routed experts K=2 (layer 20), K=3 (layer 3), K=4 (MTP layer)
        for K, pfx in EXPERT_LAYERS.items():
            t0 = time.time()
            experts = [{p: ckpt.linear(f"{pfx}.experts.{e}.{p}_proj") for p in ("gate", "up", "down")}
                       for e in range(N_EXPERTS)]
            tab, K_, Kd, eb = C.moe_tables(experts)
            assert K_ == K and Kd == K, (pfx, K_, Kd)
            log(f"loaded {pfx}: 512 experts K={K} {eb} B/expert in {time.time() - t0:.1f}s")
            moe_cells(tab, K_, Kd, eb, "real")
            del experts, tab
            free()
        # dense K=4: out_proj (6144 -> 2560) through exl3_gemm, in_proj qkv+z through sliced mgemm,
        # in_proj_qkv alone through exl3_gemm; synthetic twins at the same rotation count
        nl = len(GDN_LAYERS) if not ncu else 1
        layers = GDN_LAYERS[:nl]
        outp = [ckpt.linear(f"model.language_model.layers.{i}.linear_attn.out_proj") for i in layers]
        qkv = [ckpt.linear(f"model.language_model.layers.{i}.linear_attn.in_proj_qkv") for i in layers]
        zz = [ckpt.linear(f"model.language_model.layers.{i}.linear_attn.in_proj_z") for i in layers]
        Kd4 = outp[0][0].shape[-1] // 16
        assert Kd4 == 4 and qkv[0][0].shape[-1] == 64 and zz[0][0].shape[-1] == 64
        tw_o = synth_linear(torch, 6144, HIDDEN, 4, dev, copies=nl)
        tw_q = synth_linear(torch, HIDDEN, 10240, 4, dev, copies=nl)
        tw_z = synth_linear(torch, HIDDEN, 6144, 4, dev, copies=nl)
        for rows in ROWS:
            for src, mats in (("real", outp), ("twin", [(tw_o[0][i], tw_o[1][i], tw_o[2][i]) for i in range(nl)])):
                if ncu and src == "twin":
                    continue
                fns, tags = C.gemm(mats, rows)
                measure(f"gemm/{src}/K4/r{rows}/out_proj", fns, tags,
                        {"family": "gemm", "source": src, "K": 4, "rows": rows, "k": 6144, "n": HIDDEN,
                         "bytes": lin_bytes(6144, HIDDEN, 4), "weights": 6144 * HIDDEN, "anchor": "out_proj",
                         "act_bytes": 2 * rows * (6144 + HIDDEN)}, 1)
            for src, mats in (("real", qkv), ("twin", [(tw_q[0][i], tw_q[1][i], tw_q[2][i]) for i in range(nl)])):
                if ncu and src == "twin":
                    continue
                fns, tags = C.gemm(mats, rows)
                measure(f"gemm/{src}/K4/r{rows}/in_proj_qkv", fns, tags,
                        {"family": "gemm", "source": src, "K": 4, "rows": rows, "k": HIDDEN, "n": 10240,
                         "bytes": lin_bytes(HIDDEN, 10240, 4), "weights": HIDDEN * 10240, "anchor": "in_proj_qkv",
                         "act_bytes": 2 * rows * (HIDDEN + 10240)}, 1)
            for src, srcs in (("real", [[qkv[i], zz[i]] for i in range(nl)]),
                              ("twin", [[(tw_q[0][i], tw_q[1][i], tw_q[2][i]), (tw_z[0][i], tw_z[1][i], tw_z[2][i])]
                                        for i in range(nl)])):
                if ncu and src == "twin":
                    continue
                fns, tags = C.mgemm(srcs, rows)
                b = lin_bytes(HIDDEN, 10240, 4) + lin_bytes(HIDDEN, 6144, 4)
                measure(f"mgemm/{src}/K4/r{rows}/in_proj_qkvz", fns, tags,
                        {"family": "mgemm", "source": src, "K": 4, "rows": rows, "k": HIDDEN, "n": 16384,
                         "slices": 8, "bytes": b, "weights": HIDDEN * 16384, "anchor": "in_proj_qkvz",
                         "act_bytes": 2 * rows * HIDDEN + 4 * rows * 16384}, 1)
                if src == "real" and not ncu:   # grid knob on the served in_proj path
                    for frac in (1.0, 0.5, 0.25):
                        sms = max(1, int(round(nsm * frac)))
                        fns, tags = C.mgemm(srcs, rows, force_sms=sms)
                        measure(f"mgemm/real/K4/r{rows}/in_proj_qkvz/sms{sms}", fns, tags,
                                {"family": "mgemm", "source": "real", "K": 4, "rows": rows, "k": HIDDEN,
                                 "n": 16384, "slices": 8, "bytes": b, "weights": HIDDEN * 16384,
                                 "anchor": "in_proj_qkvz", "force_sms": sms}, 1)
        del outp, qkv, zz, tw_o, tw_q, tw_z
        free()
        # lm_head K=6 (477 MB > L2: no rotation needed) and its synthetic twin
        head = ckpt.linear(LM_HEAD)
        assert head[0].shape[-1] == 96, head[0].shape
        tw_h = synth_linear(torch, HIDDEN, head[0].shape[1] * 16, 6, dev)
        n = head[0].shape[1] * 16
        for rows in ROWS:
            for src, mats in (("real", [head]), ("twin", [tw_h])):
                if ncu and src == "twin":
                    continue
                fns, tags = C.gemm(mats, rows)
                measure(f"gemm/{src}/K6/r{rows}/lm_head", fns, tags,
                        {"family": "gemm", "source": src, "K": 6, "rows": rows, "k": HIDDEN, "n": n,
                         "bytes": lin_bytes(HIDDEN, n, 6), "weights": HIDDEN * n, "anchor": "lm_head",
                         "act_bytes": 2 * rows * (HIDDEN + n)}, 1)
        del head, tw_h
        free()

    out_f.close()
    log(f"done: {out_path}")


def dryrun(a, parts):
    """CPU-only: check every real key/shape the time/ncu modes will load, and print the cell plan."""
    import torch  # noqa: F401 - import check of the served torch
    ck = Checkpoint(a.model, "cpu")
    for K, pfx in EXPERT_LAYERS.items():
        for e in (0, 511):
            for p in ("gate", "up", "down"):
                tr, suh, svh = ck.linear(f"{pfx}.experts.{e}.{p}_proj")
                assert tr.shape[-1] == 16 * K, (pfx, p, tuple(tr.shape))
                exp = (160, 40) if p != "down" else (40, 160)
                assert tuple(tr.shape[:2]) == exp, (pfx, p, tuple(tr.shape))
        eb = sum(x.numel() * x.element_size() for p in ("gate", "up", "down")
                 for x in ck.linear(f"{pfx}.experts.0.{p}_proj"))
        print(f"[dryrun] {pfx}: K={K} expert bytes {eb}")
    for i in (GDN_LAYERS[0], GDN_LAYERS[-1]):
        for nm, shp in (("out_proj", (384, 160, 64)), ("in_proj_qkv", (160, 640, 64)), ("in_proj_z", (160, 384, 64))):
            tr, _, _ = ck.linear(f"model.language_model.layers.{i}.linear_attn.{nm}")
            assert tuple(tr.shape) == shp, (i, nm, tuple(tr.shape))
    tr, _, _ = ck.linear(LM_HEAD)
    assert tuple(tr.shape) == (160, 15520, 96), tuple(tr.shape)
    print(f"[dryrun] dense anchors OK: {len(GDN_LAYERS)} GDN layers, lm_head {tuple(tr.shape)}")
    from exllamav3.ext import exllamav3_ext as ext
    for fn in ("exl3_gemm", "exl3_mgemm", "exl3_moe_coop", "exl3_gemv_int8_max_k"):
        assert hasattr(ext, fn), fn
    print("[dryrun] ext entry points present:", "exl3_gemm exl3_mgemm exl3_moe_coop")
    n = 0
    if "gemm" in parts:
        n += len(KS) * len(GEMM_N) * len(ROWS) + len(KS) * len(NCU_GEMM_N) * 2 * 4
    if "mgemm" in parts:
        n += len(KS) * len(ROWS) * (1 + len(MGEMM_SLICES))
    if "moe" in parts:
        n += len(KS) * sum(len(v) for v in MOE_D.values())
    if "real" in parts:
        n += 3 * sum(len(v) for v in MOE_D.values()) + len(ROWS) * (6 + 3 + 2)
    print(f"[dryrun] parts {sorted(parts)}: {n} timed cells")


if __name__ == "__main__":
    main()
