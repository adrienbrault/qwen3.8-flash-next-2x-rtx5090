#!/usr/bin/env python3
"""TP=2 bounding probe, part 1: kernel scaling s = t(half) / t(full) per decode family, on ONE card (cuda:0).

Runs inside the served image (tabbyapi:stack-r4-e3r2) with the daily EXTRA_ENV. Loads real decoder modules from the
2.50 bpw checkpoint one family at a time (module.load(device), the r521 pattern), builds each tensor-parallel rank's
half with exllamav3's OWN TP import code (tpb_common.py: GatedDeltaNet / Attention / BlockSparseMLP / GatedMLP /
Linear tp_import on an in-process transport), then times full vs rank-0 half vs rank-1 half with CUDA events:

  GDN      GatedDeltaNet.forward (served fused BC_GatedDeltaNetSplit graph path)   shapes 1x4 4x2 4x4
  ATTN     Attention.forward, QSA, 8-bit paged KV at 1k context (BC attention graph path), indexer replicated
                                                                                   shapes 1x4 4x2 4x4 1x1 4x1
  MOE_K2/3 BlockSparseMLP bc.run_bszN (coop V2 routed + embedded shared expert, side-stream overlap as served);
           one K=2 and one K=3 layer; routing replicated (same selection for full and halves)  rows 1 4 8 16
  MTP_*    the MTP draft block's attention and MoE at the draft shapes (optional)   1x1 4x1
  HEAD     lm_head vocab halves (sensitivity only; the primary budget replicates the head)
  HC       gated-residual mix + apply at half hidden (sensitivity only; the design replicates the mixer)

Checks (fail loudly, exit 1, but always write the JSON):
  * every full arm launches only kernel names the served decode launched in R519 (data/r519_*_kernels.json);
    GDN/ATTN/MOE full AND half arms must launch their family's served kernel (coop V2, recurrent GDN, paged decode);
  * row-parallel halves sum to the full output (GDN, ATTN, MOE: rank0 + rank1 == full within tolerance);
  * the image's exllamav3 source files match the hashes of src/ (data/src_sha256.json) - warns if not.

Timing: warm-up (graph capture, coop autotune), then `rounds` interleaved rounds (order alternates) of `iters`
back-to-back calls between two CUDA events, with torch.cuda._sleep pre-filling the stream so the host enqueues
ahead; a sample is flagged host-bound when the host could not stay ahead. Calls rotate over several layers /
routing sets so weights stream from DRAM as in the served step instead of sitting hot in L2. Medians + min/max.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import gc
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpb_common as C  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "data")

TRUNK_SHAPES = ["1x4", "4x2", "4x4"]     # c1 d3 verify, c4 d1 verify, c4 d3 verify
DRAFT_SHAPES = ["1x1", "4x1"]            # MTP draft block at c1 / c4
MOE_ROWS = [1, 4, 8, 16]

REQUIRED_ENV = {"EXL3_MOE_COOP_V2": "1", "EXL3_SHARED_EXPERT_OVERLAP": "1", "EXL3_HC_MIX_V2": "1"}
REQUIRED_KERNEL = {
    "GDN": "cuda_recurrent_gated_delta_rule_kernel_128",
    "ATTN": "_paged_attn_decode_split_kernel",
    "MOE": "exl3_moe_coop_v2_ns::exl3_moe_coop_a_kernel",
    "HC": "gr_v2_dots_kernel",
}
HARD_PATH_FAMILIES = ("GDN", "ATTN", "MOE")

# Steps (one family each; MTP measures two). Each step is independent: a failure is recorded for that step and the
# remaining steps continue in a FRESH process (respawn), so a sticky CUDA error cannot take the other families down.
STEP_ORDER = ("GDN", "ATTN", "MOE_K2", "MOE_K3", "MTP", "HC", "HEAD")
PRIMARY_STEPS = ("GDN", "ATTN", "MOE_K2", "MOE_K3")        # the join needs all four; MTP / HC / HEAD are optional
STEP_FAMILIES = {"GDN": ("GDN",), "ATTN": ("ATTN",), "MOE_K2": ("MOE_K2",), "MOE_K3": ("MOE_K3",),
                 "MTP": ("MTP_ATTN", "MTP_MOE"), "HC": ("HC",), "HEAD": ("HEAD",)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/model")
    ap.add_argument("--out", default="tp_bound_scaling.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ctx", type=int, default=1024, help="attention context (R519 profiled at 1,024)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--sleep-ms-min", type=float, default=10.0)
    ap.add_argument("--profile-calls", type=int, default=10)
    ap.add_argument("--gdn-layers", default="0,1,2,4,5,6,8,9",
                    help="rotated so the working set exceeds L2 (served: every layer streams from DRAM)")
    ap.add_argument("--attn-layers", default="3,7,11,15,19,23,27,31,35,39,43,47")
    ap.add_argument("--moe-candidates", default="0,47,1,46,2,45,3,44,4,43,24,23,25,12,36",
                    help="scanned in order (K read from the safetensors header) for one K=2 and one K=3 layer; on the 2.50bpw mix K=2 is layers 12-36 only (R527 try 2)")
    ap.add_argument("--routing-sets", type=int, default=16)
    ap.add_argument("--max-bsz", type=int, default=4, help="served slots")
    ap.add_argument("--max-history", type=int, default=4,
                    help=">= the verify length 4 (served: max(draft depth policy, default_draft_size 4))")
    ap.add_argument("--families", default=",".join(STEP_ORDER),
                    help="steps to run, any order (run in STEP_ORDER); MOE = MOE_K2,MOE_K3")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--budget-s", type=float, default=550.0,
                    help="wall budget of the whole probe incl. respawns (box_entry.sh kills it at 570 s)")
    ap.add_argument("--deadline-at", type=float, default=0.0, help="absolute deadline (epoch s); set for respawns")
    ap.add_argument("--retried", default="", help="steps already retried once in a fresh process (respawns)")
    ap.add_argument("--no-respawn", action="store_true", help="record a failure and stop instead of respawning")
    a = ap.parse_args(argv)
    a.argv = list(sys.argv[1:] if argv is None else argv)
    return a


def parse_steps(spec: str) -> list:
    want = set()
    for f in (x.strip() for x in spec.split(",")):
        if not f:
            continue
        if f == "MOE":
            want |= {"MOE_K2", "MOE_K3"}
        elif f in STEP_ORDER:
            want.add(f)
        else:
            raise SystemExit(f"unknown family/step {f!r}; known: {', '.join(STEP_ORDER)} (MOE = both K)")
    return [s for s in STEP_ORDER if s in want]


# ------------------------------------------------------------------------------------------------------------------
# GPU primitives (the only CUDA-timing code; the CPU dry-run test replaces this class)
# ------------------------------------------------------------------------------------------------------------------

class Gpu:
    def __init__(self, torch, dev):
        self.torch, self.dev = torch, dev
        props = torch.cuda.get_device_properties(dev)
        self.clock_khz = getattr(props, "clock_rate", None) or 2_400_000
        self.l2_bytes = getattr(props, "L2_cache_size", None)
        self.name = props.name

    def sync(self):
        self.torch.cuda.synchronize(self.dev)

    def sleep_ms(self, ms):
        self.torch.cuda._sleep(int(ms * self.clock_khz))

    def events(self):
        E = self.torch.cuda.Event
        return E(enable_timing=True), E(enable_timing=True)

    def profile_kernels(self, fn, n):
        torch = self.torch
        acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=acts, record_shapes=False, with_stack=False,
                                    profile_memory=False) as prof:
            for i in range(n):
                fn(i)
            self.sync()
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            prof.export_chrome_trace(path)
            with open(path) as f:
                trace = json.load(f)
        finally:
            os.unlink(path)
        return kernel_table_from_trace(trace, n)


def kernel_table_from_trace(trace, calls):
    """Kineto GPU kernel events (cat == kernel), per call: {name: {us, launches}} (as the R465 harness counts)."""
    out = {}
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X" or str(e.get("cat", "")).lower() != "kernel":
            continue
        r = out.setdefault(e["name"], {"us": 0.0, "launches": 0.0})
        r["us"] += float(e.get("dur", 0)) / calls
        r["launches"] += 1.0 / calls
    return out


def normalize_kernel(name: str) -> str:
    """Strip the trailing argument list: 'void f<3, 2>(Params)' -> 'void f<3, 2>'."""
    name = name.strip()
    if not name.endswith(")"):
        return name
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        c = name[i]
        if c == ")":
            depth += 1
        elif c == "(":
            depth -= 1
            if depth == 0:
                return name[:i].strip()
    return name


def kernel_base(name: str) -> str:
    """'void ns::f<3, 2, true>' -> 'ns::f' (template arguments and return type stripped)."""
    n = normalize_kernel(name)
    out, depth = [], 0
    for c in n:
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif depth == 0:
            out.append(c)
    n = "".join(out).strip()
    return n[5:] if n.startswith("void ") else n


def served_kernel_names(data_dir=DATA):
    names = set()
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("r519_") and fn.endswith("_kernels.json"):
            with open(os.path.join(data_dir, fn)) as f:
                for k in json.load(f)["kernels"]:
                    names.add(normalize_kernel(k["name"]))
    return names


def time_arm(gpu, fn, iters, sleep_ms, clock=time.perf_counter):
    s, e = gpu.events()
    gpu.sync()
    gpu.sleep_ms(sleep_ms)
    t0 = clock()
    s.record()
    for i in range(iters):
        fn(i)
    e.record()
    host_ms = (clock() - t0) * 1e3
    e.synchronize()
    gpu_ms = s.elapsed_time(e)
    # host_ahead: every call was enqueued before the GPU finished the pre-fill sleep, so the GPU never waited on the
    # host and gpu_ms is pure device time
    return gpu_ms * 1e3 / iters, host_ms * 1e3 / iters, host_ms <= sleep_ms


def bench(gpu, arms: dict, a, n_rot: int = 1, clock=time.perf_counter):
    """arms: name -> fn(i). Interleaved rounds, alternating order. Returns per-arm stats (us per call).
    n_rot = modules an arm rotates over: the BC modules capture their CUDA graph on the third call of a shape, so
    every rotated module gets >= 4 warm-up calls before anything is timed."""
    for fn in arms.values():
        for i in range(max(a.warmup, 4 * n_rot)):
            fn(i)
    gpu.sync()
    host_est = 0.0
    for fn in arms.values():
        gpu.sync()
        t0 = clock()
        for i in range(8):
            fn(i)
        host_est = max(host_est, (clock() - t0) * 1e3 / 8)
        gpu.sync()
    sleep_ms = min(250.0, max(a.sleep_ms_min, 1.5 * host_est * a.iters))
    samples = {k: [] for k in arms}
    hosts = {k: [] for k in arms}
    ahead = {k: True for k in arms}
    order = list(arms)
    for r in range(a.rounds):
        for k in (order if r % 2 == 0 else order[::-1]):
            g, h, ok = time_arm(gpu, arms[k], a.iters, sleep_ms, clock)
            samples[k].append(g)
            hosts[k].append(h)
            ahead[k] = ahead[k] and ok
    out = {}
    for k in arms:
        v = samples[k]
        out[k] = {"median_us": round(statistics.median(v), 3), "min_us": round(min(v), 3),
                  "max_us": round(max(v), 3), "host_us": round(statistics.median(hosts[k]), 3),
                  "host_ahead": ahead[k], "samples_us": [round(x, 3) for x in v]}
    out["_sleep_ms"] = round(sleep_ms, 2)
    return out


# ------------------------------------------------------------------------------------------------------------------
# Probe context
# ------------------------------------------------------------------------------------------------------------------

class Probe:
    def __init__(self, a, torch, xl, gpu, served_names, stc=None, log=print):
        self.a, self.torch, self.xl, self.gpu, self.log = a, torch, xl, gpu, log
        self.stc = stc
        self.dev = torch.device(a.device)
        self.served = served_names
        self.served_bases = {kernel_base(n) for n in served_names}
        self.res = new_result()
        self.gen = None
        if self.dev.type != "meta":      # meta = the CPU dry run (shapes only)
            self.gen = torch.Generator(device=self.dev)
            self.gen.manual_seed(a.seed)

    # -- helpers -------------------------------------------------------------------------------------------------
    def module_bytes(self, prefix):
        """Checkpoint bytes under a module key (safetensors header; loader/safetensors.py list_tensors)."""
        if self.stc is None:
            return None
        return int(sum(v["n_bytes"] for v in self.stc.list_tensors(prefix, only_serializable=True).values()))

    def expert_K(self, mlp_key):
        """Routed-expert K of a MoE layer from the header: trellis (in/16, out/16, 16*K) of expert 0's up_proj."""
        t = self.stc.list_tensors(f"{mlp_key}.experts.0.up_proj", only_serializable=True)
        shape = t[f"{mlp_key}.experts.0.up_proj.trellis"]["shape"]
        return shape[-1] // 16

    def randn(self, shape, dtype):
        if self.gen is None:
            return self.torch.empty(shape, device=self.dev, dtype=dtype)
        t = self.torch.randn(shape, generator=self.gen, device=self.dev, dtype=self.torch.float)
        return t.to(dtype).contiguous()

    def check_path(self, fam, shape, arm, table, full_arm):
        """Served-path check. Hard (fails the run): the family's served kernel is missing from an arm, or a FULL arm
        launches a kernel whose base name (template arguments stripped) the served R519 decode never launched.
        Template instances R519 did not profile at this exact row count are recorded, not failed."""
        names = {normalize_kernel(n) for n in table}
        req = REQUIRED_KERNEL.get(fam)
        problems, info = [], []
        if req is not None and not any(req in n for n in names):
            problems.append(f"{fam} {shape} {arm}: served kernel '{req}' not launched; got {sorted(names)}")
        if full_arm:
            off = sorted(n for n in names if kernel_base(n) not in self.served_bases)
            if off:
                problems.append(f"{fam} {shape} {arm}: kernels never launched by the served R519 decode: {off}")
            info = sorted(n for n in names if n not in self.served and kernel_base(n) in self.served_bases)
        for p in problems:
            self.log(("PATH PROBLEM: " if fam in HARD_PATH_FAMILIES else "path note: ") + p)
        if fam in HARD_PATH_FAMILIES:
            self.res["path_problems"] += problems
        if info:
            self.res.setdefault("unprofiled_template_instances", {})[f"{fam} {shape} {arm}"] = info
        return problems

    def profile_arms(self, fam, shape, arms, full_name):
        tables = {}
        for name, fn in arms.items():
            try:
                t = self.gpu.profile_kernels(fn, self.a.profile_calls)
            except Exception as e:  # noqa: BLE001
                msg = f"{fam} {shape} {name}: profiler failed: {e!r}"
                self.log("ERROR: " + msg)
                self.res["errors"].append(msg)
                if fam in HARD_PATH_FAMILIES:
                    self.res["path_problems"].append(msg + " (served-path check could not run)")
                continue
            tables[name] = t
            self.check_path(fam, shape, name, t, name == full_name)
        return tables

    def equality(self, fam, shape, full_out, parts):
        rec = C.sum_check(full_out, parts)
        rec.update({"family": fam, "shape": shape})
        if not rec["ok"]:
            self.log(f"EQUALITY FAIL {fam} {shape}: {rec}")
            self.res["equality_failures"].append(rec)
        return rec

    @staticmethod
    def s_from(stats, full="full", halves=("rank0", "rank1")):
        f = stats[full]["median_us"]
        hs = [stats[h]["median_us"] for h in halves if h in stats]
        return round(max(hs) / f, 4) if f > 0 and hs else None

    # -- GDN -----------------------------------------------------------------------------------------------------
    def run_gdn(self, modules):
        torch, xl, a = self.torch, self.xl, self.a
        fam = {"layers": [], "info": {}, "shapes": {}, "kernels": {}, "equality": {}}
        self.res["families"]["GDN"] = fam
        cache_key = 71001
        fulls = []
        for li in [int(x) for x in a.gdn_layers.split(",")]:
            m = modules[f"model.language_model.layers.{li}.linear_attn"]
            ls = xl.GDNLayerState(m, a.max_bsz, a.max_history, cache_key)
            m.recurrent_layers.append(ls)
            m.tp_recurrent_lookup[cache_key] = ls
            m.load(self.dev)
            fulls.append(m)
            fam["layers"].append(li)
        fam["working_set_bytes"] = sum(self.module_bytes(m.key) for m in fulls)
        fam["l2_bytes"] = self.gpu.l2_bytes
        m0 = fulls[0]
        plans = C.gdn_plans(m0, xl.ratio_split)
        halves = [[None] * len(fulls) for _ in plans]
        for r, plan0 in enumerate(plans):
            for j, m in enumerate(fulls):
                plan = {m.key: plan0[m0.key]}
                h, info = C.build_half_gdn(m, plan, self.dev, xl.ext, xl.GatedDeltaNet)
                halves[r][j] = h
                if j == 0:
                    fam["info"][f"rank{r}"] = {"plan": plan[m.key], **info}
        # served decode path = the fused split-projection BC (gated_delta_net.py:687-730) on full AND half modules
        if not all(m.bc_split for m in fulls) or not all(h.bc_split for hs in halves for h in hs):
            raise RuntimeError("GDN: fused BC_GatedDeltaNetSplit path not built (full "
                               f"{[m.bc_split for m in fulls]}, halves {[h.bc_split for hs in halves for h in hs]})")
        fam["info"]["full"] = {"num_k_heads": m0.num_k_heads, "num_v_heads": m0.num_v_heads,
                               "fdim_qkv": m0.fdim_qkv, "bc_split": bool(m0.bc_split),
                               "K_qkv_z_o": [m0.qkv_proj.inner.K, m0.z_proj.inner.K, m0.o_proj.inner.K],
                               "gate_activation": m0.norm.gate_activation}
        self.log("GDN " + json.dumps(fam["info"]))

        def zero_states(ms):
            for mm in ms:
                for ls in mm.recurrent_layers:
                    ls.conv_state.zero_()
                    ls.recurrent_state.zero_()

        for sk in TRUNK_SHAPES:
            b, s = C.parse_shape(sk)
            p = C.gdn_params(xl.GDNState, xl.get_slot_tensor, cache_key, b, history=True)
            xs = [self.randn((b, s, m0.hidden_size), torch.half) for _ in fulls]
            zero_states([fulls[0], halves[0][0], halves[1][0]])
            yf = fulls[0].forward(xs[0], p).clone()
            yr = [halves[r][0].forward(xs[0], p).clone() for r in range(len(plans))]
            fam["equality"][sk] = self.equality("GDN", sk, yf, yr)

            def arm(ms):
                return lambda i: ms[i % len(ms)].forward(xs[i % len(ms)], p)
            arms = {"full": arm(fulls), **{f"rank{r}": arm(halves[r]) for r in range(len(plans))}}
            st = bench(self.gpu, arms, a, n_rot=len(fulls))
            st["s"] = self.s_from(st)
            fam["shapes"][sk] = st
            one = {"full": lambda i: fulls[0].forward(xs[0], p),
                   **{f"rank{r}": (lambda i, r=r: halves[r][0].forward(xs[0], p)) for r in range(len(plans))}}
            fam["kernels"][sk] = self.profile_arms("GDN", sk, one, "full")
            self.log(f"GDN {sk}: full {st['full']['median_us']} us, halves "
                     f"{[st[f'rank{r}']['median_us'] for r in range(len(plans))]} -> s {st['s']}")
        fam["complete"] = True
        for m in fulls:
            m.unload()
        del halves, fulls

    # -- attention -----------------------------------------------------------------------------------------------
    def run_attn(self, modules, keys, shapes, fam_name):
        torch, xl, a = self.torch, self.xl, self.a
        fam = {"keys": keys, "info": {}, "shapes": {}, "kernels": {}, "equality": {}}
        self.res["families"][fam_name] = fam
        max_q = max(C.parse_shape(k)[1] for k in shapes)
        tokens = C.attn_cache_tokens(a.max_bsz, a.ctx, max_q, xl.PAGE_SIZE)
        fulls = []
        for k in keys:
            m = modules[k]
            m.load(self.dev)
            fulls.append(m)
        fam["working_set_bytes"] = sum(self.module_bytes(m.key) for m in fulls)
        fam["l2_bytes"] = self.gpu.l2_bytes
        m0 = fulls[0]
        plans = C.attn_plans(m0, xl.ratio_split)
        halves = [[None] * len(fulls) for _ in plans]
        for r, plan0 in enumerate(plans):
            for j, m in enumerate(fulls):
                plan = {m.key: plan0[m0.key]}
                h, info = C.build_half_attn(m, plan, self.dev, xl.Attention)
                halves[r][j] = h
                if j == 0:
                    fam["info"][f"rank{r}"] = {"plan": plan[m.key], **info}
        fam["info"]["full"] = {"num_q_heads": m0.num_q_heads, "num_kv_heads": m0.num_kv_heads,
                               "head_dim": m0.head_dim, "q_out": m0.q_proj.out_features,
                               "K_q_k_v_o": [m0.q_proj.inner.K, m0.k_proj.inner.K, m0.v_proj.inner.K,
                                             m0.o_proj.inner.K],
                               "indexer": m0.qsa_indexer is not None,
                               "sparse_threshold": m0.qsa_indexer.sparse_threshold() if m0.qsa_indexer else None,
                               "cache_tokens": tokens, "ctx": a.ctx}
        self.log(f"{fam_name} " + json.dumps(fam["info"]))
        cid = [73000]

        def new_cache(mod):
            cid[0] += 1
            cl = xl.CacheLayer_qsa_quant(None, mod, cid[0], tokens, 8, 8)
            cl.alloc(self.dev)
            return cl
        caches = {id(m): new_cache(m) for m in fulls + [h for hs in halves for h in hs]}

        def zero_cache(cl):
            for attr in ("qk", "qv", "sk", "sv", "raw_k", "pooled"):
                t = getattr(cl, attr, None)
                if t is not None:
                    t.zero_()

        for sk in shapes:
            b, s = C.parse_shape(sk)
            params = {id(m): C.attn_params(torch, xl.prepare_for_attn, caches[id(m)], b, s, a.ctx, xl.PAGE_SIZE)
                      for m in fulls + [h for hs in halves for h in hs]}
            xs = [self.randn((b, s, m0.hidden_size), torch.half) for _ in fulls]
            for mm in (fulls[0], halves[0][0], halves[1][0]):
                zero_cache(caches[id(mm)])
            of = fulls[0].forward(xs[0], params[id(fulls[0])]).clone()
            orr = [halves[r][0].forward(xs[0], params[id(halves[r][0])]).clone() for r in range(len(plans))]
            fam["equality"][sk] = self.equality(fam_name, sk, of, orr)

            def arm(ms):
                return lambda i: ms[i % len(ms)].forward(xs[i % len(ms)], params[id(ms[i % len(ms)])])
            arms = {"full": arm(fulls), **{f"rank{r}": arm(halves[r]) for r in range(len(plans))}}
            st = bench(self.gpu, arms, a, n_rot=len(fulls))
            st["s"] = self.s_from(st)
            fam["shapes"][sk] = st
            one = {"full": lambda i: fulls[0].forward(xs[0], params[id(fulls[0])]),
                   **{f"rank{r}": (lambda i, r=r: halves[r][0].forward(xs[0], params[id(halves[r][0])]))
                      for r in range(len(plans))}}
            fam["kernels"][sk] = self.profile_arms("ATTN", sk, one, "full")
            self.log(f"{fam_name} {sk}: full {st['full']['median_us']} us, halves "
                     f"{[st[f'rank{r}']['median_us'] for r in range(len(plans))]} -> s {st['s']}")
        fam["complete"] = True
        for m in fulls:
            m.unload()
        del halves, fulls, caches

    # -- MoE -----------------------------------------------------------------------------------------------------
    def coop_routed(self, mlp, y, sel, rw, scratch):
        """ext.exl3_moe_coop without a shared output: rot + stage A + stage B only (the r521 'routed' arm, run on
        the box in R521; argument order exl3_moe_coop.cuh:128-157)."""
        cfg = mlp.experts_cfg
        mg = mlp.multi_gate if mlp.gated else mlp.multi_up
        mu, md = mlp.multi_up, mlp.multi_down
        act = 3 if mlp.activation_fn == "swiglu_oai" else 1 if mlp.activation_fn == "gelu" else \
            2 if mlp.activation_fn == "relu2" else 0
        had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out_f = scratch
        self.xl.ext.exl3_moe_coop(
            y, sel, rw, cfg.min_expert, cfg.max_expert, int(cfg.yh.shape[-1]),
            mg.ptrs_trellis, mg.ptrs_suh, mg.ptrs_svh,
            mu.ptrs_trellis, mu.ptrs_suh, mu.ptrs_svh,
            md.ptrs_trellis, md.ptrs_suh, md.ptrs_svh,
            None, None, None,
            mg.K, mu.K, md.K, bool(mu.mcg), bool(mu.mul1), act, float(mlp.act_limit), bool(mlp.gated),
            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out_f,
            None, None,
        )

    def coop_scratch(self, mlp):
        torch, MAX_BSZN = self.torch, self.xl.MAX_BSZN
        cfg = mlp.experts_cfg
        bszn_rows = MAX_BSZN * mlp.num_experts_per_tok
        Hi, I, Ho = int(cfg.yh.shape[-1]), int(cfg.interm_u.shape[-1]), int(cfg.out_d.shape[-1])
        return (torch.empty_like(cfg.yh),
                torch.empty((bszn_rows, Hi), dtype=torch.half, device=self.dev),
                torch.empty_like(cfg.interm_g), torch.empty_like(cfg.interm_u),
                torch.empty_like(cfg.interm_a), torch.empty_like(cfg.out_d),
                torch.zeros((bszn_rows * (I // 128) + MAX_BSZN * (Ho // 128) + 2 + (bszn_rows + 1) + bszn_rows,),
                            dtype=torch.int, device=self.dev),
                torch.empty((MAX_BSZN, mlp.hidden_size), dtype=torch.float, device=self.dev))

    def run_moe_layer(self, mlp, fam_name, rows_list, overlap=True):
        torch, xl, a = self.torch, self.xl, self.a
        fam = {"key": mlp.key, "info": {}, "rows": {}, "kernels": {}, "equality": {}, "overlap": {},
               "working_set_bytes": self.module_bytes(mlp.key), "l2_bytes": self.gpu.l2_bytes}
        self.res["families"][fam_name] = fam
        plans = C.moe_plans(mlp, xl.ratio_split)
        halves = []
        for r, plan in enumerate(plans):
            h, info = C.build_half_moe(mlp, plan, self.dev, xl.BlockSparseMLP)
            halves.append(h)
            fam["info"][f"rank{r}"] = {"plan": {k: v for k, v in plan.items()}, **info}
            # the half must run the SAME fused path as the served full layer (coop V2 + embedded shared expert)
            if bool(h.bc_sh_exp) != bool(mlp.bc_sh_exp) or h.bc is None:
                raise RuntimeError(f"{fam_name} rank{r}: half BC path differs (bc {h.bc is not None}, bc_sh_exp "
                                   f"{h.bc_sh_exp} vs full {mlp.bc_sh_exp})")
            if (h.experts_cfg.min_expert, h.experts_cfg.max_expert) != \
                    (mlp.experts_cfg.min_expert, mlp.experts_cfg.max_expert):
                raise RuntimeError(f"{fam_name} rank{r}: expert range differs from the full layer")
        sh = mlp.shared_experts
        fam["info"]["full"] = {
            "K_gate_up_down": [mlp.multi_gate.K if mlp.gated else None, mlp.multi_up.K, mlp.multi_down.K],
            "intermediate_size": mlp.intermediate_size, "num_experts": mlp.num_experts,
            "topk": mlp.num_experts_per_tok, "bc_sh_exp": bool(mlp.bc_sh_exp),
            "shared_interm": sh.ups[0].out_features if sh is not None else None,
            "shared_K_gate_up_down": [sh.gates[0].inner.K, sh.ups[0].inner.K, sh.downs[0].inner.K] if sh else None}
        fam["info"]["split"] = self.moe_split(mlp, plans)
        self.log(f"{fam_name} " + json.dumps(fam["info"], default=str))
        assert mlp.bc is not None and all(h.bc is not None for h in halves), "fused decode BC not built"
        out_bszn = mlp.experts_cfg.out_bszn
        scratch = self.coop_scratch(mlp) if overlap else None
        sh_buf = torch.empty((1, xl.MAX_BSZN, mlp.hidden_size), dtype=torch.float, device=self.dev)
        self.prewarm_shared(mlp, halves, rows_list, sh_buf)
        for rows in rows_list:
            sets = []
            for _ in range(a.routing_sets):
                y = self.randn((rows, mlp.hidden_size), torch.half)
                sel, rw = mlp.routing_fn(rows, mlp.routing_cfg, y, {})
                sets.append((y, sel.clone(), rw.clone()))
            y0, s0, w0 = sets[0]
            mlp.bc.run_bszN(y0, s0, w0)
            self.gpu.sync()
            of = out_bszn[:rows].clone()
            parts = []
            for h in halves:
                h.bc.run_bszN(y0, s0, w0)
                self.gpu.sync()
                parts.append(h.experts_cfg.out_bszn[:rows].clone())
            fam["equality"][str(rows)] = self.equality(fam_name, str(rows), of, parts)

            def arm(mod):
                return lambda i: mod.bc.run_bszN(*sets[i % len(sets)])
            arms = {"full": arm(mlp), **{f"rank{r}": arm(h) for r, h in enumerate(halves)}}
            if overlap:
                arms["routed"] = lambda i: self.coop_routed(mlp, *sets[i % len(sets)], scratch)
                if sh is not None and sh.bc is not None:
                    arms["shared"] = lambda i: sh.bc.run_bszN(sets[i % len(sets)][0].unsqueeze(0),
                                                             sh_buf[:, :rows])
            st = bench(self.gpu, arms, a, n_rot=1)
            st["s"] = self.s_from(st)
            fam["rows"][str(rows)] = st
            if overlap and "routed" in st and "shared" in st:
                hid = st["routed"]["median_us"] + st["shared"]["median_us"] - st["full"]["median_us"]
                fam["overlap"][str(rows)] = {"hidden_us": round(max(0.0, hid), 3),
                                             "residual_us": round(st["full"]["median_us"]
                                                                  - st["routed"]["median_us"], 3)}
            one = {"full": lambda i: mlp.bc.run_bszN(y0, s0, w0),
                   **{f"rank{r}": (lambda i, h=h: h.bc.run_bszN(y0, s0, w0)) for r, h in enumerate(halves)}}
            fam["kernels"][str(rows)] = self.profile_arms("MOE", str(rows), one, "full")
            self.log(f"{fam_name} rows {rows}: full {st['full']['median_us']} us, halves "
                     f"{[st[f'rank{r}']['median_us'] for r in range(len(halves))]} -> s {st['s']}")
        fam["allocator"] = self.allocator_segments()
        fam["complete"] = True
        del halves

    def moe_split(self, mlp, plans):
        """The MoE tensor split the halves were built with (moe_plans: 128-channel units of the expert intermediate
        width, ratio_split). With 640 channels = 5 units the ranks get 3 + 2 units = 384 + 256 channels, so the
        critical rank holds 60 % of the expert and shared-expert weights; s = max(rank0, rank1) / full already
        measures that rank, and 0.6 is the structural floor of s for this layer (not 0.5)."""
        sh = mlp.shared_experts
        out = {"unit_channels": 128, "moe_intermediate": mlp.intermediate_size,
               "moe": {f"rank{r}": list(p[mlp.key][:2]) for r, p in enumerate(plans)}}
        if sh is not None:
            out["shared_intermediate"] = sh.ups[0].out_features
            out["shared"] = {f"rank{r}": list(p[sh.key][:2]) for r, p in enumerate(plans)}
        widths = [b - a for a, b in out["moe"].values()]
        out["critical_rank"] = f"rank{widths.index(max(widths))}"
        out["critical_share"] = round(max(widths) / mlp.intermediate_size, 4)
        return out

    def prewarm_shared(self, mlp, halves, rows_list, sh_buf):
        """Run every shared expert once per row count on the CURRENT stream before the first overlapped call, as R521
        did (its non-overlap BC ran first). BC_GatedMLP::run_bszN_gr allocates its per-row-count scratch lazily
        (mlp.cpp:27-32); on the first call through BC_BlockSparseMLP with EXL3_SHARED_EXPERT_OVERLAP=1 that happens
        under CUDAStreamGuard(getStreamFromExternal(shared_stream)) (blocksparse_mlp.cpp:113-115), so torch's
        allocator (expandable segments, exllamav3/__init__.py:35) opens a segment owned by the side stream, and
        ~BC_BlockSparseMLP destroys that stream (blocksparse_mlp.cpp:322) while the segment is still cached. The
        next torch.cuda.empty_cache() then synchronizes the dead stream: R527's 'invalid device context'. The warm-up
        writes only the probe's own sh_buf; kernels, graphs and arithmetic of the timed calls are unchanged."""
        for rows in rows_list:
            x = self.randn((1, rows, mlp.hidden_size), self.torch.half)
            for mod in [mlp] + list(halves):
                if mod.shared_experts is not None and mod.shared_experts.bc is not None:
                    mod.shared_experts.bc.run_bszN(x, sh_buf[:, :rows])
        self.gpu.sync()

    def allocator_segments(self):
        """Caching-allocator segments by owning stream (evidence for the R527 mechanism: after prewarm_shared no
        segment may belong to a side stream)."""
        try:
            snap = self.torch.cuda.memory_snapshot()
            cur = int(self.torch.cuda.current_stream(self.dev).cuda_stream)
            side = [sg for sg in snap if int(sg.get("stream", 0)) not in (0, cur)]
            return {"segments": len(snap), "expandable": any(sg.get("is_expandable") for sg in snap),
                    "side_stream_segments": len(side),
                    "side_stream_bytes": int(sum(sg.get("total_size", 0) for sg in side))}
        except Exception as e:  # noqa: BLE001  (diagnostic only)
            return {"error": repr(e)}

    def run_moe(self, modules, K):
        """One layer with routed-expert K (2 or 3), the first of --moe-candidates that has it. No empty_cache after
        the unload (see prewarm_shared); the freed blocks are reused by the next family."""
        a = self.a
        tried = []
        self.res.setdefault("moe_layers_tried", {})[f"K{K}"] = tried
        for li in [int(x) for x in a.moe_candidates.split(",")]:
            key = f"model.language_model.layers.{li}.mlp"
            mlp = modules[key]
            try:
                k, src = self.expert_K(key), "header"
            except Exception as e:  # noqa: BLE001  (unexpected tensor naming: load the layer and ask it)
                self.log(f"MoE layer {li}: K not readable from the header ({e!r}); loading the layer")
                mlp.load(self.dev)
                k, src = mlp.multi_up.K, "loaded"
                if k != K:
                    mlp.unload()
            tried.append((li, k))
            self.log(f"MoE layer {li}: expert K = {k} ({src})")
            if k != K:
                continue
            if src == "header":
                mlp.load(self.dev)
            try:
                if mlp.multi_up.K != K:
                    raise RuntimeError(f"MoE layer {li}: loaded K {mlp.multi_up.K} != header K {K}")
                self.run_moe_layer(mlp, f"MOE_K{K}", MOE_ROWS)
            finally:
                mlp.unload()
            return
        raise RuntimeError(f"MoE: no K={K} layer among --moe-candidates {a.moe_candidates} (tried {tried})")

    # -- MTP draft block ----------------------------------------------------------------------------------------
    def run_mtp(self, config):
        mtp = self.xl.Model.from_config(config, component="mtp")
        mods = {m.key: m for m in mtp}
        ak, mk = "mtp.layers.0.self_attn", "mtp.layers.0.mlp"
        if ak not in mods or mk not in mods:
            raise KeyError(f"MTP block modules not found; have {sorted(k for k in mods if k.startswith('mtp.'))[:20]}")
        self.run_attn(mods, [ak], DRAFT_SHAPES, "MTP_ATTN")
        mlp = mods[mk]
        mlp.load(self.dev)
        self.run_moe_layer(mlp, "MTP_MOE", [1, 4])
        mlp.unload()

    # -- hyper-connection mixer (sensitivity) -------------------------------------------------------------------
    def run_hc(self, modules):
        torch, xl = self.torch, self.xl
        fam = {"info": {}, "shapes": {}, "kernels": {}}
        self.res["families"]["HC"] = fam
        keys = ["model.language_model.layers.0.attn_hyper_connection",
                "model.language_model.layers.0.mlp_hyper_connection"]
        fulls = []
        for k in keys:
            m = modules[k]
            m.load(self.dev)
            fulls.append(m)
        H, D = fulls[0].hc_mult, fulls[0].hidden_size
        halves = [C.build_half_hc(m, 0, D // 2, xl.GatedResidual) for m in fulls]
        fam["info"] = {"hc_mult": H, "hidden": D, "rank": fulls[0].rank, "half_hidden": D // 2,
                       "mix_v2": bool(xl.GatedResidual.MIX_V2), "mix_v2_min_r": xl.GatedResidual.MIX_V2_MIN_R}
        for sk in TRUNK_SHAPES + DRAFT_SHAPES:
            b, s = C.parse_shape(sk)
            xf = [self.randn((b, s, H, D), torch.float) for _ in fulls]
            yf = [self.randn((b, s, D), torch.float) for _ in fulls]
            xh = [self.randn((b, s, H, D // 2), torch.float) for _ in fulls]
            yh = [self.randn((b, s, D // 2), torch.float) for _ in fulls]

            def site(ms, xs, ys):
                def f(i):
                    j = i % len(ms)
                    post, comb, _mixed = ms[j].mix(xs[j], {})
                    ms[j].apply_(xs[j], ys[j], post, comb, {})
                return f
            arms = {"full": site(fulls, xf, yf), "rank0": site(halves, xh, yh)}
            st = bench(self.gpu, arms, self.a, n_rot=len(fulls))
            st["s"] = self.s_from(st, halves=("rank0",))
            fam["shapes"][sk] = st
            fam["kernels"][sk] = self.profile_arms("HC", sk, {"full": site(fulls[:1], xf, yf),
                                                              "rank0": site(halves[:1], xh, yh)}, "full")
            self.log(f"HC {sk}: full {st['full']['median_us']} us, half {st['rank0']['median_us']} -> s {st['s']}")
        fam["complete"] = True
        for m in fulls:
            m.unload()

    # -- LM head (sensitivity) ----------------------------------------------------------------------------------
    def run_head(self, modules):
        torch, xl = self.torch, self.xl
        fam = {"info": {}, "rows": {}, "kernels": {}, "equality": {}}
        self.res["families"]["HEAD"] = fam
        lm = modules["lm_head"]
        lm.load(self.dev)
        n = lm.out_features
        (a0, b0), _ = C.vocab_ranges(n, xl.ratio_split)
        half = C.build_half_linear_cols(lm, a0, b0, self.dev, xl.Linear)
        head_n = int(os.environ.get("EXL3_MTP_HEAD_N", "65536"))
        head_n = min(head_n, n) // 128 * 128
        pr = C.build_half_linear_cols(lm, 0, head_n, self.dev, xl.Linear)
        (pa, pb), _ = C.vocab_ranges(head_n, xl.ratio_split)
        prh = C.build_half_linear_cols(lm, pa, pb, self.dev, xl.Linear)
        fam["info"] = {"vocab": n, "K": lm.inner.K, "half_cols": [a0, b0], "draft_head_n": head_n,
                       "draft_half_cols": [pa, pb]}
        for rows, full_m, half_m, lo, hi in ((4, lm, half, a0, b0), (16, lm, half, a0, b0),
                                             (1, pr, prh, pa, pb), (4, pr, prh, pa, pb)):
            tag = f"{'lm' if full_m is lm else 'draft'}_{rows}"
            x = self.randn((1, rows, lm.in_features), torch.half)
            lf = full_m.forward(x, {}).float()
            lh = half_m.forward(x, {}).float()
            rec = C.sum_check(lf[..., lo:hi], [lh])
            fam["equality"][tag] = rec
            if not rec["ok"]:
                self.log(f"WARN head equality {tag}: {rec}")
            arms = {"full": lambda i, m=full_m, x=x: m.forward(x, {}),
                    "rank0": lambda i, m=half_m, x=x: m.forward(x, {})}
            st = bench(self.gpu, arms, self.a)
            st["s"] = self.s_from(st, halves=("rank0",))
            fam["rows"][tag] = st
            fam["kernels"][tag] = self.profile_arms("HEAD", tag, arms, "full")
            self.log(f"HEAD {tag}: full {st['full']['median_us']} us, half {st['rank0']['median_us']} -> s {st['s']}")
        fam["complete"] = True
        lm.unload()


# ------------------------------------------------------------------------------------------------------------------
# Wiring
# ------------------------------------------------------------------------------------------------------------------

class XL:
    """Every exllamav3 symbol the probe touches, resolved in one place (the CPU dry-run test binds these against
    the real classes in src/)."""

    def __init__(self):
        from exllamav3 import Config, Model
        from exllamav3.ext import exllamav3_ext as ext
        from exllamav3.constants import PAGE_SIZE
        from exllamav3.util.misc import ratio_split
        from exllamav3.modules.gated_delta_net import GatedDeltaNet, GDNLayerState, GDNState
        from exllamav3.cache.recurrent_util import _get_slot_tensor
        from exllamav3.modules.attn import Attention, prepare_for_attn
        from exllamav3.cache.qsa import CacheLayer_qsa_quant
        from exllamav3.modules.block_sparse_mlp import BlockSparseMLP, MAX_BSZN
        from exllamav3.modules.linear import Linear
        from exllamav3.modules.hyperconnections import GatedResidual
        self.Config, self.Model, self.ext, self.PAGE_SIZE, self.ratio_split = Config, Model, ext, PAGE_SIZE, ratio_split
        self.GatedDeltaNet, self.GDNLayerState, self.GDNState = GatedDeltaNet, GDNLayerState, GDNState
        self.get_slot_tensor = _get_slot_tensor
        self.Attention, self.prepare_for_attn, self.CacheLayer_qsa_quant = Attention, prepare_for_attn, CacheLayer_qsa_quant
        self.BlockSparseMLP, self.MAX_BSZN, self.Linear, self.GatedResidual = BlockSparseMLP, MAX_BSZN, Linear, GatedResidual


def source_hashes(pkg_root, rel_paths):
    out = {}
    for rp in rel_paths:
        p = os.path.join(pkg_root, rp)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[rp] = hashlib.sha256(f.read()).hexdigest()
        else:
            out[rp] = None
    return out


def check_sources(log):
    """Compare the image's exllamav3 sources with src/ (data/src_sha256.json, written by the CPU test)."""
    ref_path = os.path.join(DATA, "src_sha256.json")
    if not os.path.exists(ref_path):
        return {"checked": False, "reason": "no data/src_sha256.json"}
    import exllamav3
    root = os.path.dirname(os.path.abspath(exllamav3.__file__))
    with open(ref_path) as f:
        ref = json.load(f)
    got = source_hashes(root, list(ref))
    diff = sorted(k for k in ref if ref[k] != got.get(k))
    if diff:
        log(f"WARN: installed exllamav3 differs from src/ in {diff} -- the probe's binding test ran against src/")
    return {"checked": True, "root": root, "mismatch": diff}


def new_result() -> dict:
    return {"families": {}, "errors": [], "path_problems": [], "equality_failures": [], "steps": {},
            "respawns": [], "step_s": {}}


def finalize(res: dict, requested) -> dict:
    """ok / missing / incomplete from what is on record; called on EVERY write, so a killed or crashed run still
    leaves a consistent verdict. A family counts only when its run completed (fam["complete"]); an error raised after
    a family completed stays in `errors` but does not invalidate that family's numbers."""
    fams = res.get("families", {})

    def done(f):
        return bool(fams.get(f, {}).get("complete"))
    res["requested_steps"] = list(requested)
    res["missing_primary_families"] = [f for st in PRIMARY_STEPS if st in requested
                                       for f in STEP_FAMILIES[st] if not done(f)]
    res["incomplete_steps"] = [st for st in requested if not all(done(f) for f in STEP_FAMILIES[st])]
    res["ok"] = not (res["missing_primary_families"] or res["path_problems"] or res["equality_failures"]
                     or res.get("fatal"))
    return res


def _family_size(f: dict) -> int:
    return len(f.get("rows", {})) + len(f.get("shapes", {}))


def merge_child(res: dict, child: dict, steps_run) -> None:
    """Fold a respawned process's results into this one. The child re-ran `steps_run` from scratch: its family entry
    wins when complete (or when ours is missing or smaller); its step record replaces ours, keeping ours as
    `earlier_attempt`."""
    tag = f"[respawn pid {child.get('pid')}] "
    res["respawns"].append({k: child.get(k) for k in ("pid", "rc", "out", "elapsed_s", "ok", "fatal")} |
                           {"steps": list(steps_run)})
    res["respawns"] += child.get("respawns", [])
    res["errors"] += [tag + e for e in child.get("errors", [])]
    res["path_problems"] += child.get("path_problems", [])
    res["equality_failures"] += child.get("equality_failures", [])
    for name, fam in child.get("families", {}).items():
        old = res["families"].get(name)
        if old is None or fam.get("complete") or (not old.get("complete") and _family_size(fam) >= _family_size(old)):
            res["families"][name] = fam
    for st in steps_run:
        rec = dict(child.get("steps", {}).get(st) or {"status": "not run",
                                                     "error": "the respawned process did not reach this step"})
        if st in res["steps"]:
            rec["earlier_attempt"] = res["steps"][st]
        res["steps"][st] = rec
    res["step_s"].update(child.get("step_s", {}))
    for k in ("unprofiled_template_instances", "moe_layers_tried"):
        if child.get(k):
            res.setdefault(k, {}).update(child[k])
    if child.get("fatal"):
        res["errors"].append(tag + "FATAL " + str(child["fatal"]))


def spawn_child(a, steps, retried, deadline, log) -> dict:
    """Run `steps` in a fresh process (fresh CUDA context) with the same arguments; returns its result dict."""
    out = f"{os.path.splitext(a.out)[0]}.from_{steps[0]}.json"
    left = deadline - time.time()
    if left < 30:
        return {"errors": [f"no time left to respawn {steps} ({left:.0f} s)"], "rc": None, "out": None}
    cmd = [sys.executable, os.path.abspath(__file__)] + list(a.argv) + [
        "--families", ",".join(steps), "--retried", ",".join(retried), "--deadline-at", repr(deadline), "--out", out]
    log(f"RESPAWN: {' '.join(cmd)}")
    t0 = time.time()
    try:
        rc = subprocess.run(cmd, timeout=left + 10).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    child = None
    if os.path.exists(out):
        try:
            with open(out) as f:
                child = json.load(f)
        except Exception as e:  # noqa: BLE001
            child = {"errors": [f"respawned process wrote unreadable results {out}: {e!r}"]}
    if child is None:
        child = {"errors": [f"respawned process for {steps} exited rc {rc} without results"]}
    child.update({"rc": rc, "out": out, "elapsed_s": round(time.time() - t0, 1)})
    return child


def run_probe(a, torch, xl, gpu, served, log=print, config_kwargs=None, checkpoint=None, spawn=None) -> dict:
    """Everything after the environment checks; the CPU dry run calls this with the stub extension and a fake Gpu.
    Steps run in STEP_ORDER, each in its own try/except. On a failure the remaining steps (plus the failed one, once,
    if it did not complete) continue in a fresh process via `spawn`, so a sticky CUDA error in one family cannot
    take the others down; without `spawn` (dry run, --no-respawn) they continue in this process."""
    t_start = time.time()
    deadline = a.deadline_at or (t_start + a.budget_s)
    steps = parse_steps(a.families)
    retried = [x for x in a.retried.split(",") if x]
    # No grad-mode wrapper: the r521 probe (the template that ran on the box) used none.
    config = xl.Config.from_directory(a.model, **(config_kwargs or {}))
    model = xl.Model.from_config(config)
    modules = {m.key: m for m in model}
    probe = Probe(a, torch, xl, gpu, served, stc=config.stc, log=log)
    res = probe.res
    res.update({"args": vars(a), "pid": os.getpid(), "deadline_at": deadline, "device": gpu.name,
                "torch": torch.__version__,
                "env": {k: v for k, v in os.environ.items()
                        if k.startswith(("EXL3_", "EXLLAMAV3_", "TRITON_", "PYTORCH_", "CUDA_"))},
                "sources": check_sources(log), "served_kernel_names": len(served),
                "model": {"hidden_size": config.hidden_size, "num_hidden_layers": config.num_hidden_layers,
                          "layer_types": list(getattr(config, "layer_types", []) or [])}})
    fns = {"GDN": lambda: probe.run_gdn(modules),
           "ATTN": lambda: probe.run_attn(
               modules, [f"model.language_model.layers.{i}.self_attn" for i in a.attn_layers.split(",")],
               TRUNK_SHAPES + DRAFT_SHAPES, "ATTN"),
           "MOE_K2": lambda: probe.run_moe(modules, 2),
           "MOE_K3": lambda: probe.run_moe(modules, 3),
           "MTP": lambda: probe.run_mtp(config),
           "HC": lambda: probe.run_hc(modules),
           "HEAD": lambda: probe.run_head(modules)}
    # NOTE no torch.cuda.empty_cache() between steps: in R527 it was the call that raised (see prewarm_shared)
    for i, name in enumerate(steps):
        rec = {"status": "running", "pid": os.getpid()}
        res["steps"][name] = rec
        t0 = time.time()
        failed = False
        if t0 >= deadline:
            rec["status"] = "skipped"
            rec["error"] = f"{name} skipped: probe budget ({a.budget_s:.0f} s) exhausted"
            res["errors"].append(rec["error"])
            log("ERROR: " + rec["error"])
        else:
            try:
                fns[name]()
                rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                failed = True
                msg = f"{name} failed: {e!r}"
                tb = traceback.format_exc()
                log("ERROR: " + msg)
                log(tb)
                rec.update({"status": "failed", "error": msg})
                res["errors"].append(msg + "\n" + tb)
        rec["elapsed_s"] = res["step_s"][name] = round(time.time() - t0, 1)
        log(f"[{name}] {rec['status']} {time.time() - t0:.1f} s (total {time.time() - t_start:.1f} s)")
        res["elapsed_s"] = round(time.time() - t_start, 1)
        if checkpoint is not None:          # partial results survive a timeout kill
            res["partial"] = True
            checkpoint(finalize(res, steps))
        if failed and spawn is not None and not a.no_respawn:
            complete = all(res["families"].get(f, {}).get("complete") for f in STEP_FAMILIES[name])
            retry = [] if complete or name in retried else [name]
            rest = retry + list(steps[i + 1:])
            if rest:
                for m in modules.values():          # best effort: give the fresh process the memory back
                    try:
                        m.unload()
                    except Exception:  # noqa: BLE001
                        pass
                gc.collect()
                merge_child(res, spawn(a, rest, retried + retry, deadline, log), rest)
                break
    res["elapsed_s"] = round(time.time() - t_start, 1)
    return finalize(res, steps)


def main(argv=None) -> int:
    a = parse_args(argv)
    steps = parse_steps(a.families)
    last = {}

    def write(r):
        last["res"] = r
        with open(a.out + ".tmp", "w") as f:
            json.dump(r, f, indent=1, default=str)
        os.replace(a.out + ".tmp", a.out)

    missing = {k: os.environ.get(k) for k, v in REQUIRED_ENV.items() if os.environ.get(k) != v}
    if missing:
        msg = (f"served env flags missing {missing} -- pass the daily EXTRA_ENV (kernel paths would differ from the "
               f"served ones)")
        print("FATAL: " + msg, file=sys.stderr)
        write(finalize(new_result() | {"fatal": msg, "errors": [msg], "pid": os.getpid()}, steps))
        return 2
    import torch
    if not torch.cuda.is_available():
        write(finalize(new_result() | {"fatal": "CUDA required", "errors": ["CUDA required"], "pid": os.getpid()},
                       steps))
        print("FATAL: CUDA required", file=sys.stderr)
        return 2
    try:
        xl = XL()
        gpu = Gpu(torch, torch.device(a.device))
        served = served_kernel_names()
        res = run_probe(a, torch, xl, gpu, served, log=lambda m: print(m, flush=True), checkpoint=write,
                        spawn=spawn_child)
    except Exception as e:  # noqa: BLE001  (anything outside the per-step guards: still write a verdict)
        tb = traceback.format_exc()
        print(f"FATAL: {e!r}\n{tb}", file=sys.stderr, flush=True)
        res = last.get("res") or (new_result() | {"pid": os.getpid()})
        res["fatal"] = repr(e)
        res["errors"].append(f"FATAL {e!r}\n{tb}")
        finalize(res, steps)
    res["partial"] = False
    write(res)
    print(f"wrote {a.out}: ok={res['ok']} errors={len(res['errors'])} path_problems={len(res['path_problems'])} "
          f"equality_failures={len(res['equality_failures'])} missing={res['missing_primary_families']} "
          f"incomplete_steps={res['incomplete_steps']} respawns={len(res['respawns'])}", flush=True)
    for e in res["errors"]:
        print("ERROR:", e.splitlines()[0], file=sys.stderr)
    for p in res["path_problems"]:
        print("PATH:", p, file=sys.stderr)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
