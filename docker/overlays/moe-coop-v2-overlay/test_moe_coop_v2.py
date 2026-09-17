#!/usr/bin/env python3
"""Operator GPU test. Real one-layer Linear.load by default; no full model load.

  python3 /work/test_moe_coop_v2.py --json /results/moe-v2.json
  python3 /work/test_moe_coop_v2.py --synthetic --bits 2 --cb 2

BF16 random rows are converted to FP16, the existing public coop API's dtype.
Events time a captured full call (200 replays) and each constituent launch in an
instrumented graph (200 samples). Stage events add overhead; use full-call timing
for speed claims. --cold flushes a 256 MiB buffer before each sampled full call.
"""
import argparse
import ctypes as C
import ctypes.util
import json
import os
from pathlib import Path
import statistics
import sys


def select_mode(mode):
    os.environ["EXL3_MOE_COOP_V2"] = str(mode)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/models/qwen3.8-flash-next-exl3-3.05bpw")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--prefix", default="model.language_model")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--bits", type=int, choices=range(1, 9), default=3)
    p.add_argument("--down-bits", type=int, choices=range(1, 9))
    p.add_argument("--cb", type=int, choices=range(3), default=2)
    p.add_argument("--mode", type=int, choices=(1, 2), default=1)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--json", type=Path)
    p.add_argument("--routing", type=Path, help="Optional torch.save dict R -> (selected_experts, weights) from served routing")
    p.add_argument("--skip-stage-events", action="store_true", help="Full-call events still run; per-stage profiler timings are not substituted")
    a = p.parse_args()
    if a.iterations < 200 or a.warmup < 1:
        p.error("at least 200 iterations and positive warmup required")
    return a


class GraphKernels:
    """Read kernel nodes through the public CUDA Runtime API. No private Torch ABI.

    Graph owns copied kernel arguments. Keep source graph alive while capturing
    the instrumented graph. Topological order is mandatory (getNodes is unordered).
    """
    class Dim3(C.Structure):
        _fields_ = [(x, C.c_uint) for x in ("x", "y", "z")]

    def __init__(self, graph, expected):
        class Params(C.Structure):
            _fields_ = [("func", C.c_void_p), ("gridDim", self.Dim3),
                        ("blockDim", self.Dim3), ("sharedMemBytes", C.c_uint),
                        ("kernelParams", C.POINTER(C.c_void_p)), ("extra", C.POINTER(C.c_void_p))]
        self.graph = graph
        self.rt = C.CDLL(ctypes.util.find_library("cudart") or "libcudart.so.12")
        def bind(name, args):
            f = getattr(self.rt, name); f.restype = C.c_int; f.argtypes = args
            return f
        ptrs = C.POINTER(C.c_void_p)
        sizes = C.POINTER(C.c_size_t)
        self.nodes = bind("cudaGraphGetNodes", [C.c_void_p, ptrs, sizes])
        deps = bind("cudaGraphNodeGetDependencies", [C.c_void_p, ptrs, sizes])
        typ = bind("cudaGraphNodeGetType", [C.c_void_p, C.POINTER(C.c_int)])
        get = bind("cudaGraphKernelNodeGetParams", [C.c_void_p, C.POINTER(Params)])
        self.launch = bind("cudaLaunchKernel", [C.c_void_p, self.Dim3, self.Dim3, ptrs, C.c_size_t, C.c_void_p])
        raw = C.c_void_p(graph.raw_cuda_graph())
        count = C.c_size_t()
        self.check(self.nodes(raw, None, C.byref(count)))
        array = (C.c_void_p * count.value)()
        self.check(self.nodes(raw, array, C.byref(count)))
        pending = {}
        for node in array:
            t = C.c_int(); self.check(typ(node, C.byref(t)))
            if t.value != 0:
                raise RuntimeError(f"Expected kernel-only graph, node type={t.value}")
            n = C.c_size_t(); self.check(deps(node, None, C.byref(n)))
            ds = (C.c_void_p * n.value)(); self.check(deps(node, ds, C.byref(n)))
            pending[node] = set(ds)
        self.params = []
        done = set()
        while pending:
            ready = [n for n, d in pending.items() if d <= done]
            if len(ready) != 1:
                raise RuntimeError("Expected a single stream, totally ordered coop graph")
            node = ready[0]; par = Params(); self.check(get(node, C.byref(par)))
            self.params.append(par); done.add(node); del pending[node]
        if len(self.params) != expected:
            raise RuntimeError(f"Expected {expected} coop launches, got {len(self.params)}")

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA Runtime error {code}")

    def enqueue(self, i, stream):
        p = self.params[i]
        self.check(self.launch(p.func, p.gridDim, p.blockDim, p.kernelParams, p.sharedMemBytes, stream))


def main():
    a = arguments()
    # Import only after argument parsing; --help works on CPU without Torch.
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.model.config import Config
    from exllamav3.modules.linear import Linear
    if torch.cuda.get_device_capability(a.device) != (12, 0):
        raise RuntimeError("This tuning test requires sm_120; refusing a silent V1 fallback")
    for env in ("EXL3_MOE_COOP_DBG",):
        if env in os.environ:
            raise RuntimeError(f"Unset {env} for a correctness/performance test")
    torch.cuda.set_device(a.device)
    torch.manual_seed(a.seed)
    cfg_json = json.loads((Path(a.model) / "config.json").read_text())
    cfg_text = cfg_json.get("text_config", cfg_json)
    H, I, E, K = (int(cfg_text[k]) for k in ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok"))
    Hi, Ip, Ho = (H + 127) // 128 * 128, (I + 127) // 128 * 128, (H + 127) // 128 * 128
    if E < 16 * K or K > 32:
        raise RuntimeError("Distinct-slot stress pattern needs E >= 16*topk and topk <= 32")
    dev = torch.device(a.device)
    keep = []
    def table(ts):
        keep.extend(ts)
        return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
    def synthetic(k, n, bits):
        ts = [torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), dtype=torch.int16, device=dev) for _ in range(E)]
        def scales(width, factor):
            return [((torch.rand(width, device=dev) + 0.5) * factor *
                     (torch.randint(0, 2, (width,), device=dev) * 2 - 1)).half() for _ in range(E)]
        return (table(ts), table(scales(k, 1)), table(scales(n, k ** -0.5))), bits, a.cb
    config = None
    if a.synthetic:
        projs = [synthetic(Hi, Ip, a.bits), synthetic(Hi, Ip, a.bits), synthetic(Ip, Ho, a.down_bits or a.bits)]
        router = torch.randn((H, E), device=dev, dtype=torch.float16) * H ** -0.5
    else:
        config = Config.from_directory(a.model)
        base = f"{a.prefix}.layers.{a.layer}.mlp"
        projs = []
        for name, k, n, dtype in (("gate", Hi, Ip, torch.half), ("up", Hi, Ip, torch.half), ("down", Ip, Ho, torch.float)):
            ls = []
            for e in range(E):
                m = Linear(config=config, key=f"{base}.experts.{e}.{name}_proj", in_features=k, out_features=n, out_dtype=dtype)
                m.load(dev)
                if m.quant_type != "exl3" or m.inner.bias is not None:
                    raise RuntimeError("Real-loader test expects unbiased EXL3 experts")
                ls.append(m)
            keep.extend(ls)
            bits = {m.inner.K for m in ls}
            cbs = {(bool(m.inner.mcg), bool(m.inner.mul1)) for m in ls}
            if len(bits) != 1 or len(cbs) != 1:
                raise RuntimeError("Nonuniform expert format")
            mcg, mul1 = next(iter(cbs))
            projs.append((tuple(table([getattr(m.inner, key) for m in ls]) for key in ("trellis", "suh", "svh")), bits.pop(), 1 if mcg else 2 if mul1 else 0))
        route = Linear(config=config, key=f"{base}.gate", in_features=H, out_features=E, out_dtype=torch.half, pad_to=1)
        route.load(dev); keep.append(route); router = route.inner.weight
    g, u, d = projs
    if g[1] != u[1] or not (g[2] == u[2] == d[2]):
        raise RuntimeError("Coop requires uniform codebook and gate/up width")
    recorded = torch.load(a.routing, map_location="cpu", weights_only=True) if a.routing else {}
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev) if a.cold else None
    results = []
    failures = []
    print(json.dumps({"source": "synthetic checkpoint shapes" if a.synthetic else "one real layer via Config.from_directory + Linear.load", "H": H, "I": I, "E": E, "topk": K, "bits": [g[1], u[1], d[1]], "cb": g[2], "torch": torch.__version__, "gpu": str(torch.cuda.get_device_properties(dev)), "cache": "cold 256MiB flush" if a.cold else "warm same-layer L2", "mode": a.mode}), flush=True)

    def compare(ref, test):
        finite = bool(torch.isfinite(ref).all() and torch.isfinite(test).all())
        diff = test.double() - ref.double()
        return {"bitwise_equal": bool(torch.equal(ref.view(torch.int32), test.view(torch.int32))),
                "finite": finite, "max_abs_delta": diff.abs().max().item(),
                "relative_l2": (diff.norm() / ref.double().norm().clamp_min(1e-300)).item()}

    def capture(run):
        # keep_graph preserves kernel-node metadata for public CUDA introspection.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            run()
        graph.instantiate()
        return graph

    def timing(run, R, mode):
        select_mode(mode)
        for _ in range(a.warmup): run()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            run(); torch.cuda.synchronize()
        names = sorted({e.name for e in prof.events() if "moe_coop" in e.name and "kernel" in e.name})
        marker = "exl3_moe_coop_v2_ns::"
        if (mode != 0) != any(marker in n for n in names):
            raise RuntimeError(f"Wrong dispatch for mode {mode}: {names}")
        graph = capture(run)
        for _ in range(a.warmup): graph.replay()
        torch.cuda.synchronize()
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        if flush is None:
            start.record()
            for _ in range(a.iterations): graph.replay()
            end.record(); end.synchronize()
            us = start.elapsed_time(end) * 1000 / a.iterations
        else:
            samples = []
            for _ in range(a.iterations):
                flush.zero_(); start.record(); graph.replay(); end.record(); end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            us = statistics.mean(samples)
        result = {"call_us_cuda_events": us, "kernel_names": names, "iterations": a.iterations}
        if not a.skip_stage_events:
            labels = (["rot"] if R > 1 else []) + ["a", "b"]
            nodes = GraphKernels(graph, len(labels))
            marks = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(len(labels) + 1)]
            stage_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(stage_graph):
                marks[0].record()
                for i in range(len(labels)):
                    nodes.enqueue(i, torch.cuda.current_stream().cuda_stream)
                    marks[i + 1].record()
            stage_samples = [[] for _ in labels]
            for _ in range(a.warmup): stage_graph.replay()
            torch.cuda.synchronize()
            for _ in range(a.iterations):
                if flush is not None: flush.zero_()
                stage_graph.replay(); marks[-1].synchronize()
                for i in range(len(labels)):
                    stage_samples[i].append(marks[i].elapsed_time(marks[i + 1]) * 1000)
            result["launch_us_cuda_events_with_instrumentation"] = dict(zip(labels, map(statistics.mean, stage_samples)))
            result["launch_geometry"] = [{"stage": label, "grid": [p.gridDim.x, p.gridDim.y, p.gridDim.z], "block": [p.blockDim.x, p.blockDim.y, p.blockDim.z], "dynamic_smem": p.sharedMemBytes} for label, p in zip(labels, nodes.params)]
        return result

    for R in (1, 2, 4, 8, 16):
        S = 16 * K
        def empty(shape, dtype): return torch.empty(shape, dtype=dtype, device=dev)
        had_g, had_u = empty((S, Hi), torch.half), empty((S, Hi), torch.half)
        gu_g, gu_u = empty((S, Ip), torch.half), empty((S, Ip), torch.half)
        act, down, out = empty((S, Ip), torch.half), empty((S, Ho), torch.float), empty((16, H), torch.float)
        ctr = torch.zeros(S * (Ip // 128) + 16 * (Ho // 128) + 2 * S + 3, dtype=torch.int32, device=dev)
        common = torch.randn((1, H), device=dev)
        xbf = ((0.7 * common + 0.7 * torch.randn((R, H), device=dev)) * 0.2).bfloat16()
        x = xbf.half()
        patterns = ["same_one", "same_top10", "distinct", "router_overlap", "masked_empty", "shared_merge"]
        if R in recorded or str(R) in recorded: patterns.append("recorded")
        for pattern in patterns:
            topk = 1 if pattern == "same_one" else K
            if pattern == "same_one":
                sel = torch.zeros((R, 1), dtype=torch.int64, device=dev)
            elif pattern == "same_top10":
                sel = torch.arange(K, device=dev).expand(R, K).contiguous()
            elif pattern in ("distinct", "masked_empty", "shared_merge"):
                sel = torch.arange(R * K, device=dev).view(R, K)
            else:
                sel = torch.topk(x @ router, K, dim=-1).indices.contiguous()
            rw = torch.softmax(torch.randn((R, topk), device=dev), -1).half()
            if pattern == "router_overlap":
                logits = x @ router
                rw = torch.softmax(logits.gather(1, sel).float(), -1).half()
            if pattern == "recorded":
                rs, rr = recorded.get(R, recorded.get(str(R)))
                sel, rw = rs.to(device=dev, dtype=torch.int64).contiguous(), rr.to(device=dev, dtype=torch.float16).contiguous()
                assert sel.shape == rw.shape == (R, K)
                assert int(sel.min()) >= 0 and int(sel.max()) < E
            minimum, maximum = -1, -1
            if pattern == "masked_empty":
                rw[0].zero_()
                if R > 1: rw[-1, ::2] = 0
                minimum, maximum = 0, max(1, E // 8)
            shared = torch.randn((R, H), device=dev) if pattern == "shared_merge" else None
            shared_w = (torch.randn(H, device=dev) * 0.02).half() if shared is not None else None
            def run():
                ext.exl3_moe_coop(x, sel, rw, minimum, maximum, Hi,
                    *g[0], *u[0], *d[0], None, None, None,
                    g[1], u[1], d[1], g[2] == 1, g[2] == 2, 0, 0.0, True,
                    had_g, had_u, gu_g, gu_u, act, down, ctr, out[:R], shared, shared_w)
            select_mode(0); run(); ref = out[:R].clone()
            select_mode(a.mode); run(); got = out[:R].clone()
            metrics = compare(ref, got)
            # Repeated and alternating calls catch counter-reset/stream-order errors.
            for mode in (a.mode, 0, a.mode, 0, a.mode):
                select_mode(mode); run()
                check = compare(ref if mode == 0 else got, out[:R])
                if not check["finite"] or not check["bitwise_equal"]:
                    failures.append((R, pattern, "not repeatable", check))
            if not metrics["finite"] or (a.mode == 1 and not metrics["bitwise_equal"]):
                failures.append((R, pattern, "V1/V2 mismatch", metrics))
            count = torch.bincount(sel.flatten(), minlength=E).cpu()
            unique = int((count > 0).sum()); runs = int(((count + 7) // 8).sum())
            record = {"R": R, "pattern": pattern, "unique_experts_unmasked": unique, "runs_unmasked": runs, **metrics}
            record["v1"] = timing(run, R, 0)
            record["v2"] = timing(run, R, a.mode)
            record["speedup"] = record["v1"]["call_us_cuda_events"] / record["v2"]["call_us_cuda_events"]
            # Timing must leave valid output too, including extracted launch argument lifetimes.
            post = compare(got, out[:R])
            if not post["finite"] or not post["bitwise_equal"]:
                failures.append((R, pattern, "after timing", post))
            results.append(record); print(json.dumps(record), flush=True)
    payload = {"results": results, "failures": failures, "mode": a.mode, "source": "synthetic" if a.synthetic else "real", "seed": a.seed}
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True); a.json.write_text(json.dumps(payload, indent=2) + "\n")
    print("FAIL" if failures else "PASS", "bit-exact" if a.mode == 1 else "experimental delta report", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
