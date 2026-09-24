"""moefast r3: the served side-stream shared expert and router, rebuilt from the checkpoint for P0 and parity.

What BC_BlockSparseMLP::run_bszN does around the routed coop kernels (libtorch/blocksparse_mlp.cpp), with the same
bound classes and kernels:
  router      ext.routing_std(y, gate (H, E), ...) on the main stream = hgemm + routing_std_topk (the served
              routing_std for bsz > 1, modules/block_sparse_mlp_routing.py:101-111); its outputs are not used
              (the benches route with their own precomputed sets) -- it only occupies the main stream as served
  shared      ext.BC_GatedMLP built like GatedMLP.load (modules/mlp.py:640-690: fused gate/up MultiLinear pointer
              table -> exl3_mgemm, silu, down BC_LinearEXL3 -> exl3_gemm/gemv), replayed through its own graph by
              run_bszN on a side torch stream, between an input event and a done event (the served overlap block)
  join        ext.exl3_moe_coop_ev(..., wait_before_b = done.cuda_event): the launcher waits on it right before
              stage B, as run_bszN does with shared_done
Fork points: "late" = after the router (served, run_bszN), "early" = before the router (r3, start_shared).

Weights: model.language_model.layers.<L>.mlp.{shared_expert.{gate,up,down}_proj, gate, shared_expert_gate} for
every layer L found (so a rotation over the 48 copies reads the shared expert and router from DRAM, as serving
does). A missing key falls back to a synthetic tensor of the served shape and is reported (decode cost does not
depend on the trellis bits, d0_microbench docstring).
"""
import os
import random

MAXB, TOPK = 16, 10


def _flags(ckpt, prefix):
    m = ckpt.map
    return f"{prefix}.mcg" in m, f"{prefix}.mul1" in m


def _linear(torch, d0, ckpt, prefix, k, n, K_synth, dev, notes):
    try:
        tr, suh, svh = (ckpt.get(f"{prefix}.{s}") for s in ("trellis", "suh", "svh"))
        mcg, mul1 = _flags(ckpt, prefix)
        return tr.contiguous(), suh.contiguous(), svh.contiguous(), mcg, mul1, "real"
    except KeyError:
        notes.add(f"synthetic {prefix.split('.')[-1]}")
        tr, suh, svh = d0.synth_linear(torch, k, n, K_synth, dev)
        return tr, suh, svh, False, True, "synth"


class SharedSet:
    """Per-layer shared experts (BC_GatedMLP), shared-gate weights and router gates, plus the scratch they share
    (one set per device in serving too: g_tensor_cache)."""

    def __init__(self, torch, ext, d0, ckpt, dev, layers=range(48), K_synth=4, log=print):
        self.torch, self.ext, self.dev = torch, ext, dev
        t = torch
        H, E = d0.HIDDEN, d0.N_EXPERTS
        notes = set()
        self.entries = []
        self.keep = []
        for L in layers:
            pfx = f"model.language_model.layers.{L}.mlp"
            g = _linear(torch, d0, ckpt, f"{pfx}.shared_expert.gate_proj", H, d0.MOE_I, K_synth, dev, notes)
            u = _linear(torch, d0, ckpt, f"{pfx}.shared_expert.up_proj", H, d0.MOE_I, K_synth, dev, notes)
            I = g[0].shape[1] * 16
            dn = _linear(torch, d0, ckpt, f"{pfx}.shared_expert.down_proj", I, H, K_synth, dev, notes)
            K = g[0].shape[-1] // 16
            assert u[0].shape[-1] // 16 == K and g[3:5] == u[3:5] == dn[3:5], f"{pfx}: shared gate/up/down codebook or K differ"
            try:
                rg = ckpt.get(f"{pfx}.gate.weight").half()
                rg = (rg.T if rg.shape[0] == E else rg).contiguous()
                assert tuple(rg.shape) == (H, E), tuple(rg.shape)
            except KeyError:
                notes.add("synthetic router gate")
                rg = (t.randn((H, E), device=dev) * 0.02).half()
            try:
                sg = ckpt.get(f"{pfx}.shared_expert_gate.weight").half().flatten().contiguous()
                assert sg.numel() == H
            except KeyError:
                notes.add("synthetic shared_expert_gate")
                sg = (t.randn((H,), device=dev) * 0.02).half()
            self.entries.append(dict(layer=L, K=K, I=I, mcg=g[3], mul1=g[4], gu=(g, u), down=dn, router=rg, sh_gate=sg))
        I = self.entries[0]["I"]
        # scratch shared by every copy, shapes as GatedMLP.bsz1_pa_args (guh, gu, a, down_xh)
        self.guh = t.empty((2, MAXB, H), dtype=t.half, device=dev)
        self.gu = t.empty((2, MAXB, I), dtype=t.half, device=dev)
        self.a = t.empty((1, MAXB, I), dtype=t.half, device=dev)
        self.down_xh = t.empty((1, MAXB, I), dtype=t.half, device=dev)
        self.lin_xh = t.empty((1, I), dtype=t.half, device=dev)
        for e in self.entries:
            (g, u), dn = e["gu"], e["down"]
            ptr = lambda j: t.tensor([g[j].data_ptr(), u[j].data_ptr()], dtype=t.long, device=dev)
            e["ptrs"] = (ptr(0), ptr(1), ptr(2))
            down_bc = ext.BC_LinearEXL3(dn[0], dn[1], dn[2], dn[0].shape[-1] // 16, None, dn[3], dn[4], self.lin_xh)
            e["bc"] = ext.BC_GatedMLP(self.guh, self.gu, self.a, self.down_xh, e["ptrs"][0], e["ptrs"][1], e["ptrs"][2],
                                      e["K"], e["mcg"], e["mul1"], True, False, False, None, None, down_bc, 0.0)
            e["down_bc"] = down_bc
        # the shared expert output, (1, MAX_BSZN, H) fp32 as BlockSparseMLP's sh_exp_t
        self.sh_out = t.zeros((1, MAXB, H), dtype=t.float, device=dev)
        self.notes = sorted(notes)
        log(f"[side] {len(self.entries)} shared experts K={sorted({e['K'] for e in self.entries})} I={I} "
            f"codebook mcg={self.entries[0]['mcg']} mul1={self.entries[0]['mul1']}; " +
            ("; ".join(self.notes) if self.notes else "all tensors real"))

    def warm(self, rows_list):
        """BC_GatedMLP.run_bszN is eager on its first call per row count and captures its graph on the second:
        do both for every copy outside any timing."""
        t = self.torch
        for rows in rows_list:
            x3 = t.zeros((1, rows, self.guh.shape[-1]), dtype=t.half, device=self.dev)
            for e in self.entries:
                for _ in range(3):
                    e["bc"].run_bszN(x3, self.sh_out[:, :rows])
            t.cuda.synchronize()

    def router_scratch(self, rows):
        t = self.torch
        E = self.entries[0]["router"].shape[1]
        return (t.empty((rows, E), dtype=t.half, device=self.dev), t.empty((rows, TOPK), dtype=t.long, device=self.dev),
                t.empty((rows, TOPK), dtype=t.half, device=self.dev))


class Overlap:
    """The served fork/join: input event on the current stream, side stream waits, shared graph on the side stream,
    done event. prio: None = the served nonblocking side stream (priority 0); 'high' = the greatest priority."""

    def __init__(self, torch, dev, prio=None):
        self.torch = torch
        p = 0
        if prio == "high":
            p = min(torch.cuda.Stream.priority_range())      # CUDA: lower number = higher priority
        self.side = torch.cuda.Stream(device=dev, priority=p)
        self.priority = self.side.priority
        self.ev_in = torch.cuda.Event()
        self.ev_done = torch.cuda.Event()
        # initialise both events (cuda_event is 0 until the first record)
        self.ev_in.record(); self.ev_done.record(); torch.cuda.synchronize()

    def fork(self, bc, x3, d3):
        t = self.torch
        self.ev_in.record(t.cuda.current_stream())
        self.side.wait_event(self.ev_in)
        with t.cuda.stream(self.side):
            bc.run_bszN(x3, d3)
        self.ev_done.record(self.side)

    @property
    def done_handle(self):
        return self.ev_done.cuda_event


def coop_args(tab, K, Kd, scr, H):
    return ((tab["gate_trellis"], tab["gate_suh"], tab["gate_svh"], tab["up_trellis"], tab["up_suh"], tab["up_svh"],
             tab["down_trellis"], tab["down_suh"], tab["down_svh"], None, None, None, K, K, Kd, False, True, 0, 0.0, True,
             scr["had_g"], scr["had_u"], scr["gu_g"], scr["gu_u"], scr["act"], scr["d_out"], scr["ctr"], scr["out"]))


def make_call(torch, ext, side, ov, tab, K, Kd, scr, H, x, sel, rw, j, fork, router=True):
    """One served MoE layer call: fork in {'serial', 'late', 'early', 'none'}.
      serial: shared expert on the main stream first, coop without a wait (EXL3_SHARED_EXPERT_OVERLAP=0)
      late:   router, fork, coop with the join (served overlap)
      early:  fork, router, coop with the join (r3 EXL3_SHARED_EXPERT_EARLY)
      none:   no shared expert at all (r2's P0: sh_out = None)
    j selects the shared-expert / router copy."""
    e = side.entries[j % len(side.entries)]
    rows = x.shape[0]
    x3 = x.unsqueeze(0)
    d3 = side.sh_out[:, :rows]
    rs = side.router_scratch(rows) if router else None
    args = coop_args(tab, K, Kd, scr, H)
    gate_w = e["sh_gate"]

    def f():
        if fork == "serial":
            if router:
                ext.routing_std(x, e["router"], rs[0], rs[1], rs[2], None, None, None)
            e["bc"].run_bszN(x3, d3)
            ext.exl3_moe_coop_ev(x, sel, rw, -1, -1, H, *args, d3, gate_w, 0)
            return
        if fork == "none":
            if router:
                ext.routing_std(x, e["router"], rs[0], rs[1], rs[2], None, None, None)
            ext.exl3_moe_coop_ev(x, sel, rw, -1, -1, H, *args, None, None, 0)
            return
        if fork == "early":
            ov.fork(e["bc"], x3, d3)
        if router:
            ext.routing_std(x, e["router"], rs[0], rs[1], rs[2], None, None, None)
        if fork == "late":
            ov.fork(e["bc"], x3, d3)
        ext.exl3_moe_coop_ev(x, sel, rw, -1, -1, H, *args, d3, gate_w, ov.done_handle)
    return f


def kernel_role(name):
    n = name
    if "moe_coop" in n:
        if "head_kernel" in n:
            return "head"
        if "rot_kernel" in n:
            return "rot"
        if "_a_kernel" in n:
            return "A"
        if "_b_kernel" in n:
            return "B"
        return "moe?"
    if "mgemm" in n or "act_mul" in n or "exl3_gemv" in n or "exl3_gemm_kernel" in n or "gemv_int8" in n or "silu" in n:
        return "shared"
    return "router"


def timeline(kernels):
    """kernels: [(start_us, end_us, role)] of consecutive calls. Per call (one B each): A->B gap, shared start - A start,
    shared end - A end, shared span, head->B span (first main MoE kernel to B end), call span (first kernel of the
    call to B end). Medians over calls; None when a call has no shared kernels."""
    import statistics as st
    ks = sorted(kernels)
    bs = [k for k in ks if k[2] == "B"]
    per = {k: [] for k in ("gap", "sh_start", "sh_end", "sh_span", "moe_span", "call_span")}
    prev_end = -1e18
    for b in bs:
        a = max((k for k in ks if k[2] == "A" and k[0] < b[0]), default=None, key=lambda k: k[0])
        if a is None or a[0] < prev_end:
            prev_end = b[1]
            continue
        win = [k for k in ks if prev_end <= k[0] <= b[0]]
        sh = [k for k in win if k[2] == "shared"]
        head = [k for k in win if k[2] in ("head", "rot", "A")]
        per["gap"].append(b[0] - a[1])
        per["moe_span"].append(b[1] - min(k[0] for k in head))
        per["call_span"].append(b[1] - min(k[0] for k in win))
        if sh:
            s0, s1 = min(k[0] for k in sh), max(k[1] for k in sh)
            per["sh_start"].append(s0 - a[0])
            per["sh_end"].append(s1 - a[1])
            per["sh_span"].append(s1 - s0)
        prev_end = b[1]
    return {k: (st.median(v) if v else None) for k, v in per.items()} | {"calls": len(per["gap"])}
