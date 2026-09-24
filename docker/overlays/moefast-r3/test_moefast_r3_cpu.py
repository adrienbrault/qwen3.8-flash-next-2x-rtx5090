#!/usr/bin/env python3
"""moefast r3 CPU tests (no GPU).

Part 1 (plain python, runs anywhere): the P0 helpers the gate depends on.
  - moefast_r3_side.timeline(): per-call A->B gap / shared start / MoE span on synthetic kernel lists shaped like
    R703b's M2 (shared overlaps A) and R2 (shared after A, B waits) layers, and an early-fork layer.
  - bench_moefast_p0.pick(): the pre-registered R3 rule (lever E vs E+P, mode per row class, MAP string).
Part 2 (needs torch + the installed exllamav3, e.g. inside the image during docker build; skipped otherwise):
  BlockSparseMLP.forward with a recording fake bound class: with EXL3_SHARED_EXPERT_EARLY on the bound class,
  start_shared(y) is called exactly once, BEFORE the router, with the very tensor later passed to run_bszN; it is
  not called when the guard fails (flag off, routed pre-norm, latent projection, CPU offload / split, broadcast,
  not quantized, no_reconstruct, bsz > MAX_BSZN, no embedded shared expert).
Run: python3 test_moefast_r3_cpu.py
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
fails = 0


def check(cond, msg):
    global fails
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails += 1


# ------------------------------------------------------------------------------------------- part 1
import moefast_r3_side as side  # noqa: E402  (torch-free at import)


def layer(t0, rot, a, gap, b, sh0, sh_span, early_router=0.0):
    ks = []
    if early_router:
        ks.append((t0 - early_router, t0 - 1.0, "router"))
    if rot:
        ks.append((t0, t0 + rot, "rot"))
    a0 = t0 + rot
    ks.append((a0, a0 + a, "A"))
    b0 = a0 + a + gap
    ks.append((b0, b0 + b, "B"))
    s = a0 + sh0
    ks += [(s, s + sh_span * 0.5, "shared"), (s + sh_span * 0.55, s + sh_span * 0.6, "shared"),
           (s + sh_span * 0.65, s + sh_span, "shared")]
    return ks


m2 = layer(0, 3.58, 38.24, 1.90, 26.03, -3.58, 31.15) + layer(200, 3.58, 38.24, 1.90, 26.03, -3.58, 31.15)
r2 = layer(0, 0, 34.46, 14.14, 24.99, 22.94, 22.21) + layer(200, 0, 34.46, 14.14, 24.99, 22.94, 22.21)
tm2, tr2 = side.timeline(m2), side.timeline(r2)
check(tm2["calls"] == 2 and abs(tm2["gap"] - 1.90) < 1e-6 and abs(tm2["sh_start"] + 3.58) < 1e-6,
      f"timeline M2-like layer: gap {tm2['gap']:.2f}, shared start - A start {tm2['sh_start']:.2f}")
check(abs(tr2["gap"] - 14.14) < 1e-6 and abs(tr2["sh_start"] - 22.94) < 1e-6 and abs(tr2["moe_span"] - 73.59) < 1e-6,
      f"timeline R2-like layer: gap {tr2['gap']:.2f}, shared start {tr2['sh_start']:.2f}, moe span {tr2['moe_span']:.2f}")
e = layer(0, 0, 34.46, 1.9, 24.99, -8.0, 22.0, early_router=10.0)
te = side.timeline(e)
check(te["sh_start"] < 0 and te["call_span"] > te["moe_span"], "timeline early fork: shared starts before A; call span includes the router")
check(side.kernel_role("void exl3_moe_coop_r2_ns::exl3_moe_coop_r2_a_kernel<2, 2, true>(MoeCoopParams, R2Args)") == "A"
      and side.kernel_role("exl3_moe_coop_r2_ns::exl3_moe_coop_r2_head_kernel(unsigned int)") == "head"
      and side.kernel_role("exl3_moe_coop_ns::exl3_moe_coop_rot_kernel(MoeCoopParams)") == "rot"
      and side.kernel_role("void exl3_mgemm_kernel<4, false, 2, 1>(...)") == "shared"
      and side.kernel_role("void exl3_gemv_kernel<4, true, 2, 1, 0, false>(...)") == "shared"
      and side.kernel_role("routing_std_topk_kernel(...)") == "router", "kernel_role classes")

try:
    import bench_moefast_p0 as p0  # noqa: E402  (imports d0_microbench: torch-free at import)
except ImportError as x:
    p0 = None
    print(f"skip pick() tests: {x}")
if p0:
    def summ(vals):
        """vals: {(rows, D): {arm: us per call}} -> summary with identical K2/K3 cells"""
        out = {}
        for (rows, D), us in vals.items():
            for K in (2, 3):
                out[f"K{K}/r{rows}/D{D}"] = dict(rows=rows, D=D, us=dict(us))
        return out
    base = {"m2E": 60.0, "m3E": 55.0, "m2EP": 60.0, "m3EP": 54.0}
    s = summ({(4, 28): base, (16, 77): {"m2E": 150.0, "m3E": 149.0, "m2EP": 150.0, "m3EP": 148.0}})
    env, why = p0.pick(s)
    check(env == "EXL3_MOE_COOP_V3=3 EXL3_SHARED_EXPERT_EARLY=1 EXL3_SHARED_EXPERT_PRIO=1", f"pick: EP wins both, mode 3 everywhere -> {env}")
    s = summ({(4, 28): base, (16, 77): {"m2E": 150.0, "m3E": 149.0, "m2EP": 150.0, "m3EP": 149.5}})
    env, _ = p0.pick(s)
    check(env == "EXL3_MOE_COOP_V3=3 EXL3_SHARED_EXPERT_EARLY=1", f"pick: EP loses at 16 rows -> lever E -> {env}")
    s = summ({(4, 28): base, (16, 77): {"m2E": 148.0, "m3E": 149.0, "m2EP": 147.0, "m3EP": 148.5}})
    env, _ = p0.pick(s)
    check(env == "EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=13-16:2 EXL3_SHARED_EXPERT_EARLY=1 EXL3_SHARED_EXPERT_PRIO=1",
          f"pick: mode 2 faster at 16 rows -> MAP 13-16:2 -> {env}")
    s = summ({(4, 28): {"m2E": 50.0, "m3E": 55.0, "m2EP": 50.0, "m3EP": 56.0},
              (16, 77): {"m2E": 148.0, "m3E": 149.0, "m2EP": 147.0, "m3EP": 150.0}})
    env, _ = p0.pick(s)
    check(env == "EXL3_MOE_COOP_V3=2 EXL3_SHARED_EXPERT_EARLY=1", f"pick: mode 2 everywhere -> V3=2 + E -> {env}")
    s = summ({(4, 28): base, (8, 40): {"m2E": 70.0, "m3E": 72.0, "m2EP": 70.0, "m3EP": 71.0},
              (12, 60): {"m2E": 100.0, "m3E": 101.0, "m2EP": 99.0, "m3EP": 100.0},
              (16, 77): {"m2E": 150.0, "m3E": 149.0, "m2EP": 150.0, "m3EP": 148.0}})
    env, _ = p0.pick(s)
    check(env == "EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=5-12:2 EXL3_SHARED_EXPERT_EARLY=1 EXL3_SHARED_EXPERT_PRIO=1",
          f"pick: 5-12 rows from r8/r12 cells -> {env}")
    env, why = p0.pick(summ({(4, 28): base}))
    check(env is None, f"pick: missing 16-row cell -> no pick ({why})")

# ------------------------------------------------------------------------------------------- part 2
try:
    import torch  # noqa: F401
    from exllamav3.modules import block_sparse_mlp as bsm
except Exception as x:          # plain host: no torch / no installed exllamav3
    import os
    required = os.environ.get("MOEFAST_R3_REQUIRE_PART2") == "1"   # set in the image build: a skip there is a failure
    check(not required, f"part 2 (forward guard) importable, or not required on this host: {type(x).__name__}: {x}")
    if not required:
        print("skip part 2 (forward guard)")
    bsm = None

if bsm is not None:
    import torch

    class FakeBC:
        def __init__(self, early):
            self.shared_early = early
            self.calls = []

        def start_shared(self, y):
            self.calls.append(("start", y))

        def run_bszN(self, y, sel, w):
            self.calls.append(("run", y))

    def make(early=True, **over):
        m = bsm.BlockSparseMLP.__new__(bsm.BlockSparseMLP)
        H, E, K = 64, 16, 4
        bc = FakeBC(early)
        calls = bc.calls
        attrs = dict(
            alt_residual_channel=False, hidden_size=H, expert_size=H, bc=bc, router_pre_norm=None,
            routing_gate=object(), routing_cfg=None, num_experts_per_tok=K, num_experts=E, routed_pre_norm=None,
            latent_in=None, routing_device=None, cpu_split_first=None, cpu_offload=False, intermediate_size=32,
            num_local_experts=E, f_threshold=10 ** 9, is_quantized=True, support_quant_paths=True,
            config=types.SimpleNamespace(infer_params=types.SimpleNamespace(no_reconstruct=False)),
            experts_cfg=types.SimpleNamespace(out_bszn=torch.zeros((16, H))), bc_sh_exp=True, latent_out=None,
            tp_reduce=False, routed_post_norm=None, shared_experts=None, shared_experts_post_norm=None,
            shared_gate=None, device="cpu", bcast_sel_bsz1=None, bcast_weights_bsz1=None,
        )
        attrs.update(over)
        for k, v in attrs.items():
            object.__setattr__(m, k, v)

        def routing_fn(bsz, cfg, z, params):
            calls.append(("route", z))
            return torch.zeros((bsz, K), dtype=torch.long), torch.zeros((bsz, K), dtype=torch.half)
        object.__setattr__(m, "routing_fn", routing_fn)
        return m, calls

    def run(m, rows=4):
        x = torch.randn((1, rows, 64))
        try:
            m.forward(x, {})
        except Exception as ex:          # the non-fused branches need real weights; only the calls before matter
            return x, repr(ex)
        return x, None

    m, calls = make()
    x, err = run(m)
    kinds = [c[0] for c in calls]
    check(err is None and kinds == ["start", "route", "run"], f"early on: call order {kinds} (err {err})")
    check(calls[0][1] is calls[2][1], "start_shared and run_bszN receive the same tensor object")
    for name, over, rows in (("flag off", dict(), 4), ("routed pre-norm", dict(routed_pre_norm=object()), 4),
                             ("latent in", dict(latent_in=object()), 4), ("cpu offload", dict(cpu_offload=True), 4),
                             ("cpu split", dict(cpu_split_first=3), 4), ("broadcast", dict(routing_device=0), 4),
                             ("not quantized", dict(is_quantized=False), 4),
                             ("no_reconstruct", dict(config=types.SimpleNamespace(infer_params=types.SimpleNamespace(no_reconstruct=True))), 4),
                             ("no embedded shared expert", dict(bc_sh_exp=False), 4),
                             ("empty slice", dict(intermediate_size=0), 4), ("bsz 17", dict(), 17)):
        m, calls = make(early=(name != "flag off"), **over)
        run(m, rows)
        check("start" not in [c[0] for c in calls], f"guard: no early fork when {name}")

print("CPU TESTS " + ("PASS" if fails == 0 else f"FAIL ({fails})"))
sys.exit(1 if fails else 0)
