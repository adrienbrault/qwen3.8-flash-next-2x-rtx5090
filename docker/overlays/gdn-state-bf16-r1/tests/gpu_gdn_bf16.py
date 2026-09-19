#!/usr/bin/env python3
"""GPU harness for gdn-state-bf16 r1 (EXL3_GDN_STATE_BF16).

Three parts, all at the Qwen3.8-Flash-Next GDN geometry (16 k-heads, 48 v-heads, 128x128, 4 slots,
4 state planes = MTP depth 3), plus a small generic-kernel geometry (64x64 heads):

1. Flag-off identity (fp32 state, the served path).
   --dump PATH  run in the SERVED image (tabbyapi:nvme-tier-r4-e3det): writes the fp32 cases (inputs and
                outputs) to PATH. Uses only served entry points.
   --ref PATH   run in the PATCHED image: replays the dumped inputs through the patched extension and
                requires torch.equal on every output and state tensor. Any mismatch exits 1.
   In-process identity (both images): gated_delta_rule_fn (chunked prefill) vs a direct vendored fla call,
   fp32 rewind via GDNLayerState jobs vs torch copy. The chunked prefill is NOT part of the cross-image
   dump gate: its Triton kernels autotune per process, which is independent of this patch.
2. Flag-on structure (patched image): verify window == the same tokens decoded one by one (torch.equal,
   fp32 and bf16), the MTP accept/rewind schedule == sequential decode of the accepted tokens (torch.equal),
   the bf16 rewind copies exactly one plane (sentinels), allocation / checkpoint size / default-off import.
   Any failure exits 1.
3. Flag-on numerics (patched image, informational): bf16 vs fp32 state and core output over --steps
   recurrent decode steps (error growth by decay class), over an MTP schedule, and over a --prefill-tokens
   chunked prefill (2048-token chunks) followed by decode. Optional kernel timing (--bench).

    python3 gpu_gdn_bf16.py --device cuda:0 --dump /out/served-fp32.pt            # served image
    python3 gpu_gdn_bf16.py --device cuda:0 --ref /out/served-fp32.pt --json r.json  # patched image
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch


NK, NV, DK, DV = 16, 48, 128, 128       # Qwen3.8-Flash-Next GDN layer
SLOTS, PLANES = 4, 4                    # 4 decode slots, max_history + 1 = 4 (MTP depth 3)
GEN_NK, GEN_NV, GEN_D = 4, 8, 64        # generic (non-128) kernel geometry
CHUNK = 2048                            # served max_chunk_size


def ext():
    from exllamav3.ext import exllamav3_ext
    return exllamav3_ext


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------

def head_log_rates(nv: int) -> torch.Tensor:
    """log10 |g| per head: three equal classes, slow (long memory), medium, fast decay"""
    c = nv // 3
    r = torch.empty(nv)
    r[:c] = torch.linspace(-4.0, -3.0, c)
    r[c:2 * c] = torch.linspace(-2.0, -1.0, c)
    r[2 * c:] = torch.linspace(-0.5, 0.3, nv - 2 * c)
    return r


def head_classes(nv: int) -> dict:
    c = nv // 3
    return {"slow": slice(0, c), "medium": slice(c, 2 * c), "fast": slice(2 * c, nv)}


def make_inputs(gen, bsz, seqlen, nk, nv, dk, dv, device):
    """Post-conv activations (silu), per-head log decay g (fp32, < 0) and beta (bf16), as the GDN forward
    hands them to the delta rule"""
    dim = 2 * nk * dk + nv * dv
    gdev = gen.device if hasattr(gen, "device") else torch.device("cpu")
    mixed = torch.nn.functional.silu(torch.randn(bsz, seqlen, dim, generator = gen, device = gdev)).to(torch.bfloat16)
    rates = head_log_rates(nv).to(gdev)
    g = -(10.0 ** rates).view(1, 1, nv) * torch.exp(0.25 * torch.randn(bsz, seqlen, nv, generator = gen, device = gdev))
    beta = torch.sigmoid(torch.randn(bsz, seqlen, nv, generator = gen, device = gdev)).to(torch.bfloat16)
    return mixed.to(device).contiguous(), g.float().to(device).contiguous(), beta.to(device).contiguous()


def make_state(gen, slots, planes, nv, dk, dv, device, dtype = torch.float):
    """bf16-representable start so the fp32 and bf16 arms begin from the same values"""
    gdev = gen.device if hasattr(gen, "device") else torch.device("cpu")
    s = (0.1 * torch.randn(slots, planes, nv, dk, dv, generator = gen, device = gdev)).to(torch.bfloat16).float()
    return s.to(device = device, dtype = dtype).contiguous()


def rule(mixed, g, beta, state, slots, history, nk, nv, dk, dv):
    bsz, seqlen = mixed.shape[:2]
    out = torch.empty(bsz, seqlen, nv, dv, dtype = torch.bfloat16, device = mixed.device)
    ext().cuda_recurrent_gated_delta_rule(mixed, g, beta, state, out, nk, nv, dk, dv, slots, history)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Part 1: flag-off identity cases (served entry points only)
# ---------------------------------------------------------------------------------------------------------------

IDENTITY_CASES = [
    # name, bsz, seqlen, history, geometry
    ("decode_bsz1", 1, 1, False, "qwen"),        # v_split 4
    ("decode_bsz4", 4, 1, False, "qwen"),        # v_split 1
    ("decode_bsz3_perm", 3, 1, False, "qwen"),
    ("verify_bsz1_q4", 1, 4, True, "qwen"),
    ("verify_bsz4_q4", 4, 4, True, "qwen"),
    ("verify_bsz2_q2", 2, 2, True, "qwen"),
    ("multitok_bsz2_q8", 2, 8, False, "qwen"),   # fused recurrent, no history (seqlen < num_v_heads)
    ("generic_decode_bsz1", 1, 1, False, "generic"),
    ("generic_verify_bsz2_q4", 2, 4, True, "generic"),
    ("generic_multitok_bsz4_q3", 4, 3, False, "generic"),
]


def geometry(name):
    return (NK, NV, DK, DV) if name == "qwen" else (GEN_NK, GEN_NV, GEN_D, GEN_D)


def build_identity_inputs(seed: int):
    gen = torch.Generator().manual_seed(seed)
    cases = {}
    for name, bsz, seqlen, history, geo in IDENTITY_CASES:
        nk, nv, dk, dv = geometry(geo)
        mixed, g, beta = make_inputs(gen, bsz, seqlen, nk, nv, dk, dv, "cpu")
        state = make_state(gen, SLOTS, PLANES, nv, dk, dv, "cpu")
        slots = torch.randperm(SLOTS, generator = gen)[:bsz].to(torch.int32)
        cases[name] = {"mixed": mixed, "g": g, "beta": beta, "state": state, "slots": slots}
    # rewind: state plane k -> plane 0 for a few (slot, last_history, num_tokens)
    cases["rewind"] = {"state": make_state(gen, SLOTS, PLANES, NV, DK, DV, "cpu")}
    return cases


REWIND_JOBS = [(0, 3, 1), (1, 3, 3), (2, 2, 2), (3, 3, 2)]  # (slot, last_history, num_tokens)


def run_identity(inputs: dict, device) -> dict:
    out = {}
    for name, bsz, seqlen, history, geo in IDENTITY_CASES:
        nk, nv, dk, dv = geometry(geo)
        c = inputs[name]
        state = c["state"].to(device).clone()
        o = rule(c["mixed"].to(device), c["g"].to(device), c["beta"].to(device), state,
                 c["slots"].to(device), history, nk, nv, dk, dv)
        out[name] = {"out": o.cpu(), "state": state.cpu()}
    state = inputs["rewind"]["state"].to(device).clone()
    layer = fake_layer_state(state)
    for host_gap in (False, True):
        s = state.clone()
        layer.recurrent_state = s
        jobs = rewind_jobs(layer, REWIND_JOBS, host_gap)
        ext().batched_state_rewind(jobs, torch.device(device).index)
        out[f"rewind_hostgap{int(host_gap)}"] = {"state": s.cpu()}
    torch.cuda.synchronize(device)
    return out


class _FakeGDN:
    def __init__(self, nv, dk, dv, dtype):
        self.fdim_qkv = 2 * NK * dk + nv * dv
        self.conv_kernel_size = 4
        self.num_v_heads, self.k_head_dim, self.v_head_dim = nv, dk, dv
        self.recurrent_state_dtype = dtype


def fake_layer_state(state: torch.Tensor):
    from exllamav3.modules.gated_delta_net import GDNLayerState
    slots, planes, nv, dk, dv = state.shape
    layer = GDNLayerState(_FakeGDN(nv, dk, dv, state.dtype), slots, planes - 1, 0)
    layer.recurrent_state = state
    layer.device = state.device
    return layer


def rewind_jobs(layer, jobs, host_gap: bool):
    import exllamav3.modules.gated_delta_net as gdn
    saved = gdn._host_gap_rewind
    gdn._host_gap_rewind = host_gap
    try:
        return [j for j in (layer.rewind_state_job(*spec) for spec in jobs) if j is not None]
    finally:
        gdn._host_gap_rewind = saved


def rewind_reference(state: torch.Tensor, jobs):
    s = state.clone()
    for slot, last_history, num_tokens in jobs:
        s[slot, 0] = s[slot, last_history + 1 - num_tokens]
    return s


# ---------------------------------------------------------------------------------------------------------------
# In-process identity (both images)
# ---------------------------------------------------------------------------------------------------------------

def prefill_fn(mixed, g, beta, state, save_state = True):
    from exllamav3.modules.gated_delta_net_fn import gated_delta_rule_fn
    return gated_delta_rule_fn(
        mixed_qkv = mixed, beta = beta, g = g, recurrent_state = state, recurrent_slots = None,
        history = False, save_state = save_state, num_k_heads = NK, num_v_heads = NV,
        k_dim = NK * DK, v_dim = NV * DV, k_head_dim = DK, v_head_dim = DV, params = {},
    )


def prefill_direct_fp32(mixed, g, beta, state):
    from exllamav3.vendor.fla import chunk_gated_delta_rule
    bsz, seqlen, _ = mixed.shape
    q, k, v = torch.split(mixed, [NK * DK, NK * DK, NV * DV], dim = -1)
    q = q.view(bsz, seqlen, -1, DK)
    k = k.view(bsz, seqlen, -1, DK)
    v = v.view(bsz, seqlen, -1, DV)
    s = state[0, 0].unsqueeze(0)
    o, new_state = chunk_gated_delta_rule(q, k, v, g = g, beta = beta, initial_state = s,
                                          output_final_state = True, use_qk_l2norm_in_kernel = True)
    s.copy_(new_state)
    return o


def in_process_identity(device, seed) -> dict:
    res = {}
    gen = torch.Generator(device = device).manual_seed(seed)
    mixed, g, beta = make_inputs(gen, 1, 2 * CHUNK, NK, NV, DK, DV, device)
    s0 = make_state(gen, 1, PLANES, NV, DK, DV, device)
    # warm both (Triton autotune) so the compared calls use settled configs
    prefill_fn(mixed, g, beta, s0.clone())
    prefill_direct_fp32(mixed, g, beta, s0.clone())
    a_state, b_state = s0.clone(), s0.clone()
    a = prefill_fn(mixed, g, beta, a_state)
    b = prefill_direct_fp32(mixed, g, beta, b_state)
    res["prefill_fn_vs_direct_fp32"] = bool(torch.equal(a, b) and torch.equal(a_state, b_state))
    st = make_state(gen, SLOTS, PLANES, NV, DK, DV, device)
    layer = fake_layer_state(st)
    ok = True
    for host_gap in (False, True):
        s = st.clone()
        layer.recurrent_state = s
        ext().batched_state_rewind(rewind_jobs(layer, REWIND_JOBS, host_gap), torch.device(device).index)
        ok = ok and torch.equal(s, rewind_reference(st, REWIND_JOBS))
    res["rewind_fp32_vs_torch_copy"] = bool(ok)
    return res


# ---------------------------------------------------------------------------------------------------------------
# Part 2: flag-on structure (patched image)
# ---------------------------------------------------------------------------------------------------------------

def structure_checks(device, seed) -> dict:
    res = {}
    gen = torch.Generator(device = device).manual_seed(seed + 1)
    for dtype in (torch.float, torch.bfloat16):
        tag = "bf16" if dtype == torch.bfloat16 else "fp32"
        for bsz, q, geo in ((1, 4, "qwen"), (4, 4, "qwen"), (2, 3, "qwen"), (2, 4, "generic")):
            nk, nv, dk, dv = geometry(geo)
            mixed, g, beta = make_inputs(gen, bsz, q, nk, nv, dk, dv, device)
            st = make_state(gen, SLOTS, PLANES, nv, dk, dv, device, dtype)
            slots = torch.randperm(SLOTS, device = device)[:bsz].to(torch.int32)
            # verify window with history
            sv = st.clone()
            ov = rule(mixed, g, beta, sv, slots, True, nk, nv, dk, dv)
            # the same tokens one by one (no history), keeping each intermediate state
            ss = st.clone()
            ok = True
            for t in range(q):
                ot = rule(mixed[:, t:t + 1].contiguous(), g[:, t:t + 1].contiguous(), beta[:, t:t + 1].contiguous(),
                          ss, slots, False, nk, nv, dk, dv)
                ok = ok and torch.equal(ot, ov[:, t:t + 1])
                plane = 0 if t == q - 1 else t + 1
                ok = ok and torch.equal(ss[slots.long(), 0], sv[slots.long(), plane])
            res[f"verify_eq_sequential_{tag}_{geo}_bsz{bsz}_q{q}"] = bool(ok)
        res[f"mtp_schedule_eq_sequential_{tag}"] = bool(mtp_schedule(device, seed + 2, dtype, steps = 64)["equal_to_sequential"])

    # bf16 rewind copies exactly one plane: planes 1..3 of the slot and the next slot stay untouched
    for host_gap in (False, True):
        st = make_state(gen, SLOTS, PLANES, NV, DK, DV, device, torch.bfloat16)
        layer = fake_layer_state(st)
        before = st.clone()
        ext().batched_state_rewind(rewind_jobs(layer, [(0, 3, 1)], host_gap), torch.device(device).index)
        ok = torch.equal(st[0, 0], before[0, 3]) and torch.equal(st[0, 1:], before[0, 1:]) and torch.equal(st[1:], before[1:])
        res[f"rewind_bf16_one_plane_hostgap{int(host_gap)}"] = bool(ok)
        s2 = before.clone()
        layer.recurrent_state = s2
        ext().batched_state_rewind(rewind_jobs(layer, REWIND_JOBS, host_gap), torch.device(device).index)
        res[f"rewind_bf16_vs_torch_copy_hostgap{int(host_gap)}"] = bool(torch.equal(s2, rewind_reference(before, REWIND_JOBS)))

    # allocation / checkpoint accounting
    from exllamav3.modules.gated_delta_net import GDNLayerState
    f = GDNLayerState(_FakeGDN(NV, DK, DV, torch.float), SLOTS, PLANES - 1, 0)
    b = GDNLayerState(_FakeGDN(NV, DK, DV, torch.bfloat16), SLOTS, PLANES - 1, 0)

    class _NoAttr(_FakeGDN):
        pass
    m = _NoAttr(NV, DK, DV, torch.float)
    del m.recurrent_state_dtype
    n = GDNLayerState(m, SLOTS, PLANES - 1, 0)
    served_cp = f.module.fdim_qkv * 4 * 2 + NV * DK * DV * 4
    res["alloc_fp32_default"] = n.recurrent_state.dtype == torch.float and f.recurrent_state.dtype == torch.float
    res["alloc_bf16"] = b.recurrent_state.dtype == torch.bfloat16
    res["checkpoint_size_fp32_unchanged"] = f.get_checkpoint_size() == served_cp == n.get_checkpoint_size()
    res["checkpoint_size_bf16"] = b.get_checkpoint_size() == f.module.fdim_qkv * 4 * 2 + NV * DK * DV * 2
    res["storage_bytes"] = {"fp32": f.storage_size(), "bf16": b.storage_size()}

    # import-time selector
    code = "import exllamav3.modules.gated_delta_net as g; print(int(g._gdn_state_bf16))"
    env_off = {k: v for k, v in os.environ.items() if k != "EXL3_GDN_STATE_BF16"}
    off = subprocess.run([sys.executable, "-c", code], env = env_off, capture_output = True, text = True)
    on = subprocess.run([sys.executable, "-c", code], env = {**env_off, "EXL3_GDN_STATE_BF16": "1"},
                        capture_output = True, text = True)
    res["selector_default_off"] = off.returncode == 0 and off.stdout.strip().endswith("0")
    res["selector_on"] = on.returncode == 0 and on.stdout.strip().endswith("1")
    return res


# ---------------------------------------------------------------------------------------------------------------
# Part 3: numerics
# ---------------------------------------------------------------------------------------------------------------

def rel_state(b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """relative Frobenius error per (slot, head) of plane 0: [..., H, K, V] -> [..., H]"""
    d = (b.float() - f).flatten(-2).norm(dim = -1)
    return d / f.flatten(-2).norm(dim = -1).clamp_min(1e-30)


def rel_out(b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """relative error per (token, head) of the core output: [bsz, q, H, V] -> [bsz, q, H]"""
    d = (b.float() - f.float()).norm(dim = -1)
    return d / f.float().norm(dim = -1).clamp_min(1e-30)


def cos_out(b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(b.float(), f.float(), dim = -1)


def by_class(t: torch.Tensor, nv: int) -> dict:
    """t [..., H] -> {class: {max, mean}}"""
    out = {}
    for name, sl in head_classes(nv).items():
        x = t[..., sl]
        out[name] = {"max": float(x.max()), "mean": float(x.mean())}
    return out


def decode_error(device, seed, steps, checkpoints) -> list:
    gen = torch.Generator(device = device).manual_seed(seed + 3)
    s0 = make_state(gen, SLOTS, PLANES, NV, DK, DV, device)
    sf, sb = s0.clone(), s0.to(torch.bfloat16)
    slots = torch.arange(SLOTS, device = device, dtype = torch.int32)
    rows = []
    out_rel_acc, out_cos_min = [], []
    for step in range(1, steps + 1):
        mixed, g, beta = make_inputs(gen, SLOTS, 1, NK, NV, DK, DV, device)
        of = rule(mixed, g, beta, sf, slots, False, NK, NV, DK, DV)
        ob = rule(mixed, g, beta, sb, slots, False, NK, NV, DK, DV)
        out_rel_acc.append(rel_out(ob, of))
        out_cos_min.append(cos_out(ob, of).min())
        if step in checkpoints:
            rs = rel_state(sb[:, 0], sf[:, 0])                            # [slots, H]
            one_round = rel_state(sf[:, 0].to(torch.bfloat16), sf[:, 0])  # storing fp32 once as bf16
            ro = torch.stack(out_rel_acc[-min(16, len(out_rel_acc)):])   # last <=16 steps
            rows.append({
                "step": step,
                "state_rel": by_class(rs, NV),
                "state_rel_one_rounding": by_class(one_round, NV),
                "state_growth_vs_one_rounding": {
                    k: rs[..., sl].max().item() / max(one_round[..., sl].max().item(), 1e-30)
                    for k, sl in head_classes(NV).items()
                },
                "state_max_abs": float((sb[:, 0].float() - sf[:, 0]).abs().max()),
                "out_rel_last16": by_class(ro, NV),
                "out_cos_min_so_far": float(torch.stack(out_cos_min).min()),
            })
    return rows


def mtp_schedule(device, seed, dtype, steps = 128, depth = 3) -> dict:
    """Verify windows of depth+1 tokens with history, random acceptance, rewind through GDNLayerState jobs and
    ext.batched_state_rewind (as GDNState.rewind does). Returns the final plane-0 state and whether it equals
    decoding only the accepted tokens one by one."""
    gen = torch.Generator(device = device).manual_seed(seed)
    q = depth + 1
    st = make_state(gen, SLOTS, PLANES, NV, DK, DV, device, dtype)
    seq = st.clone()
    layer = fake_layer_state(st)
    slots = torch.arange(SLOTS, device = device, dtype = torch.int32)
    equal = True
    outs = []
    for _ in range(steps):
        mixed, g, beta = make_inputs(gen, SLOTS, q, NK, NV, DK, DV, device)
        ov = rule(mixed, g, beta, st, slots, True, NK, NV, DK, DV)
        accepted = int(torch.randint(1, q + 1, (1,), generator = gen, device = device).item())
        rejected = q - accepted
        if rejected:
            jobs = rewind_jobs(layer, [(s, q - 1, rejected) for s in range(SLOTS)], True)
            ext().batched_state_rewind(jobs, torch.device(device).index)
        for t in range(accepted):
            os_ = rule(mixed[:, t:t + 1].contiguous(), g[:, t:t + 1].contiguous(), beta[:, t:t + 1].contiguous(),
                       seq, slots, False, NK, NV, DK, DV)
            equal = equal and torch.equal(os_, ov[:, t:t + 1])
        outs.append(ov[:, :accepted])
        equal = equal and torch.equal(st[:, 0], seq[:, 0])
    return {"equal_to_sequential": bool(equal), "state": st[:, 0].clone(), "outs": outs}


def mtp_error(device, seed, steps) -> dict:
    f = mtp_schedule(device, seed + 4, torch.float, steps)
    b = mtp_schedule(device, seed + 4, torch.bfloat16, steps)
    ro = torch.cat([rel_out(ob, of).flatten(0, 1) for ob, of in zip(b["outs"], f["outs"])])
    return {
        "windows": steps,
        "accepted_tokens": int(ro.shape[0] // SLOTS),
        "state_rel": by_class(rel_state(b["state"], f["state"]), NV),
        "out_rel": by_class(ro, NV),
        "fp32_equal_to_sequential": f["equal_to_sequential"],
        "bf16_equal_to_sequential": b["equal_to_sequential"],
    }


def prefill_error(device, seed, tokens, decode_after) -> dict:
    gen = torch.Generator(device = device).manual_seed(seed + 5)
    s0 = make_state(gen, 1, PLANES, NV, DK, DV, device)
    s0.zero_()                                          # a fresh prompt starts from zero state
    sf, sb = s0.clone(), s0.to(torch.bfloat16)
    rows = []
    for c in range(tokens // CHUNK):
        mixed, g, beta = make_inputs(gen, 1, CHUNK, NK, NV, DK, DV, device)
        of = prefill_fn(mixed, g, beta, sf)
        ob = prefill_fn(mixed, g, beta, sb)
        rows.append({
            "tokens": (c + 1) * CHUNK,
            "state_rel": by_class(rel_state(sb[:, 0], sf[:, 0]), NV),
            "state_rel_one_rounding": by_class(rel_state(sf[:, 0].to(torch.bfloat16), sf[:, 0]), NV),
            "out_rel": by_class(rel_out(ob, of), NV),
            "out_cos_min": float(cos_out(ob, of).min()),
        })
    slots = torch.zeros(1, device = device, dtype = torch.int32)
    dec = []
    for step in range(1, decode_after + 1):
        mixed, g, beta = make_inputs(gen, 1, 1, NK, NV, DK, DV, device)
        of = rule(mixed, g, beta, sf, slots, False, NK, NV, DK, DV)
        ob = rule(mixed, g, beta, sb, slots, False, NK, NV, DK, DV)
        dec.append(rel_out(ob, of))
    after = {"decode_steps": decode_after}
    if dec:
        after["out_rel"] = by_class(torch.stack(dec), NV)
        after["state_rel"] = by_class(rel_state(sb[:, 0], sf[:, 0]), NV)
    return {"chunks": rows, "decode_after": after}


def bench(device, seed, iters = 200) -> dict:
    gen = torch.Generator(device = device).manual_seed(seed + 6)
    res = {}
    for dtype in (torch.float, torch.bfloat16):
        tag = "bf16" if dtype == torch.bfloat16 else "fp32"
        for bsz, q, hist in ((1, 1, False), (4, 1, False), (1, 4, True), (4, 4, True)):
            mixed, g, beta = make_inputs(gen, bsz, q, NK, NV, DK, DV, device)
            st = make_state(gen, SLOTS, PLANES, NV, DK, DV, device, dtype)
            slots = torch.arange(bsz, device = device, dtype = torch.int32)
            for _ in range(10):
                rule(mixed, g, beta, st, slots, hist, NK, NV, DK, DV)
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            e0.record()
            for _ in range(iters):
                rule(mixed, g, beta, st, slots, hist, NK, NV, DK, DV)
            e1.record()
            torch.cuda.synchronize(device)
            res[f"{tag}_bsz{bsz}_q{q}{'_hist' if hist else ''}_us"] = 1000.0 * e0.elapsed_time(e1) / iters
    return res


# ---------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default = "cuda:0")
    ap.add_argument("--json", default = None, help = "write the full report here")
    ap.add_argument("--dump", default = None, help = "served image: write fp32 identity cases to this file and stop")
    ap.add_argument("--ref", default = None, help = "patched image: torch.equal the fp32 path against this dump")
    ap.add_argument("--seed", type = int, default = 1234)
    ap.add_argument("--steps", type = int, default = 512)
    ap.add_argument("--mtp-windows", type = int, default = 128)
    ap.add_argument("--prefill-tokens", type = int, default = 32768)
    ap.add_argument("--decode-after-prefill", type = int, default = 64)
    ap.add_argument("--no-bench", action = "store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report = {"device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "env_EXL3_GDN_STATE_BF16": os.environ.get("EXL3_GDN_STATE_BF16")}
    failures = []

    if args.dump:
        inputs = build_identity_inputs(args.seed)
        outputs = run_identity(inputs, device)
        torch.save({"seed": args.seed, "inputs": inputs, "outputs": outputs}, args.dump)
        report["dump"] = {"path": args.dump, "cases": sorted(outputs)}
        report["in_process_identity"] = in_process_identity(device, args.seed)
        failures += [k for k, v in report["in_process_identity"].items() if not v]
    else:
        if args.ref:
            ref = torch.load(args.ref, map_location = "cpu")
            outputs = run_identity(ref["inputs"], device)
            ident = {}
            for case, tensors in ref["outputs"].items():
                ident[case] = all(torch.equal(tensors[k], outputs[case][k]) for k in tensors)
            report["flag_off_identity_vs_served"] = ident
            failures += [f"identity:{k}" for k, v in ident.items() if not v]
        report["in_process_identity"] = in_process_identity(device, args.seed)
        failures += [k for k, v in report["in_process_identity"].items() if not v]

        st = structure_checks(device, args.seed)
        report["structure"] = st
        failures += [k for k, v in st.items() if isinstance(v, bool) and not v]

        t0 = time.time()
        cps = sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, args.steps} & set(range(1, args.steps + 1)))
        report["decode_error"] = decode_error(device, args.seed, args.steps, set(cps))
        report["mtp_error"] = mtp_error(device, args.seed, args.mtp_windows)
        if not report["mtp_error"]["fp32_equal_to_sequential"] or not report["mtp_error"]["bf16_equal_to_sequential"]:
            failures.append("mtp_error:equal_to_sequential")
        if args.prefill_tokens >= CHUNK:
            report["prefill_error"] = prefill_error(device, args.seed, args.prefill_tokens, args.decode_after_prefill)
        report["numerics_seconds"] = time.time() - t0
        if not args.no_bench:
            report["bench_us"] = bench(device, args.seed)

    report["failures"] = failures
    report["pass"] = not failures
    text = json.dumps(report, indent = 1)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text + "\n")
    print(text if not args.json else json.dumps({"pass": report["pass"], "failures": failures, "json": args.json}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
