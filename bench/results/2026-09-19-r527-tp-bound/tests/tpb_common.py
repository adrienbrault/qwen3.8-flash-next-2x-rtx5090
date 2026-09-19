"""TP=2 bounding probe: shared helpers (no GPU calls here; imported by the GPU probe and by the CPU tests).

Every half-module is built by the engine's OWN tensor-parallel import code (the functions exllamav3's TP loader
runs in each worker process), driven by an in-process producer/consumer pair instead of the shared-memory arena:

  model/model_tp_shared.py:26   SMProducer.send(tensor)                       -> StubProducer.send
  model/model_tp_shared.py:202  SMConsumer.recv(imp, cuda, slice_dim, first, last)  -> StubConsumer.recv
                                (narrow() before the copy, same as the real consumer)
  model/model_tp_alloc.py:156   TPAllocator.compile_tp_plan -> plan[key] = (first, last, unit)
  util/misc.py:74               ratio_split (how the allocator divides channel units between ranks)

Half builders (all with skip_reduction=True: the all-reduce is timed separately and added in the budget):
  GatedDeltaNet.tp_import        gated_delta_net.py:1299   unit "K-heads"   (16 K-heads -> 8 + 8)
  Attention.tp_import            attn.py:1253              unit "heads"     (2 KV heads -> 1 + 1, 24 Q -> 12 + 12)
  BlockSparseMLP.tp_import       block_sparse_mlp.py:1491  unit "channels"  (640 -> 384 + 256, 128-channel units)
  GatedMLP.tp_import (shared)    mlp.py:854                unit "channels"  (640 -> 384 + 256)
  Linear.tp_import_split         linear.py:731             column split of lm_head (vocab)
"""
from __future__ import annotations

import math

TP = 2
PAGE_SIZE_DEFAULT = 256


# ------------------------------------------------------------------------------------------------------------------
# In-process stand-ins for the TP shared-memory transport
# ------------------------------------------------------------------------------------------------------------------

class StubProducer:
    """Same contract as SMProducer.send (model_tp_shared.py:61): returns a descriptor dict; None -> none_tensor."""

    def __init__(self):
        self.sent = 0

    def send(self, tensor, cache_id=None):
        if tensor is None:
            return {"method": "none_tensor"}
        self.sent += 1
        return {"method": "local", "tensor": tensor}


class StubConsumer:
    """Same contract as SMConsumer.recv (model_tp_shared.py:202-257): optional narrow(slice_dim, first, last-first)
    BEFORE the copy, then a contiguous copy to the worker device (cuda=True) or a contiguous clone."""

    def __init__(self, device):
        self.device = device
        self.calls = []

    def recv(self, imp, cuda=False, slice_dim=None, first=None, last=None):
        self.calls.append((slice_dim, first, last))
        if imp.get("method") == "none_tensor":
            return None
        assert imp.get("method") == "local", f"unexpected descriptor {imp.get('method')}"
        t = imp["tensor"]
        if slice_dim is not None:
            t = t.narrow(slice_dim, first, last - first)
        if cuda:
            return t.to(self.device, copy=True, memory_format=_contiguous())
        return t.clone(memory_format=_contiguous())


def _contiguous():
    import torch
    return torch.contiguous_format


def local_context(device):
    """The dict TP workers pass to every tp_import (model_tp_fn.py): device, consumer, output_device.
    output_device == device makes BlockSparseMLP.tp_import keep the (replicated) router locally."""
    return {"device": device, "consumer": StubConsumer(device), "output_device": device}


# ------------------------------------------------------------------------------------------------------------------
# Plans
# ------------------------------------------------------------------------------------------------------------------

def rank_ranges(n_units: int, width: int, ratio_split=None, n_ranks: int = TP):
    """Contiguous per-rank ranges exactly as TPAllocator.compile_tp_plan builds them (model_tp_alloc.py:156-176):
    units are divided by ratio_split(n_units, equal weights, chunk_size=1), then scaled by channel_width."""
    if ratio_split is None:
        ratio_split = ratio_split_ref
    counts = ratio_split(n_units, [1] * n_ranks, chunk_size=1)
    out, start = [], 0
    for c in counts:
        out.append((start * width, (start + c) * width))
        start += c
    return out


def ratio_split_ref(d, weights, chunk_size=128):
    """Verbatim copy of exllamav3/util/misc.py:74-86 (used when the package is not importable; the CPU test asserts
    it matches the real function)."""
    assert d % chunk_size == 0, "Total must be divisible by chunk size"
    total_chunks = d // chunk_size
    total_weight = sum(weights)
    ideal_chunks = [total_chunks * w / total_weight for w in weights]
    base_chunks = [int(c) for c in ideal_chunks]
    remainder = total_chunks - sum(base_chunks)
    residuals = [c - int(c) for c in ideal_chunks]
    for i in sorted(range(len(residuals)), key=lambda i: -residuals[i])[:remainder]:
        base_chunks[i] += 1
    final_alloc = [c * chunk_size for c in base_chunks]
    assert sum(final_alloc) == d
    return final_alloc


def gdn_plans(gdn, ratio_split=None):
    """GatedDeltaNet.make_tp_allocation (gated_delta_net.py:1204-1248): unit K-heads, channel_width 1 when
    k_head_dim >= 128."""
    cw, units = 1, gdn.num_k_heads
    while cw * gdn.k_head_dim < 128:
        assert units % 2 == 0
        cw *= 2
        units //= 2
    return [{gdn.key: (a, b, "K-heads")} for a, b in rank_ranges(units, cw, ratio_split)]


def attn_plans(attn, ratio_split=None):
    """Attention.make_tp_allocation (attn.py:1145-1188): unit heads (KV heads), channel_width 1 when head_dim >= 128."""
    cw, units = 1, attn.num_kv_heads
    while cw * attn.head_dim < 128:
        assert units % 2 == 0
        cw *= 2
        units //= 2
    return [{attn.key: (a, b, "heads")} for a, b in rank_ranges(units, cw, ratio_split)]


def moe_plans(mlp, ratio_split=None):
    """BlockSparseMLP.make_tp_allocation with moe_tensor_split (block_sparse_mlp.py:1399-1437): 128-channel units of
    the expert intermediate width; the shared GatedMLP likewise (mlp.py:794-823)."""
    moe_units = mlp.ups[0].out_features // 128
    r_moe = rank_ranges(moe_units, 128, ratio_split)
    plans = []
    if mlp.shared_experts is not None:
        sh = mlp.shared_experts
        r_sh = rank_ranges(sh.ups[0].out_features // 128, 128, ratio_split)
    for r in range(TP):
        p = {mlp.key: (r_moe[r][0], r_moe[r][1], "channels")}
        if mlp.shared_experts is not None:
            p[sh.key] = (r_sh[r][0], r_sh[r][1], "channels")
        plans.append(p)
    return plans


def vocab_ranges(n_out: int, ratio_split=None):
    """Linear.make_tp_allocation (linear.py:674-692): 128-channel units of out_features."""
    return rank_ranges(n_out // 128, 128, ratio_split)


# ------------------------------------------------------------------------------------------------------------------
# Half builders (engine TP import paths)
# ------------------------------------------------------------------------------------------------------------------

def build_half_gdn(full, plan, device, ext, GatedDeltaNet):
    """GatedDeltaNet.tp_import on a stub transport. Fixes one engine-TP defect on the way: GatedRMSNorm.tp_export
    (gated_rmsnorm.py:134-149) drops gate_activation and tp_import rebuilds BC_GatedRMSNorm with gate_act = 0 (silu),
    while qwen4_exp's GDN norm is sigmoid-gated (qwen4_exp.py build_qwen4_block, output_gate_type). Same kernel, but the
    halves would not sum to the full layer. The half's norm BC is rebuilt with the served gate_act and its fused
    BC_GatedDeltaNetSplit is rebuilt around it (load_local, gated_delta_net.py:592)."""
    exported = full.tp_export(plan, StubProducer())
    half = GatedDeltaNet.tp_import(local_context(device), exported, plan, skip_reduction=True)
    act = full.norm.gate_activation
    n = half.norm
    fixed = n.gate_activation != act
    n.gate_activation = act
    # BC_GatedRMSNorm(weight, rms_norm_eps, constant_bias, w_groups, gate_first, gate_act): gated_rmsnorm_bc.h
    n.bc = ext.BC_GatedRMSNorm(n.weight, n.rms_norm_eps, n.constant_bias, n.groups, n.gate_first,
                               1 if act == "sigmoid" else 0)
    half.load_local(device)
    return half, {"norm_gate_activation_fixed": fixed, "gate_activation": act,
                  "num_k_heads": half.num_k_heads, "num_v_heads": half.num_v_heads,
                  "fdim_qkv": half.fdim_qkv, "bc_split": bool(half.bc_split),
                  "qkvz_bundle": getattr(half, "multi_qkvz", None) is not None}


def build_half_attn(full, plan, device, Attention):
    """Attention.tp_import with the QSA indexer REPLICATED (DESIGN-TP §5.1 v1). Attention.tp_export refuses a module
    with an indexer (attn.py:1193), so the indexer is detached for the export and the loaded indexer object (same
    device) is attached to the half afterwards: both ranks run the full indexer, exactly the replicate policy."""
    idx = full.qsa_indexer
    full.qsa_indexer = None
    try:
        exported = full.tp_export(plan, StubProducer())
    finally:
        full.qsa_indexer = idx
    half = Attention.tp_import(local_context(device), exported, plan, skip_reduction=True)
    half.qsa_indexer = idx
    return half, {"num_q_heads": half.num_q_heads, "num_kv_heads": half.num_kv_heads,
                  "q_out": half.q_proj.out_features, "kv_out": half.k_proj.out_features,
                  "o_in": half.o_proj.in_features,
                  "qkv_bundle": getattr(half, "multi_qkv", None) is not None}


def build_half_moe(full, plan, device, BlockSparseMLP):
    """BlockSparseMLP.tp_import, unit "channels" (tensor split, no expert parallelism): every expert's gate/up is
    column-sliced and down row-sliced (block_sparse_mlp.py:1531-1540); the shared expert likewise through
    GatedMLP.tp_import; router and shared gate replicated."""
    exported = full.tp_export(plan, StubProducer())
    half = BlockSparseMLP.tp_import(local_context(device), exported, plan, skip_reduction=True)
    return half, {"intermediate_size": half.intermediate_size,
                  "num_local_experts": half.num_local_experts,
                  "shared_interm": half.shared_experts.ups[0].out_features if half.shared_experts else None,
                  "bc_built": half.bc is not None, "bc_sh_exp": bool(half.bc_sh_exp),
                  "min_max_expert": (half.experts_cfg.min_expert, half.experts_cfg.max_expert)}


def build_half_linear_cols(full, first, last, device, Linear):
    """Linear.tp_import_split column split (linear.py:731-746)."""
    plan = {}
    exported = full.tp_export(plan, StubProducer())
    half = Linear.tp_import_split(local_context(device), exported, plan, (True, first, last))
    return half


def build_half_hc(full, first, last, GatedResidual):
    """Hidden-channel slice of a gated-residual site: the sensitivity arm for a SHARDED hyper-connection mixer (HF
    tp plan rowwise_split_input on input_mix_weight_down). The design's v1 and exllamav3's own export
    (hyperconnections.py:497) REPLICATE the mixer; the primary budget uses s = 1 for it. This half is timing-faithful
    only: its dots are partial and a sharded mixer needs an extra all-reduce of them (counted in the join)."""
    H, D = full.hc_mult, full.hidden_size
    d = last - first
    half = GatedResidual(config=None, key=full.key + ".tp_half", hc_mult=H, hidden_size=d,
                         rms_norm_eps=full.rms_eps, use_combine=full.use_combine, out_dtype=full.out_dtype)
    half.device = full.device
    half.norm_w_raw = full.norm_w_raw.reshape(H, D)[:, first:last].reshape(-1).contiguous()
    rank = full.rank
    down = full.down_h.reshape(rank, H, D)[:, :, first:last].reshape(rank, H * d).contiguous()
    up = full.up_h.reshape(H, D, rank)[:, first:last, :].reshape(H * d, rank).contiguous()
    inject = None
    if full.inject_h is not None:
        inject = full.inject_h.reshape(H, H, D)[:, :, first:last].reshape(H, H * d).contiguous()
    half._prepare(down, up, inject)
    return half


# ------------------------------------------------------------------------------------------------------------------
# Forward-call parameter dicts (served decode shapes)
# ------------------------------------------------------------------------------------------------------------------

def gdn_params(GDNState, get_slot_tensor, cache_key: int, bsz: int, history: bool = True):
    """The params the generator hands a GDN layer at decode (cache/recurrent_util.py prepare_for_recurrence +
    generator 'recurrent_history' = draft verification). GDNState(exported=True) is the TP-worker form: the layer
    resolves its state through module.tp_recurrent_lookup[cache] (gated_delta_net.py:1019-1022)."""
    states = [GDNState(cache=cache_key, slot=i, position=0, exported=True) for i in range(bsz)]
    return {"recurrent_states": states,
            "recurrent_slots": get_slot_tensor(tuple(range(bsz))),
            "recurrent_history": history}


def attn_pages_per_seq(ctx: int, qlen: int, page_size: int = PAGE_SIZE_DEFAULT) -> int:
    return (ctx + qlen + page_size - 1) // page_size + 1


def attn_params(torch, prepare_for_attn, cache_layer, bsz: int, qlen: int, ctx: int,
                page_size: int = PAGE_SIZE_DEFAULT):
    """Paged-attention params in the generator's 'block_table' form (attn.py prepare_flash_attn, second branch):
    per-job distinct pages, cache_seqlens = context length (int32, host), positions = cache_seqlens."""
    pps = attn_pages_per_seq(ctx, qlen, page_size)
    params = {
        "attn_mode": "flash_attn",
        "cache": cache_layer,
        "block_table": torch.arange(bsz * pps, dtype=torch.int32).view(bsz, pps),
        "cache_seqlens": torch.full((bsz,), ctx, dtype=torch.int32),
    }
    prepare_for_attn(torch.zeros((bsz, qlen), dtype=torch.long), params)
    return params


def attn_cache_tokens(max_bsz: int, ctx: int, max_qlen: int, page_size: int = PAGE_SIZE_DEFAULT) -> int:
    return max_bsz * attn_pages_per_seq(ctx, max_qlen, page_size) * page_size


# ------------------------------------------------------------------------------------------------------------------
# Correctness metric for "sum of rank halves == full" (row-parallel outputs)
# ------------------------------------------------------------------------------------------------------------------

def sum_check(full, parts, tol_rel=2e-2, min_cos=0.9999):
    """full, parts: tensors of equal shape (fp32 compare). rel = max|full - sum(parts)| / max|full|."""
    f = full.float().flatten()
    s = parts[0].float().flatten().clone()
    for p in parts[1:]:
        s += p.float().flatten()
    denom = float(f.abs().max()) or 1.0
    rel = float((f - s).abs().max()) / denom
    fn, sn = float(f.norm()), float(s.norm())
    cos = float((f * s).sum()) / (fn * sn) if fn > 0 and sn > 0 else (1.0 if fn == sn else 0.0)
    finite = bool(math.isfinite(rel) and math.isfinite(cos))
    return {"rel_maxabs": rel, "cos": cos, "max_abs_full": denom,
            "ok": finite and rel <= tol_rel and cos >= min_cos}


def shape_key(bsz: int, seqlen: int) -> str:
    return f"{bsz}x{seqlen}"


def parse_shape(k: str):
    b, s = k.split("x")
    return int(b), int(s)
