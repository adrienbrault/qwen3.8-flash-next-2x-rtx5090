"""Eager, RNG-free FLy sidecar. Native samples and serial consumption remain authoritative."""
import math
import torch


def record(job, key, n=1):
    job.fast_metrics[key] += n


def eligibility(job, mode, lookup_mask=None):
    if not job.name_policy.fast:
        return "native"
    if mode not in ("greedy", "sampled"):
        return "sampler"
    if len(job.sequences) != 1:
        return "sequences"
    if job.return_probs or job.return_top_tokens or job.return_logits:
        return "logprobs"
    if job.new_tokens < 0 or job.prefix_token is not None:
        return "healing"
    if job.forced_ids is not None or job.forced_sample:
        return "forced"
    if job.filters or getattr(job.sampler, "reqs_past_ids", False):
        return "stateful"
    if job.new_tokens < job.min_new_tokens:
        return "min_tokens"
    if job.checkpoint or job.checkpoint_rewound:
        return "checkpoint"
    pending = job.prompt_lookup_adaptive.pending
    if (lookup_mask is not None and any(lookup_mask)) or (
        pending and pending.get("state") != "OFF" and pending.get("chain") is not None
    ):
        return "lookup"
    return None


def processed_logits(logits, terminal, vocab_size, mask=None, ctrl=None):
    """Mirror the terminal's effective support on a PRIVATE FP32 tensor.

    Mode 3 uses the actual native FSCtrl, not a sorted top-p approximation.
    fs_bin's individual FP32 operations deliberately remain separate.
    """
    x = logits.reshape(-1, logits.shape[-1]).float().clone()
    width = min(x.shape[-1], vocab_size)
    if mask is not None:
        width = min(width, mask.shape[-1] * (32 if mask.dtype == torch.int32 else 1))
    x[:, width:] = -math.inf
    if mask is not None:
        if mask.dtype == torch.int32:
            ids = torch.arange(width, device=x.device)
            words = mask.to(device=x.device).reshape(mask.shape[0], -1).long()
            allowed = ((words[:, ids // 32] >> (ids % 32)) & 1).bool()
            x[:, :width].masked_fill_(~allowed, -math.inf)
        else:
            x[:, :width] += mask.to(device=x.device).reshape(mask.shape[0], -1)[:, :width].float()
    mode = getattr(terminal, "mode", 0)
    # NaN/+inf are invalid statistics; -inf is a legitimate masked value.
    if bool((torch.isnan(x) | torch.isposinf(x)).any()):
        return None
    keep = torch.isfinite(x)
    maximum = x.max(dim=-1, keepdim=True).values
    if mode == 2:
        keep &= x >= maximum + terminal.minp_log
    elif mode == 3:
        if ctrl is None:
            return None
        m, status, kb, ks = ctrl
        if bool((status != 0).any()) or not bool(torch.isfinite(m).all()):
            return None
        if terminal.filters & 4:
            keep &= x >= m + terminal.minp_log
        # FSCtrl::m is the native masked maximum; all maxima survive cutoff ties.
        scale = torch.tensor(32.0, device=x.device, dtype=torch.float32) * terminal.inv_temp_filter
        db = (m - x) * scale
        # Mask before float->int conversion, avoiding casts of +/-inf.
        db = torch.where(torch.isfinite(db), db, torch.zeros_like(db))
        b = db.to(torch.int32).clamp(max=1023)
        s = ((db - b.float()) * 1024.0).to(torch.int32).clamp(0, 1023)
        # Native comparisons cast b to unsigned. In particular, an overflowing
        # float->int bin must not become a negative bin that passes every cutoff.
        ub = b.long() & 0xffffffff
        keep &= (x == m) | (ub < kb) | ((ub == kb) & (s <= ks))
    elif mode not in (0, 1):
        return None
    inv_temp = 1.0 if mode == 0 else terminal.inv_temp
    if not math.isfinite(inv_temp) or inv_temp <= 0:
        return None
    return (x * inv_temp).masked_fill(~keep, -math.inf)


def native_ctrl(terminal, logits):
    if getattr(terminal, "mode", 0) != 3:
        return None
    rows = logits.numel() // logits.shape[-1]
    # sampling_fused.cuh: two 1024*(u64 mass + u32 count) histograms,
    # followed by FSCtrl {float m, uint status, ref_bucket, k_rem, keep_b, keep_s,...}.
    hist = terminal.histograms[(logits.device, rows)].view(rows, -1)
    words = hist[:, 24576:24600].contiguous().clone().view(torch.int32)
    return (words[:, 0:1].contiguous().view(torch.float32), words[:, 1:2],
            words[:, 4:5], words[:, 5:6])


def rescue_support(processed, draft):
    """Full-vocabulary normalization; no top-three renormalization."""
    if processed is None:
        return None
    support = torch.isfinite(processed).any(dim=-1)
    q = torch.softmax(processed, dim=-1)
    top = q.topk(min(3, q.shape[-1]), dim=-1).values
    h3 = -torch.special.xlogy(top, top).sum(dim=-1)
    if not bool(support.all()) or not bool(torch.isfinite(h3).all()):
        return None
    d = draft.to(processed.device).long().reshape(-1)
    n = d.numel()
    valid_ids = (d >= 0) & (d < processed.shape[-1])
    finite_draft = processed[:n].gather(1, d.clamp(0, processed.shape[-1]-1)[:, None]).isfinite().reshape(-1)
    return (support[:n] & torch.isfinite(h3[:n]) & (h3[:n] >= 0.3)
            & valid_ids & finite_draft).cpu().tolist()


def window(job, logits, drafts, mode, vocab_size, lookup_mask=None):
    reason = eligibility(job, mode, lookup_mask)
    if reason:
        if job.name_policy.fast:
            record(job, "fallback_" + reason)
        return None
    steps = getattr(job.sampler, "steps", ())
    if len(steps) != 1 or type(steps[0]).__name__ not in ("SS_Argmax", "SS_Fused"):
        record(job, "fallback_terminal")
        return None
    terminal = steps[0]
    mask = job.device_logit_mask
    if mask is not None and not (
        getattr(job.sampler, "fused_only", True) and mask.is_contiguous()
        and mask.dtype in (torch.half, torch.int32)
        and (mask.shape[0] == 1 or (mask.shape[0] == logits.numel() // logits.shape[-1]
             and (mask.dtype == torch.int32 or mask.shape[-1] == logits.shape[-1])))
    ):
        record(job, "fallback_mask_layout")
        return None
    L = processed_logits(logits, terminal, vocab_size, job.device_logit_mask,
                         native_ctrl(terminal, logits))
    allowed = rescue_support(L, drafts)
    if allowed is None:
        record(job, "fallback_statistics")
    else:
        record(job, "eligible_windows")
    return allowed


def choose(native, drafts, allowed, i):
    """W=1: only an immediate NATIVE match confirms; no decision survives a window."""
    if allowed is None or i + 1 >= len(drafts):
        return native[i], False
    if native[i] != drafts[i] and allowed[i] and native[i+1] == drafts[i+1]:
        return drafts[i], True
    return native[i], False
