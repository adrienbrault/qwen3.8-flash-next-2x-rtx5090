#!/usr/bin/env python3
"""CPU stand-in for the n-gram subset of exllamav3_ext, for tests that run the patched
NGramEmbedding end to end without a CUDA device or a real table.

ngram_hash_cpu is ported line for line from exllamav3_ext/ngram.cu (the eos-segmented hash,
the sort + dedup and the head assignment are simple arithmetic, so a faithful Python port is
bit-comparable with the C++ by construction of the test's expected values). ngram_gather_cpu
delegates to index_select on a fake handle's local tensor; ngram_dequant is the reference
dequant_rows of ngram_codec.py. DiskTensorHandle._ensure_open/read_rows are monkeypatched on
the class (the real one opens files).
"""
from __future__ import annotations

import threading

import torch


def ngram_hash_cpu(
    ids,                    # (bsz, ctx + seq) int64 CPU
    seq_len,
    multipliers,            # (ngram_size) int64 CPU
    offsets,                # (num_heads) int64 CPU
    sizes,                  # (num_heads) int64 CPU
    heads_per_ngram,
    eos_token,
    uids,                   # (cap) int64 CPU out
    inverse,                # (bsz * seq * num_heads) int64 CPU out
    heads,                  # (cap) int32 CPU out
):
    assert ids.device.type == "cpu" and ids.dtype == torch.int64 and ids.is_contiguous()
    bsz = ids.size(0)
    T = ids.size(1)
    ctx = T - seq_len
    ngram_size = multipliers.numel()
    H = offsets.numel()
    assert ctx >= 0 and H == (ngram_size - 1) * heads_per_ngram
    n = bsz * seq_len * H
    assert uids.numel() >= n and inverse.numel() >= n and heads.numel() >= n

    mult = multipliers.tolist()
    offs = offsets.tolist()
    szs = sizes.tolist()

    hk = []   # (hash, idx)
    M64 = (1 << 64) - 1
    for b in range(bsz):
        row = ids[b].tolist()
        prev_eos = -1
        for p in range(T):
            if p > 0 and row[p - 1] == eos_token:
                prev_eos = p - 1
            if p >= ctx:
                seg_start = prev_eos + 1
                mixed = (row[p] * mult[0]) & M64
                for s in range(1, ngram_size):
                    src = row[p - s] if (p - s >= seg_start and p - s >= 0) else eos_token
                    mixed ^= (src * mult[s]) & M64
                    mixed &= M64
                    lo = (s - 1) * heads_per_ngram
                    for h in range(lo, lo + heads_per_ngram):
                        # mirror the kernel: `(int64_t) mixed % szs[h]; if (m < 0) m += szs[h]`.
                        # C's % truncates toward zero and the +size adjust then equals Python's
                        # floor-mod on the signed reinterpretation, which is what torch's
                        # reference (torch.remainder on int64) computes as well.
                        m_signed = mixed - (1 << 64) if mixed >> 63 else mixed
                        m = m_signed % szs[h]
                        hk.append((m + offs[h], (b * seq_len + (p - ctx)) * H + h))
    hk.sort(key = lambda t: t[0])
    u = -1
    last = None
    for i, (hv, idx) in enumerate(hk):
        if last is None or hv != last:
            last = hv
            u += 1
            uids[u] = hv
            h = len([o for o in offs if o <= hv]) - 1     # upper_bound - 1
            heads[u] = h
        inverse[idx] = u
    return u + 1


def ngram_gather_cpu(fd, base_offset, row_bytes, uids, uid_base, out):
    assert uids.device.type == "cpu" and uids.dtype == torch.int64 and uids.is_contiguous()
    store = _FD_STORES.get(fd)
    assert store is not None, "fake handle not opened"
    U = uids.numel()
    assert out.size(0) >= U and out.size(1) * out.element_size() == row_bytes
    if not U:
        return
    out[:U] = store.index_select(0, (uids - uid_base).contiguous())


_FD_LOCK = threading.Lock()
_FD_STORES = {}
_FD_NEXT = [1000]


def register_table(rows: torch.Tensor) -> int:
    """Give the fake a backing store and return the 'fd' to hand _ensure_open()."""
    with _FD_LOCK:
        fd = _FD_NEXT[0]
        _FD_NEXT[0] += 1
        _FD_STORES[fd] = rows
    return fd


def ngram_dequant(packed, K, heads, bias, out):
    """Reference dequant via ngram_codec (fp16 rounding included there)."""
    from exllamav3.modules.quant.exl3_lib.ngram_codec import dequant_rows, mul1_codebook
    rows = dequant_rows(packed.cpu(), K, mul1_codebook("cpu"), None)
    # bias application mirrors the kernel: per-row head bias added to the decoded value,
    # matching fetch_rows' reference (dequant_rows with the per-row gathered bias)
    b = bias.cpu()[heads.long().cpu()].float().half().float()
    out.copy_((rows + b).half().to(out.device))     # the GPU harness passes device tensors (R559 try 2b)


def reset():
    with _FD_LOCK:
        _FD_STORES.clear()
