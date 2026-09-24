#!/usr/bin/env python3
"""moefast V3, CPU-only model check (no GPU, no torch).

Replays the index arithmetic of the served V2 GEMV tile (exl3_moe_coop_v2_kernel.cuh: gemv_tile_v2) and of
V3 (exl3_moe_coop_v3_kernel.cuh: v3_stream, gemv_tile_v3, gemv_chunk_narrow_v3) as symbolic traces:

  for every output element (row, column): the ordered list of k-warp partials summed into it, and for each
  partial the ordered list of fp16 accumulation segments (k-slices accumulated in the MMA accumulator between
  two folds to fp32).

If the traces are equal, every output is the same sequence of the same fp16 MMAs, the same fp16 -> fp32 folds
and the same fp32 additions, i.e. bitwise identical (the MMA inputs are the same words: checked separately by
the stream replay below and, on the GPU, by test_moefast_parity.py).

It also replays V3's issue/consume counters (cp.async ring slots, unit jumps, activation pointer walk) and
checks that the ring slot consumed at each step holds the k-slice the partial expects and that the activation
fragment loaded for it is the same k-slice.

This is a model of the code, written from it by hand. It catches mapping mistakes, not compiler behaviour.
Run: python3 test_moefast_trace_cpu.py
"""
import itertools
import sys

WK = 16          # warps per block
PF = 4           # V2 prefetch ring depth
FOLD = 4         # V2 fold cadence
V3_PF = 4        # V3 cp.async ring slots
WNT = 2
COLS = WNT * 16  # 32


def ceil_div(a, b):
    return (a + b - 1) // b


def tile_wn(wide):
    return 4 if wide else 1


# ---------------------------------------------------------------------------------------------- V2 model
def v2_partial_segments(k_begin, k_end, wide, wk):
    """gemv_tile_v2: warp (wn, wk) of a block. Returns the fp16 segments (lists of k-slices) of its partial."""
    WKK = WK // tile_wn(wide)
    kslices = k_end - k_begin
    chunk = ceil_div(kslices, WKK)
    ks0 = k_begin + wk * chunk
    myn = max(0, min(chunk, k_end - ks0))
    segs, cur = [], []
    ib = 0
    while ib < myn:                              # for (ib = 0; ib < myn; ib += PF)
        for d in range(PF):                      #   for d < PF: i = ib + d; if (i >= myn) break
            i = ib + d
            if i >= myn:
                break
            cur.append(ks0 + i)                  #   MMA of slice ks0 + i into ch
            if (d + 1) % FOLD == 0 or i + 1 == myn:
                segs.append(tuple(cur))          #   fold ch -> acc0, ch = 0
                cur = []
        ib += PF
    assert not cur
    return tuple(segs)


def v2_trace(k_begin, k_end, wide, n_groups):
    """{(group, col_in_group): [partial segments in reduction order j = 0..WKK-1]}"""
    WN = tile_wn(wide)
    WKK = WK // WN
    tcols = WN * COLS
    out = {}
    for g in range(n_groups):
        for c in range(tcols):
            wn = c // COLS
            parts = []
            for j in range(WKK):                 # sum += sh_red[(j * red_rows + r) * TCOLS + c]
                warp = j * WN + wn               # wk = warp / WN = j, wn = warp % WN
                assert warp // WN == j and warp % WN == wn
                parts.append(v2_partial_segments(k_begin, k_end, wide, j))
            out[(g * tcols + c)] = tuple(parts)
    return out


# ---------------------------------------------------------------------------------------------- V3 model
def v3_stream(chunk, k_end, ks0, ustride, nq):
    """v3_stream: returns per-unit segments, and checks the ring / activation walk."""
    myn_q = [max(0, min(chunk, k_end - (ks0 + q * ustride))) for q in range(nq)]
    total = sum(myn_q)
    # myn non-increasing and only the last non-empty unit may be short
    for q in range(nq - 1):
        assert myn_q[q] >= myn_q[q + 1]
        if myn_q[q] < chunk:
            assert myn_q[q + 1] == 0, myn_q

    # issue side
    ring = [None] * V3_PF
    pending = []                                 # commit groups in order: slot or None (empty)
    isrc = ks0                                   # k-slice the next issue copies
    ileft, islot, ii = total, 0, 0

    def issue_next():
        nonlocal isrc, ileft, islot, ii
        if ileft > 0:
            pending.append((islot, isrc))
            ileft -= 1
            isrc += 1
            if nq > 1:
                ii += 1
                if ii == chunk:
                    ii = 0
                    isrc += ustride - chunk      # ujump
        else:
            pending.append(None)
        islot = (islot + 1) & (V3_PF - 1)

    completed = 0

    def wait_group(n):                           # at most n most recent groups pending
        nonlocal completed
        while len(pending) - completed > n:
            g = pending[completed]
            if g is not None:
                ring[g[0]] = g[1]
            completed += 1

    for _ in range(V3_PF - 1):
        issue_next()

    anext = ks0                                  # activation pointer, in k-slices
    act_loaded = anext if total > 0 else None
    cslot, left = 0, total
    units = []
    for q in range(nq):
        myn = myn_q[q]
        segs, cur = [], []
        for i in range(myn):
            wait_group(V3_PF - 2)
            # refill of the slot read in the previous step happens after the warp barrier
            issue_next()
            expect = ks0 + q * ustride + i
            assert ring[cslot] == expect, (q, i, cslot, ring, expect)
            assert act_loaded == expect, (q, i, act_loaded, expect)
            # the slot being refilled must not be the one consumed now
            assert pending[-1] is None or pending[-1][0] != cslot
            cur.append(expect)
            cslot = (cslot + 1) & (V3_PF - 1)
            left -= 1
            anext += 1
            if nq > 1 and i + 1 == myn:
                anext += ustride - chunk
            act_loaded = anext if left > 0 else None
            if ((i + 1) & (FOLD - 1)) == 0 or i + 1 == myn:
                segs.append(tuple(cur))
                cur = []
        assert not cur
        units.append(tuple(segs))
    wait_group(0)
    return units


def v3_trace_single(k_begin, k_end, wide, n_groups):
    """gemv_tile_v3: V3 stream with one unit per warp, V2 reduction."""
    WN = tile_wn(wide)
    WKK = WK // WN
    tcols = WN * COLS
    kslices = k_end - k_begin
    chunk = ceil_div(kslices, WKK)
    part = {}
    for warp in range(WK):
        wk = warp // WN
        ks0 = k_begin + wk * chunk
        (segs,) = v3_stream(chunk, k_end, ks0, 0, 1)
        part[warp] = segs
    out = {}
    for g in range(n_groups):
        for c in range(tcols):
            wn = c // COLS
            out[g * tcols + c] = tuple(part[j * WN + wn] for j in range(WKK))
    return out


def v3_trace_merge(k_begin, k_end, n_chunks128):
    """gemv_chunk_narrow_v3: warp w = (gg = w & 3, vk_lo = w >> 2), units vk = vk_lo + 4 q; running sum over
    rounds q, j = 0..3 (vk = 4 q + j)."""
    GPC, NQ = 4, WK // 4
    kslices = k_end - k_begin
    chunk = ceil_div(kslices, WK)
    part = {}                                    # (gg, vk) -> segments
    for warp in range(WK):
        gg, vk_lo = warp % GPC, warp // GPC
        units = v3_stream(chunk, k_end, k_begin + vk_lo * chunk, NQ * chunk, NQ)
        for q in range(NQ):
            part[(gg, vk_lo + q * NQ)] = units[q]
    out = {}
    for c128 in range(n_chunks128):
        for c in range(128):
            gg = c // COLS
            order = []
            for q in range(NQ):                  # rounds
                for j in range(GPC):             # s += sh_red[(j * ROWS + r) * 128 + c], vk = 4q + j
                    order.append(part[(gg, q * NQ + j)])
            out[c128 * 128 + c] = tuple(order)
    return out


# ---------------------------------------------------------------------------------------------- checks
def check(name, a, b):
    if a != b:
        bad = [k for k in a if a[k] != b.get(k)]
        print(f"FAIL {name}: {len(bad)} outputs differ, first {bad[0]}:\n  v2 {a[bad[0]]}\n  v3 {b.get(bad[0])}")
        return False
    return True


def main():
    ok = True
    n = 0
    # widths: served Hi = 2560 (A: 160 k-slices), I = 640 (B: 40 k-slices); plus other I/Hi to stress the
    # partition edge cases (short and empty k-warps), and split-k 1..4 (EXL3_MOE_COOP_KSPLIT)
    for kslices_all, ksplit, wide in itertools.product([160, 40, 8, 17, 33, 48, 64, 96, 128, 200], [1, 2, 3, 4],
                                                       [False, True]):
        for ks in range(ksplit):
            k_begin = kslices_all * ks // ksplit
            k_end = kslices_all * (ks + 1) // ksplit
            if k_end <= k_begin:
                continue
            ref = v2_trace(k_begin, k_end, wide, 2)
            ok &= check(f"single k{kslices_all} ksplit{ksplit}.{ks} wide{wide}", ref, v3_trace_single(k_begin, k_end, wide, 2))
            n += 1
            if not wide:
                ok &= check(f"merge k{kslices_all} ksplit{ksplit}.{ks}", v2_trace(k_begin, k_end, False, 8),
                            v3_trace_merge(k_begin, k_end, 2))
                n += 1
    # the served shapes explicitly
    served = {
        "A wide Hi 2560 (r4, r16)": (0, 160, True),
        "B narrow I 640 (r2..r12)": (0, 40, False),
        "B wide I 640 (r13+)": (0, 40, True),
    }
    for name, (kb, ke, wide) in served.items():
        ref = v2_trace(kb, ke, wide, 4)
        ok &= check(name, ref, v3_trace_single(kb, ke, wide, 4))
        segs = [len(p) for p in ref[0]]
        print(f"  {name}: k-warps {len(ref[0])}, slices per k-warp {[sum(len(s) for s in p) for p in ref[0]]}, "
              f"fold segments {segs}")
    ok &= check("B narrow merged I 640", v2_trace(0, 40, False, 80), v3_trace_merge(0, 40, 20))
    print(f"{n} partition/ksplit cases + served shapes: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
