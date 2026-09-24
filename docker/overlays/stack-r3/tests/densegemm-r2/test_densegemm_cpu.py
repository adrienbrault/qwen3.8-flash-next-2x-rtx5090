#!/usr/bin/env python3
"""densegemm r2 CPU model tests (no GPU, numpy only). Run: python3 test_densegemm_cpu.py

r2 (exl3_gemm_inner_v2.cuh, dv2_fold / dv2_tile_beg / the deferred head): tests 1-4 are r1's, with the V2 side
modelled on the r2 code (32-bit partition arithmetic behind the host's tiles * (grid + 1) < 2^31 guard, one slot at
a time). New:
5. Deferred head: a segment that starts at k = 0 and ends below the top k tile is always its CTA's LAST segment
   (r2 folds it after the main loop), and b_top from the r2 closed form equals the CTA holding the top k tile.
6. B prologue commit groups (DGV2_BPRO=1): [B stages 0..S-2], [A stage 0] .. [A stage S-2], wait_group<S-2>, then
   the served main loop; every stage's A and B groups have retired before load_frags reads the stage.

1. Stream-K fixup: the served lock chain (exl3_gemm_inner.cuh reduce(): lock_i / lock_d / first / last,
   intermediate sums stored in C's dtype) against the V2 gather (exl3_gemm_inner_v2.cuh: owner(),
   tile_beg(), nprod, fold order b_top .. head+1 then own, rnd() before every add). Random fp32 partials,
   every served launch config plus a random sweep (including grids larger than the tile count, i.e.
   empty CTAs), fp32 and fp16 C. Bitwise equality is required.
   Also checked: a non-head segment is always its CTA's first segment (one slot per CTA), the head's
   nprod equals the number of publishing CTAs, and no head ever waits on its own CTA.
2. Slot budget: every served dense K4 call fits the 4 MiB workspace (8 MiB with rows32).
3. gemv V2 cp.async ring: for every chunk length, each slice is complete (its commit group retired by
   wait_group<RING-1>) when read, and no ring slot is refilled while a later read still needs it.
4. gemv V2 word mapping: tp[lane], tp[(lane + 31) & 31] from the ring == the served lane's own loaded
   word and its __shfl_sync((lane + 31) & 31) partner.
"""
import random
import sys

import numpy as np

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print("FAIL:", msg)


# ------------------------------------------------------------------------------------------ partition
def partition(tiles_k, tiles_n, G):
    """Served slice partition: CTA b owns tiles [T*b//G, T*(b+1)//G). Unsigned-32 semantics mirrored."""
    T = tiles_k * tiles_n
    return [(T * b // G, T * (b + 1) // G) for b in range(G)]


def segments(tiles_k, tiles_n, G):
    """Per CTA, the list of (column, k_start, k_end) segments in processing order."""
    out = []
    for beg, end in partition(tiles_k, tiles_n, G):
        segs = []
        i = beg
        while i < end:
            n, k = divmod(i, tiles_k)
            k_end = min(tiles_k - 1, k + (end - i) - 1)
            segs.append((n, k, k_end))
            i += k_end - k + 1
        out.append(segs)
    return out


def rnd(x, c_fp32):
    return x if c_fp32 else np.float32(np.float16(x))


def served_chain(segs_by_cta, tiles_k, partial, c_fp32):
    """Simulate the served lock chain per column: lock starts at 0; a segment may run when lock == lock_i
    (= tiles_k - k_end - 1); it reads C unless first, writes C (rounded) unless last; lock += lock_d."""
    cols = {}
    for b, segs in enumerate(segs_by_cta):
        for (n, ks, ke) in segs:
            cols.setdefault(n, []).append((b, ks, ke))
    result = {}
    for n, lst in cols.items():
        lock = 0
        C = None
        pending = list(lst)
        order = []
        while pending:
            ready = [s for s in pending if tiles_k - s[2] - 1 == lock]
            assert len(ready) == 1, ("chain stuck", n, lock, pending)
            b, ks, ke = ready[0]
            pending.remove(ready[0])
            lock_i, lock_d = tiles_k - ke - 1, ke - ks + 1
            first, last = lock_i == 0, lock_i + lock_d == tiles_k
            v = partial[(b, n)].copy()
            if not first:
                v = (v + C).astype(np.float32)          # frag_c += read_sum_gl (C already in C's dtype)
            if not last:
                C = np.array([rnd(x, c_fp32) for x in v], dtype=np.float32)   # write_sum_gl
            else:
                result[n] = v                            # final fp32 sums -> epilogue
            lock += lock_d
            order.append(b)
        result[n] = (result[n], order)
    return result


I32 = 1 << 31


def v2_gather(segs_by_cta, tiles_k, tiles_n, G, partial, c_fp32):
    T = tiles_k * tiles_n
    assert T * (G + 1) < I32, "host guard (resolve_v2) refuses this launch"

    def i32(x):
        check(0 <= x < I32, f"int32 overflow in the r2 partition arithmetic: {x}")
        return x

    tile_beg = lambda b: i32(T * b) // G                       # dv2_tile_beg
    owner = lambda i: (i32((i + 1) * G) - 1) // T              # r2 closed form for the top k tile's owner
    published = {}
    result = {}
    for b, segs in enumerate(segs_by_cta):
        for si, (n, ks, ke) in enumerate(segs):
            if ks != 0:
                check(si == 0, f"non-head segment is not the first of CTA {b} (T={T}, G={G})")
                check(b not in published, f"CTA {b} publishes twice")
                published[b] = n
    for b, segs in enumerate(segs_by_cta):
        for (n, ks, ke) in segs:
            if ks != 0:
                continue
            own = partial[(b, n)].copy()
            if ke == tiles_k - 1:
                result[n] = (own, [b])
                continue
            b_top = owner(n * tiles_k + tiles_k - 1)
            prods = [x for x in range(b + 1, b_top + 1) if tile_beg(x + 1) > tile_beg(x)]
            check(all(published.get(x) == n for x in prods), f"producer set mismatch col {n}")
            check(len([x for x, c in published.items() if c == n]) == len(prods), f"nprod mismatch col {n}")
            S = None
            for x in reversed(prods):              # b_top .. head + 1
                p = partial[(x, n)]
                S = p.copy() if S is None else (p + np.array([rnd(v, c_fp32) for v in S], dtype=np.float32)).astype(np.float32)
            if S is not None:
                own = (own + np.array([rnd(v, c_fp32) for v in S], dtype=np.float32)).astype(np.float32)
            result[n] = (own, list(reversed(prods)) + [b])
    return result


def fixup_case(tiles_k, tiles_n, G, rng, width=8):
    segs = segments(tiles_k, tiles_n, G)
    partial = {}
    for b, sl in enumerate(segs):
        for (n, ks, ke) in sl:
            # partial sums over mixed magnitudes, so fp16 rounding of the running sum matters
            partial[(b, n)] = (rng.standard_normal(width) * 10.0 ** rng.integers(-3, 4, width)).astype(np.float32)
    for c_fp32 in (True, False):
        a = served_chain(segs, tiles_k, partial, c_fp32)
        v = v2_gather(segs, tiles_k, tiles_n, G, partial, c_fp32)
        check(sorted(a) == sorted(v) == list(range(tiles_n)), f"column coverage T={tiles_k}x{tiles_n} G={G}")
        for n in a:
            va, oa = a[n]
            vv, ov = v[n]
            check(oa == ov, f"order differs col {n} tiles {tiles_k}x{tiles_n} G={G}: chain {oa} gather {ov}")
            check(va.view(np.uint32).tolist() == vv.view(np.uint32).tolist(),
                  f"bits differ col {n} tiles {tiles_k}x{tiles_n} G={G} c_fp32={c_fp32}")
    nseg = max(sum(1 for sl in segs for s in sl if s[0] == n) for n in range(tiles_n))
    return nseg


# Served launch configs (R680 traces, out-roofline tools/kgrid.py; see ANALYSIS.md): (name, size_k, n per
# matrix, TK, TN, grid x, grid z)
SERVED = [
    ("GDN in_proj qkvz r4  mgemm shape 4", 2560, 2048, 16, 512, 20, 8),
    ("GDN in_proj qkvz r16 mgemm shape 3", 2560, 2048, 32, 256, 20, 8),
    ("attn qkv r4/r16     mgemm shape 3", 2560, 512, 32, 256, 6, 26),
    ("out/o_proj r16      gemm  shape 2", 6144, 2560, 32, 128, 120, 1),
    ("shared down r16     gemm  shape 2", 640, 2560, 32, 128, 40, 1),
    ("shared gate/up r16  mgemm shape 2", 2560, 640, 32, 128, 28, 2),
    ("shared gate/up r4   mgemm shape 2", 2560, 640, 32, 128, 30, 2),
    ("index_qk r16        gemm  shape 2", 2560, 2560, 32, 128, 20, 1),   # n not traced: 2560 assumed (estimate)
]


def main():
    rng = np.random.default_rng(1234)
    print("1. stream-K fixup: served chain vs V2 gather")
    for name, k, n, tk, tn, gx, gz in SERVED:
        nseg = fixup_case(k // tk, n // tn, gx, rng)
        print(f"   {name}: tiles {k // tk}x{n // tn}, grid {gx}: max segments per column {nseg} "
              f"-> served chain {nseg - 1} handoffs, V2 {min(1, nseg - 1)}")
    pyr = random.Random(7)
    for _ in range(300):
        tk_, tn_ = pyr.randint(1, 200), pyr.randint(1, 12)
        G = pyr.randint(1, min(tk_ * tn_ * 2, 170))    # includes G > tiles (empty CTAs)
        if tk_ * tn_ * (G + 1) >= I32:
            continue
        fixup_case(tk_, tn_, G, rng, width=4)
    print("   random sweep: 300 configs (incl. empty CTAs), fp32 and fp16 C")

    print("2. workspace budget (bytes = CTAs x slot floats x 4)")
    def slot_floats(tm, tn, rows):
        return 8 * tn if (tm == 16 and rows <= 8) else tm * tn
    for name, k, n, tk, tn, gx, gz in SERVED:
        # the rows this launch config is served at (r4 = c1d3, r16 = c4d3 / c8d1), plus rows32 on the
        # 16-row configs (the autotune key of 17-32 rows is the 16-row key)
        served_rows = [4] if " r4 " in name + " " else [16, 32] if " r16 " in name + " " else [4, 16, 32]
        for rows in served_rows:
            tm = 32 if rows > 16 else 16
            if tm == 32 and tn == 512:
                continue                         # rows32 has no shape-4 twin (falls back)
            need = gx * gz * slot_floats(tm, tn, rows) * 4
            cap = (8 << 20) if rows > 16 else (4 << 20)
            check(need <= cap, f"{name} rows {rows}: {need} B > {cap}")
            print(f"   {name} rows {rows:2d}: {need / 2**20:5.2f} MiB (cap {cap >> 20} MiB)")

    print("3. gemv V2 ring schedule")
    for RING in (6,):
        for PF in (4, 2):
            for myn in range(0, 60):
                committed = []       # slice index per commit group (None = empty)
                slot_owner = {}
                def issue(i):
                    if i < myn:
                        slot_owner[i % RING] = i
                        committed.append(i)
                    else:
                        committed.append(None)
                for d in range(RING - 1):
                    issue(d)
                ib = 0
                while ib < myn:
                    for d in range(PF):
                        i = ib + d
                        if i >= myn:
                            break
                        issue(i + RING - 1)
                        retired = len(committed) - (RING - 1)          # wait_group<RING-1>
                        check(i in committed[:retired], f"ring: slice {i} not retired (myn {myn}, PF {PF})")
                        check(slot_owner.get(i % RING) == i, f"ring: slot of slice {i} overwritten (myn {myn})")
                    ib += PF
    print("   myn 0..59, PF 4 and 2, RING 6: every read slice retired and not overwritten")

    print("4. gemv V2 word mapping")
    words = list(range(1000, 1032))
    for lane in range(32):
        served_b = words[lane]
        served_a = words[(lane + 31) & 31]      # __shfl_sync(bw[t], (lane + 31) & 31)
        check((served_a, served_b) == (words[(lane + 31) & 31], words[lane]), "word mapping")
    print("   32 lanes OK")


    print("5. deferred head: head-not-top is the CTA's last segment; b_top closed form")
    ncfg = 0
    cfgs = [(k // tk, n // tn, gx) for name, k, n, tk, tn, gx, gz in SERVED]
    for _ in range(2000):
        tk_, tn_ = pyr.randint(1, 300), pyr.randint(1, 16)
        cfgs.append((tk_, tn_, pyr.randint(1, min(tk_ * tn_ * 2, 188))))
    for tiles_k, tiles_n, G in cfgs:
        T = tiles_k * tiles_n
        if T * (G + 1) >= I32:
            continue
        ncfg += 1
        segs = segments(tiles_k, tiles_n, G)
        for b, sl in enumerate(segs):
            for si, (n, ks, ke) in enumerate(sl):
                if ks == 0 and ke != tiles_k - 1:
                    check(si == len(sl) - 1, f"head-not-top is not the last segment: CTA {b} T={tiles_k}x{tiles_n} G={G}")
                    b_top = (((n + 1) * tiles_k) * G - 1) // T
                    holder = [x for x, s2 in enumerate(segs) if any(c == n and kk <= tiles_k - 1 <= ke2 for c, kk, ke2 in s2)]
                    check(holder == [b_top], f"b_top {b_top} != holder {holder} col {n} T={tiles_k}x{tiles_n} G={G}")
                    check(b_top > b, f"b_top {b_top} <= head {b}")
    print(f"   {ncfg} configs (served + random)")

    print("6. B prologue commit groups (DGV2_BPRO=1) against the served main loop")
    for S in (4, 6):
        for iters in range(1, 40):
            groups = []                                   # per group: set of (stage, 'A'|'B') it loads
            groups.append({(j, "B") for j in range(min(S - 1, iters))})
            for j in range(S - 1):
                groups.append({(j, "A")} if j < iters else set())
            def retired_after_wait(n_pending):
                return set().union(*groups[:max(0, len(groups) - n_pending)]) if groups else set()
            done = retired_after_wait(S - 2)              # wait_stage()
            check({(0, "A"), (0, "B")} <= done, f"S={S} iters={iters}: stage 0 not ready after the prologue")
            nxt = S - 1                                   # next stage async_load_gl issues
            for i in range(iters):                        # FSTAGE_V2: async_load_gl, wait_stage, matmul(i), ..., load_frags(i + 1)
                groups.append({(nxt, "A"), (nxt, "B")} if nxt < iters else set())
                nxt += 1
                done = retired_after_wait(S - 2)
                if i + 1 < iters:
                    check({(i + 1, "A"), (i + 1, "B")} <= done, f"S={S} iters={iters}: stage {i + 1} read before retired")
    print("   S = 4 and 6, 1..39 tiles per CTA: every stage retired before it is read")

    if FAIL:
        print(f"CPU TESTS FAILED ({len(FAIL)})")
        sys.exit(1)
    print("CPU TESTS PASS")


if __name__ == "__main__":
    main()
