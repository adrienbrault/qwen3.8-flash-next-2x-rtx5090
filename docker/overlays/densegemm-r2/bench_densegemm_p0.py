#!/usr/bin/env python3
"""densegemm r2 P0: microseconds per call of every dense K=4 projection of the served model, served kernels vs
the V2 kernels, at the served decode row counts, in-process, one card, real weights.

Arms (ext.dense_v2_set_mode, the same switch EXL3_DENSE_V2 / EXL3_DENSE_ROWS32 set at the first call):
  m0 = served kernels             m1 = V2 gemm + mgemm + gemv (+ rows32)
  m2 = V2 gemm + mgemm only (+ rows32)       m3 = V2 gemv only
  m4 = EXL3_DENSE_ROWS32 only (EXL3_DENSE_V2 off): the one-pass TILESIZE_M = 32 twins at rows 17-32, served
       kernels at <= 16 rows. This is piece 4a on its own, the arm rows32-r3 (c6-c8 at depth 2 = 18-24 rows) needs.
rows32 is ON in m1/m2 for rows > 16 (it is independent of the mode, and the only change at those rows).
Rows: 4 (c1d3), 8, 16 (c4d3 / c8d1), 18 / 21 / 24 (c6d2 / c7d2 / c8d2 under rows32-r3), 32.

Projections (calls per decode step, R680 trace, out-roofline tools/kgrid.py):
  gdn.in_proj    sliced mgemm qkv + z  36     critical path
  attn.qkv       sliced mgemm q, k, v  12     critical path
  gdn.out_proj   gemm                  36     critical path
  attn.o_proj    gemm                  12     critical path
  attn.index_qk  gemm                  12     critical path
  shared.gate_up mgemm, 2 matrices     48     overlap stream (mostly hidden, R703b)
  shared.down    gemm                  48     overlap stream
Each cell rotates over the real weights of every layer of that projection (>= 2x the 96 MiB L2 for every
family except index_qk, whose 12 layers are listed with their byte total), so the timings are DRAM timings.

Fair rotation: the arms of a cell are timed in R rounds, the arm order rotating each round (m0 m1 m2 m3,
m1 m2 m3 m0, ...); a cell's number per arm is the median of its per-round medians (d0 Timer: 200 calls per
round after 20 warm-up, GPU queue pre-filled behind a sleep so host launch latency is excluded).

Output: one [p0] line per cell and arm, a per-projection table, the step totals
  crit_us(arm, rows) = sum over critical-path projections of calls/step x us/call
and a RECOMMEND block: per row count, the arm with the lowest critical-path total; the recommended flag
setting is the arm that wins at rows 16 while costing at most 2 % of a c1d3 step at rows 4 (the step time is
an argument, default 13.0 ms = the c1d3 d0 step of R701, an estimate you should replace with the gate's own
OFF number). --json writes everything.

  python3 bench_densegemm_p0.py --model /models/<ckpt> --json /results/p0.json [--rounds 5]

r2 adds the P1 arm gate the r2 gate script reads (last lines):
  P0-CONTROL attn.index_qk r16 m2 <us> vs m0 <us> (<delta>): the no-fixup cell; a twin that still runs out of local
             memory shows here first (R710: 13.9 -> 36.7 us)
  P0-CALL <proj> r<rows> m2 <us> m0 <us> <pct> ok|LOSS   per critical-path gemm/mgemm cell at rows 4 and 16
  P0-ARM GM|ON ELIGIBLE|NOT (<reason>)
  P0-ARMS <GM ON | GM | none>
  P0-ROWS32 <proj> r<rows> m4 <us> m0 <us> <pct> ok|LOSS   per engaged cell at rows 18, 21, 24 (vs served two-pass)
  P0-ROWS32-VERDICT READY|NOT (<reason>)   READY: no engaged critical-path cell at 18 / 21 / 24 rows more than 3 %
             slower than served two-pass, and the critical-path total lower at each of those rows
GM (m2, gemm + mgemm twins only) is eligible when (a) no engaged critical-path gemm/mgemm cell at rows 4 or 16 is more
than 3 % slower than served, (b) the critical-path total at rows 16 is lower than served, and (c) the rows-4 total
passes the c1 bound (RECOMMEND rule). ON (m1, every twin) additionally needs GM eligible and its own (b) and (c).
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "d0"))

P = "model.language_model.layers"
GDN = [i for i in range(48) if i % 4 != 3]
ATT = [i for i in range(48) if i % 4 == 3]
LAYERS = list(range(48))
CRIT = {"gdn.in_proj": 36, "attn.qkv": 12, "gdn.out_proj": 36, "attn.o_proj": 12, "attn.index_qk": 12}
SIDE = {"shared.gate_up": 48, "shared.down": 48}
ARMS = {0: (0, 0), 1: (1, 1), 2: (2, 1), 3: (3, 0), 4: (0, 1)}
ROWS = [4, 8, 16, 18, 21, 24, 32]


def log(*a):
    print("[p0]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--step-ms-c1", type=float, default=13.0,
                    help="c1d3 step time for the 2 %% bound (estimate unless taken from the gate's OFF arm)")
    a = ap.parse_args()
    rows_list = [int(r) for r in a.rows.split(",")]

    import torch
    import exllamav3_ext as ext
    from d0_microbench import Checkpoint, Timer

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    assert ext.dense_v2_revision == 2
    ck = Checkpoint(a.model, dev)
    keys = set(ck.map)
    timer = Timer(torch, a.reps, a.warm)

    def lin(prefix):
        if f"{prefix}.trellis" not in keys or f"{prefix}.mul1" not in keys:
            return None
        L = ck.linear(prefix)
        return L if L[0].shape[-1] // 16 == 4 else None

    def nbytes(L):
        return sum(t.numel() * t.element_size() for t in L)

    # ---- builders: each returns a list of callables, one per rotation copy (layer)
    def b_gemm(prefixes, rows):
        mats = [m for m in (lin(p) for p in prefixes) if m is not None]
        if not mats:
            return None, 0
        k, n = mats[0][0].shape[0] * 16, mats[0][0].shape[1] * 16
        x = (torch.randn((rows, k), device=dev) * 0.5).half()
        y = torch.empty((rows, n), dtype=torch.half, device=dev)
        xh = torch.empty_like(x)
        fns = [(lambda tr=tr, suh=suh, svh=svh: ext.exl3_gemm(x, tr, y, suh, xh, svh, -1, False, True, 0))
               for tr, suh, svh in mats]
        return fns, sum(nbytes(m) for m in mats)

    def b_sliced(groups, rows):
        fns, tot = [], 0
        for prefixes in groups:
            srcs = [lin(p) for p in prefixes]
            if any(s is None for s in srcs):
                continue
            tot += sum(nbytes(s) for s in srcs)
            K = 4
            k = srcs[0][0].shape[0] * 16
            widths = [s[0].shape[1] * 16 for s in srcs]
            width = math.gcd(*widths)
            x3 = (torch.randn((1, rows, k), device=dev) * 0.5).half()
            xh = torch.empty((len(srcs), rows, k), dtype=torch.half, device=dev)
            outs = [torch.empty((rows, w), dtype=torch.float, device=dev) for w in widths]
            tp, sp, tg, off, st = [], [], [], [], []
            for i, (tr, suh, svh) in enumerate(srcs):
                n_i = tr.shape[1] * 16
                for n0 in range(0, n_i, width):
                    tp.append(tr.data_ptr() + (n0 // 16) * 16 * K * tr.element_size())
                    sp.append(svh.data_ptr() + n0 * svh.element_size())
                    tg.append(i); off.append(n0); st.append(n_i)
            S = len(tg)
            T = lambda v, dt: torch.tensor(v, dtype=dt, device=dev)
            args = (x3, T(tp, torch.long), torch.zeros((S, 1, width), dtype=torch.float, device=dev).expand(S, rows, width),
                    T([s[1].data_ptr() for s in srcs], torch.long), xh, T(sp, torch.long), None, None, K, -1, False,
                    True, -1, -1, 0, 1, T([width] * S, torch.int32),
                    T([outs[tg[j]].data_ptr() + off[j] * 4 for j in range(S)], torch.long), T(st, torch.int32),
                    T(tg, torch.int32), len(srcs))
            fns.append(lambda args=args, keep=(srcs, outs): ext.exl3_mgemm(*args))
        return (fns or None), tot

    def b_gate_up(prefixes, rows):
        fns, tot = [], 0
        for pre in prefixes:
            g, u = lin(f"{pre}.gate_proj"), lin(f"{pre}.up_proj")
            if g is None or u is None:
                continue
            tot += nbytes(g) + nbytes(u)
            k, n = g[0].shape[0] * 16, g[0].shape[1] * 16
            x3 = (torch.randn((1, rows, k), device=dev) * 0.5).half()
            T = lambda v: torch.tensor(v, dtype=torch.long, device=dev)
            c = torch.empty((2, rows, n), dtype=torch.half, device=dev)
            xh = torch.empty((2, rows, k), dtype=torch.half, device=dev)
            args = (x3, T([g[0].data_ptr(), u[0].data_ptr()]), c, T([g[1].data_ptr(), u[1].data_ptr()]), xh,
                    T([g[2].data_ptr(), u[2].data_ptr()]), None, None, 4, -1, False, True, -1, -1, 0)
            fns.append(lambda args=args, keep=(g, u): ext.exl3_mgemm(*args))
        return (fns or None), tot

    SA = f"{P}.{{}}.self_attn"
    builders = {
        "gdn.in_proj": lambda r: b_sliced([[f"{P}.{i}.linear_attn.in_proj_qkv", f"{P}.{i}.linear_attn.in_proj_z"]
                                          for i in GDN], r),
        "attn.qkv": lambda r: b_sliced([[f"{SA.format(i)}.{m}" for m in ("q_proj", "k_proj", "v_proj")] for i in ATT], r),
        "gdn.out_proj": lambda r: b_gemm([f"{P}.{i}.linear_attn.out_proj" for i in GDN], r),
        "attn.o_proj": lambda r: b_gemm([f"{SA.format(i)}.o_proj" for i in ATT], r),
        "attn.index_qk": lambda r: b_gemm([f"{SA.format(i)}.indexer.index_qk_proj" for i in ATT], r),
        "shared.gate_up": lambda r: b_gate_up([f"{P}.{i}.mlp.shared_expert" for i in LAYERS], r),
        "shared.down": lambda r: b_gemm([f"{P}.{i}.mlp.shared_expert.down_proj" for i in LAYERS], r),
    }

    res = {}   # (proj, rows) -> {arm: {us, us_batch, rounds, engaged}}
    for proj, build in builders.items():
        for rows in rows_list:
            fns, tot = build(rows)
            if fns is None:
                log(f"{proj} r{rows}: no K4 mul1 weights, skipped")
                continue
            # warm the autotune cache on the served arm (V2 dispatch only engages on a cache hit)
            ext.dense_v2_set_mode(0, 0)
            for f in fns:
                f()
            torch.cuda.synchronize()
            per = {arm: [] for arm in ARMS}
            batch = {arm: [] for arm in ARMS}
            engaged = {}
            order = list(ARMS)
            for rnd in range(a.rounds):
                rot = order[rnd % len(order):] + order[:rnd % len(order)]
                for arm in rot:
                    ext.dense_v2_set_mode(*ARMS[arm])
                    c0 = tuple(ext.dense_v2_launch_counts())
                    r = timer.run(fns)
                    c1 = tuple(ext.dense_v2_launch_counts())
                    per[arm].append(r["us_each"])
                    batch[arm].append(r["us_batch"])
                    engaged[arm] = (c1[0] - c0[0]) + (c1[1] - c0[1]) > 0
            ext.dense_v2_set_mode(0, 0)
            cell = {}
            for arm in ARMS:
                us = statistics.median(per[arm])
                cell[arm] = dict(us=us, us_batch=statistics.median(batch[arm]), rounds=per[arm], engaged=engaged[arm],
                                 spread=max(per[arm]) - min(per[arm]))
            res[(proj, rows)] = dict(arms=cell, bytes_all_layers=tot, copies=len(fns))
            b0 = cell[0]["us"]
            log(f"{proj:15s} r{rows:2d} copies {len(fns):2d} ({tot / 2**20:6.1f} MiB)  "
                + "  ".join(f"m{arm} {cell[arm]['us']:7.2f}us{'' if arm == 0 or cell[arm]['engaged'] else '(served)'}"
                            f" {100 * (cell[arm]['us'] / b0 - 1):+5.1f}%" for arm in ARMS)
                + f"  spread m0 {cell[0]['spread']:.2f}us")
            del fns
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ---- step totals and recommendation
    def total(arm, rows, table):
        s = 0.0
        for proj, calls in table.items():
            c = res.get((proj, rows))
            if c is None:
                return None
            s += calls * c["arms"][arm]["us"]
        return s

    summary = {"rows": {}, "recommend": None, "step_ms_c1_assumed": a.step_ms_c1}
    print("\nRECOMMEND (critical-path us per step = sum calls/step x us/call; shared expert listed apart)")
    print(f"{'rows':>4} " + " ".join(f"{'m' + str(arm) + ' crit':>12}" for arm in ARMS) + "  best   "
          + " ".join(f"{'m' + str(arm) + ' side':>12}" for arm in ARMS))
    for rows in rows_list:
        crit = {arm: total(arm, rows, CRIT) for arm in ARMS}
        side = {arm: total(arm, rows, SIDE) for arm in ARMS}
        if any(v is None for v in crit.values()):
            continue
        best = min(crit, key=crit.get)
        summary["rows"][rows] = dict(crit=crit, side=side, best=best)
        print(f"{rows:4d} " + " ".join(f"{crit[arm]:12.1f}" for arm in ARMS) + f"  m{best}    "
              + " ".join(f"{side[arm]:12.1f}" if side[arm] is not None else f"{'-':>12}" for arm in ARMS))
    if 4 in summary["rows"] and 16 in summary["rows"]:
        r4, r16 = summary["rows"][4]["crit"], summary["rows"][16]["crit"]
        bound_us = 0.02 * a.step_ms_c1 * 1000
        cands = []
        for arm in (1, 2, 3):
            gain16 = r16[0] - r16[arm]
            cost4 = r4[arm] - r4[0]
            ok = gain16 > 0 and (cost4 <= 0 or (cost4 <= bound_us and gain16 > cost4))
            cands.append((arm, gain16, -cost4, ok))
            print(f"m{arm}: rows 16 {gain16:+.1f} us/step saved, rows 4 {-cost4:+.1f} us/step saved -> "
                  f"{'eligible' if ok else 'not eligible'}")
        ok = [c for c in cands if c[3]]
        if ok:
            arm = max(ok, key=lambda c: (c[1] + max(c[2], 0.0), c[1]))[0]
            m, r32 = ARMS[arm]
            summary["recommend"] = dict(arm=arm, EXL3_DENSE_V2=m, EXL3_DENSE_ROWS32=r32)
            print(f"RECOMMEND EXL3_DENSE_V2={m} EXL3_DENSE_ROWS32={r32} (arm m{arm})")
        else:
            print("RECOMMEND none (no arm saves time at rows 16 within the c1d3 bound)")
    # ---- r2: P1 arm gate (see the docstring)
    arms_p1 = []
    if 4 in summary["rows"] and 16 in summary["rows"]:
        ctl = res.get(("attn.index_qk", 16))
        if ctl:
            c0, c2 = ctl["arms"][0]["us"], ctl["arms"][2]["us"]
            print(f"P0-CONTROL attn.index_qk r16 m2 {c2:.2f} us vs m0 {c0:.2f} us ({c2 - c0:+.2f} us)")
        losses = []
        for proj in CRIT:
            for rows in (4, 16):
                c = res.get((proj, rows))
                if c is None or not c["arms"][2]["engaged"]:
                    continue
                u0, u2 = c["arms"][0]["us"], c["arms"][2]["us"]
                pct = 100 * (u2 / u0 - 1)
                bad = pct > 3.0
                if bad:
                    losses.append(f"{proj} r{rows} {pct:+.1f} %")
                print(f"P0-CALL {proj} r{rows} m2 {u2:.2f} m0 {u0:.2f} {pct:+.1f} % {'LOSS' if bad else 'ok'}")
        r4, r16 = summary["rows"][4]["crit"], summary["rows"][16]["crit"]
        bound_us = 0.02 * a.step_ms_c1 * 1000

        def totals_ok(arm):
            gain16 = r16[0] - r16[arm]
            cost4 = r4[arm] - r4[0]
            if gain16 <= 0:
                return False, f"rows 16 total {-gain16:+.1f} us/step vs served"
            if cost4 > 0 and (cost4 > bound_us or gain16 <= cost4):
                return False, f"rows 4 total {cost4:+.1f} us/step outside the c1 bound"
            return True, f"rows 16 {-gain16:+.1f} us/step, rows 4 {cost4:+.1f} us/step"

        ok2, why2 = totals_ok(2)
        gm = ok2 and not losses
        print(f"P0-ARM GM {'ELIGIBLE' if gm else 'NOT'} ({why2}{'; per-call LOSS: ' + ', '.join(losses) if losses else ''})")
        ok1, why1 = totals_ok(1)
        on = gm and ok1
        print(f"P0-ARM ON {'ELIGIBLE' if on else 'NOT'} ({why1}{'' if gm else '; GM not eligible'})")
        arms_p1 = (["GM"] if gm else []) + (["ON"] if on else [])
    summary["arms_p1"] = arms_p1
    print("P0-ARMS " + (" ".join(arms_p1) if arms_p1 else "none"))
    # ---- r2: ROWS32 (piece 4a) on its own, m4 = EXL3_DENSE_ROWS32=1 with EXL3_DENSE_V2 off, vs served two-pass
    r32rows = [r for r in (18, 21, 24) if r in summary["rows"]]
    r32_losses, r32_notes, r32_eng = [], [], 0
    for rows in [r for r in rows_list if r > 16]:
        for proj in list(CRIT) + list(SIDE):
            c = res.get((proj, rows))
            if c is None or not c["arms"][4]["engaged"]:
                continue
            u0, u4 = c["arms"][0]["us"], c["arms"][4]["us"]
            pct = 100 * (u4 / u0 - 1)
            bad = pct > 3.0
            if rows in r32rows and proj in CRIT:
                r32_eng += 1
                if bad:
                    r32_losses.append(f"{proj} r{rows} {pct:+.1f} %")
            print(f"P0-ROWS32 {proj} r{rows} m4 {u4:.2f} m0 {u0:.2f} {pct:+.1f} % {'LOSS' if bad else 'ok'}")
    for rows in r32rows:
        c = summary["rows"][rows]["crit"]
        r32_notes.append(f"r{rows} crit {c[4] - c[0]:+.1f} us/step")
        if c[4] >= c[0]:
            r32_losses.append(f"r{rows} crit total {c[4] - c[0]:+.1f} us/step")
    r32_ready = bool(r32rows) and r32_eng > 0 and not r32_losses
    summary["rows32_ready"] = r32_ready
    print(f"P0-ROWS32-VERDICT {'READY' if r32_ready else 'NOT'} ("
          + ("; ".join(r32_notes) if r32_notes else "no rows 18-24 measured")
          + (f"; engaged cells {r32_eng}" if r32_eng else "; NO engaged rows32 cell")
          + (f"; LOSS: {', '.join(r32_losses)}" if r32_losses else "") + ")")
    if a.json:
        out = dict(cells={f"{p}/r{r}": v for (p, r), v in res.items()}, summary=summary,
                   calls_per_step=dict(CRIT, **SIDE))
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("P0 DONE", flush=True)


if __name__ == "__main__":
    main()
