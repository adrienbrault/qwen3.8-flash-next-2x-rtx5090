#!/usr/bin/env python3
"""r813_curve_compare.py -- the R813 decode curve (the served merge-tok-r1 daily) against R787a's (rebase-dev-r3), per
concurrency and prompt kind. REPORT ONLY: nothing is gated on it, and it never changes a unit's VERDICT. Stdlib only.

WHAT IT READS: <dir>/records.jsonl of each round (probes/fn_bench.py, one JSON line per request, tags
<ARM><boot>-c<conc>-<kind>, e.g. NEW1-c4-code), and, for the report-only cross-check, <dir>/container-<ARM><boot>.log.

METRICS (per '<conc>-<kind>' shape, both boots pooled):
  per_stream   median over requests of decode_tps = (tokens - 1) / (t_last - t_first)       } bench/plot.py decode_rates(),
  decode_agg   mean over rounds of the sum of the round's decode_tps                          } same definitions, so the
  wall_agg     mean over rounds of the round's tokens / round wall (secondary)                } table agrees with what the
  ttft         median over requests of the time to the first token                           } public figure prints
  ms_step      median over requests of 1000 * decode_window_s / (client_frames - 1)
  tok_step     mean over requests of completion_tokens / client_frames
  fn_bench's client_frames is the number of SSE frames with text = verify steps (REVIEW-R787 section 3: 759 frames =
  2,048 - 1,289 accepted in the container log; R787a code c1 343 steps -> 10.02 ms/step, 2.99 tokens/step, which these
  two formulas reproduce). The first frame is the prefill's token, so the decode window holds client_frames - 1 steps.
  boot spread  per shape, |decode_agg(boot1) - decode_agg(boot2)| / their mean: the within-round noise reference.

CROSS-CHECK (report only): per boot, tokens / step from the container log (parse_container.py, lines with gen >= the
forced length, sum(gen) / sum(gen - accepted); these include fn_bench's unrecorded warm-up round, so 4 x sum(conc) x 2
kinds = 288 lines per boot for c1..c8) against the client's sum(tokens) / sum(frames) over the recorded rounds. Agreement
within ~1 % says frames == verify steps held on this image too.

usage: r813_curve_compare.py --new DIR --ref DIR [--arm NEW] [--new-label R813] [--ref-label R787a] [--json OUT]
rc 0 = a table was printed (shapes missing on either side print n/a); rc 2 = a records.jsonl is missing or has no record
of the arm.
"""
import argparse, collections, json, os, statistics as st, sys


def load(path, arm):
    """(reqs[shape] = [record], rounds[(shape, tag, run)] = [record], boots[shape][tag] = [[record] per run])"""
    reqs, rounds = collections.defaultdict(list), collections.defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tag = r.get("tag") or ""
        if not r.get("ok") or tag[: len(arm)] != arm or len(tag) <= len(arm) or not tag[len(arm)].isdigit() \
                or "-" not in tag:
            continue
        shape = tag.split("-", 1)[1]
        reqs[shape].append(r)
        rounds[(shape, tag, r.get("run"))].append(r)
    return reqs, rounds


def metrics(reqs, rounds):
    out = {}
    for shape, rs in reqs.items():
        rr = {k: v for k, v in rounds.items() if k[0] == shape}
        dec = [r["decode_tps"] for r in rs if r.get("decode_tps")]
        ttft = [r["ttft_s"] for r in rs if r.get("ttft_s") is not None]
        ms = [1000.0 * r["decode_window_s"] / (r["client_frames"] - 1) for r in rs
              if (r.get("client_frames") or 0) >= 2 and r.get("decode_window_s")]
        tps = [r["completion_tokens"] / r["client_frames"] for r in rs
               if r.get("client_frames") and r.get("completion_tokens")]
        dagg = [sum(r["decode_tps"] for r in v if r.get("decode_tps")) for v in rr.values()]
        wagg = [sum(r["completion_tokens"] or 0 for r in v) / v[0]["round_wall_s"] for v in rr.values()
                if v[0].get("round_wall_s")]
        per_boot = collections.defaultdict(list)
        for (_, tag, _), v in rr.items():
            per_boot[tag.split("-", 1)[0]].append(sum(r["decode_tps"] for r in v if r.get("decode_tps")))
        pb = {b: st.mean(v) for b, v in sorted(per_boot.items())}
        spread = None
        if len(pb) >= 2:
            vals = list(pb.values())
            spread = (max(vals) - min(vals)) / st.mean(vals) if st.mean(vals) else None
        out[shape] = {
            "n": len(rs),
            "per_stream": st.median(dec) if dec else None,
            "decode_agg": st.mean(dagg) if dagg else None,
            "wall_agg": st.mean(wagg) if wagg else None,
            "ttft": st.median(ttft) if ttft else None,
            "ms_step": st.median(ms) if ms else None,
            "tok_step": st.mean(tps) if tps else None,
            "boots": pb,
            "boot_spread": spread,
        }
    return out


def ratio(a, b):
    return a / b if (a is not None and b) else None


def f(x, fmt):
    return format(x, fmt) if x is not None else "n/a"


def cross_check(d, arm, reqs, tokens):
    """Per boot: (lines, tau_container, tau_client) or a reason string. Report only."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import parse_container
    except Exception as e:  # the table does not need it
        return {"-": f"parse_container.py not importable ({e.__class__.__name__})"}
    client = collections.defaultdict(lambda: [0, 0])
    for rs in reqs.values():
        for r in rs:
            if r.get("client_frames") and r.get("completion_tokens"):
                b = r["tag"].split("-", 1)[0]
                client[b][0] += r["completion_tokens"]
                client[b][1] += r["client_frames"]
    out = {}
    for b in sorted(client):
        p = os.path.join(d, f"container-{b}.log")
        if not os.path.exists(p):
            out[b] = "no container log"
            continue
        lines = [x for x in parse_container.parse(p) if x["gen"] >= tokens]
        g = sum(x["gen"] for x in lines)
        s = sum(x["gen"] - x["acc"] for x in lines)
        tc = g / s if s else None
        tk = client[b][0] / client[b][1] if client[b][1] else None
        out[b] = (len(lines), tc, tk)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="the R813 results dir (records.jsonl, container-NEW<b>.log)")
    ap.add_argument("--ref", required=True, help="the R787a results dir")
    ap.add_argument("--arm", default="NEW")
    ap.add_argument("--new-label", default="R813")
    ap.add_argument("--ref-label", default="R787a")
    ap.add_argument("--concs", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    concs = [int(c) for c in a.concs.split(",")]
    got = {}
    for side, d in (("new", a.new), ("ref", a.ref)):
        p = os.path.join(d, "records.jsonl")
        if not os.path.exists(p):
            print(f"compare: {side} {p} missing: no table")
            return 2
        reqs, rounds = load(p, a.arm)
        if not reqs:
            print(f"compare: {side} {p} has no ok record of arm {a.arm}: no table")
            return 2
        got[side] = (reqs, metrics(reqs, rounds))
    N, R = got["new"][1], got["ref"][1]
    nl, rl = a.new_label, a.ref_label
    print(f"decode curve {nl} ({a.new}) vs {rl} ({a.ref}); arm {a.arm}; both boots pooled; ratios {nl}/{rl}")
    print("  per-stream = median decode_tps; aggregate = mean over rounds of the round's summed decode_tps; ms/step = median"
          " 1000*window/(frames-1); tok/step = mean tokens/frames; boot spread = |boot1-boot2|/mean of the aggregate")
    hdr = (f"  {'shape':10} {'per-stream ' + nl + ' / ' + rl + ', ratio':<34}   {'aggregate, ratio':<20}   "
           f"{'ms/step, ratio':<20}   {'tok/step, ratio':<18}   {'n':<7}   boot spread {nl} / {rl}")
    rows, summ = [], collections.defaultdict(list)
    for kind in ("code", "prose"):
        print(hdr)
        for c in concs:
            s = f"c{c}-{kind}"
            n, r = N.get(s), R.get(s)
            g = lambda m, k: (m or {}).get(k)
            row = {"shape": s, "conc": c, "kind": kind}
            for k in ("per_stream", "decode_agg", "ms_step", "tok_step", "ttft", "wall_agg"):
                row[k] = {"new": g(n, k), "ref": g(r, k), "ratio": ratio(g(n, k), g(r, k))}
                if row[k]["ratio"] is not None:
                    summ[(kind, k, "c1" if c == 1 else "c2-c8")].append(row[k]["ratio"])
            row["n"] = {"new": g(n, "n"), "ref": g(r, "n")}
            row["boot_spread"] = {"new": g(n, "boot_spread"), "ref": g(r, "boot_spread")}
            rows.append(row)
            ps, ag, ms, ts = row["per_stream"], row["decode_agg"], row["ms_step"], row["tok_step"]
            bs = row["boot_spread"]
            print(f"  {kind:5} c{c:<3} {f(ps['new'], '10.1f')} / {f(ps['ref'], '<10.1f')}     {f(ps['ratio'], '6.3f')}"
                  f"   {f(ag['new'], '5.0f')} / {f(ag['ref'], '<5.0f')} {f(ag['ratio'], '6.3f')}"
                  f"   {f(ms['new'], '5.2f')} / {f(ms['ref'], '<5.2f')} {f(ms['ratio'], '6.3f')}"
                  f"   {f(ts['new'], '4.2f')} / {f(ts['ref'], '<4.2f')} {f(ts['ratio'], '6.3f')}"
                  f"   {row['n']['new'] if row['n']['new'] is not None else '-':>3}/{row['n']['ref'] if row['n']['ref'] is not None else '-':<3}"
                  f"   {f(None if bs['new'] is None else 100 * bs['new'], '4.1f')}% / {f(None if bs['ref'] is None else 100 * bs['ref'], '4.1f')}%")
    print(f"  summary ({nl}/{rl}, median of the per-shape ratios). Cross-session: the boot-spread column is only the"
          " within-session noise; the cross-day reference on this instrument is R787a/R719b, ms/step 1.000-1.022 over the 16"
          " shapes (median ~1.015, REVIEW-R787 section 4), so a ratio inside that band is not separable from day-to-day drift:")
    for kind in ("code", "prose"):
        parts = []
        for k, lab in (("ms_step", "ms/step"), ("tok_step", "tok/step"), ("per_stream", "per-stream"),
                       ("decode_agg", "aggregate")):
            for grp in ("c1", "c2-c8"):
                v = summ.get((kind, k, grp))
                if v:
                    parts.append(f"{lab} {grp} {st.median(v):.3f}" + (f" [{min(v):.3f}..{max(v):.3f}]" if len(v) > 1 else ""))
        print(f"    {kind:5}: " + ("; ".join(parts) if parts else "no shape on both sides"))
    tokens = max((r.get("max_tokens") or 0) for rs in got["new"][0].values() for r in rs) or 1024
    xc = {}
    for side, d, lab in (("new", a.new, nl), ("ref", a.ref, rl)):
        xc[side] = cross_check(d, a.arm, got[side][0], tokens)
        for b, v in xc[side].items():
            if isinstance(v, str):
                print(f"  cross-check {lab} {b}: {v}")
            else:
                n_l, tc, tk = v
                d_ = f"{100 * (tc / tk - 1):+.2f} %" if (tc and tk) else "n/a"
                print(f"  cross-check {lab} {b}: container {n_l} lines with >= {tokens} tokens, tokens/step {f(tc, '.3f')}"
                      f" (incl. the warm-up round); client {f(tk, '.3f')} (recorded rounds); container vs client {d_}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"new": a.new, "ref": a.ref, "arm": a.arm, "rows": rows,
                       "cross_check": {s: {b: (v if isinstance(v, str) else list(v)) for b, v in x.items()}
                                       for s, x in xc.items()}}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
