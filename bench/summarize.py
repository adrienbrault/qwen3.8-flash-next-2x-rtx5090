#!/usr/bin/env python3
"""Per-arm summary of probe.py records: one row per tag/concurrency, from the JSONL and nothing else.

The probe's own stdout is a convenience; this reads the records so a table in the docs can be regenerated rather
than transcribed. It also reports what the probe's per-round summary cannot: the spread across requests in a round
and whether every request in the round finished at the forced length.

usage: summarize.py results/2026-09-16-r340-ci-depth/records.jsonl [more.jsonl ...]
"""
import json, statistics, sys

for path in sys.argv[1:]:
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("ok"):
            rows.append(r)
    if not rows:
        print(f"== {path}: no successful records")
        continue
    print(f"== {path}  ({len(rows)} successful requests)")
    keys = sorted({(r.get("tag", "?"), r.get("ctx_requested", 0), r["conc"], r.get("run")) for r in rows})
    print(f"{'tag':<22}{'ctx~':>8}{'c':>3}{'run':>4}{'recs':>5}{'n':>7}{'decode/stream':>15}{'aggregate':>11}"
          f"{'ttft':>8}{'spread':>9}  finish")
    for tag, ctx, conc, run in keys:
        grp = [r for r in rows if (r.get("tag", "?"), r.get("ctx_requested", 0), r["conc"], r.get("run")) == (tag, ctx, conc, run)]
        # A request that stopped for any reason other than the forced length is not a throughput sample: the loop
        # detector can end a long greedy run early, and including that row drags the median down.
        grp_rates = [r for r in grp if r.get("finish_reason") in (None, "length")] or grp
        n = [r.get("completion_tokens") or 0 for r in grp_rates]
        dec = [r.get("decode_tps") for r in grp_rates if r.get("decode_tps")]
        ttft = [r.get("ttft_s") for r in grp_rates if r.get("ttft_s")]
        wall = grp[0].get("round_wall_s") or 0
        agg = round(sum(n) / wall, 1) if wall else None
        spread = f"{min(dec):.1f}-{max(dec):.1f}" if dec else "-"
        fl = ",".join(sorted({r.get("finish_reason") or "?" for r in grp}))
        # recs is the group size. If it exceeds `conc`, two attempts wrote to the same file and the aggregate is
        # inflated by the sum of both -- which is exactly how a doubled c1 row read 410 t/s against a 209 t/s
        # decode rate. Aggregate is therefore only printed when the group is one round's worth of requests.
        if len(grp) != conc:
            agg = None
        print(f"{tag:<22}{ctx:>8}{conc:>3}{run:>4}{len(grp):>5}{int(statistics.median(n)):>7}"
              f"{statistics.median(dec) if dec else 0:>15.1f}{(f'{agg:.1f}' if agg is not None else 'AMBIG'):>11}"
              f"{statistics.median(ttft) if ttft else 0:>8.2f}{spread:>9}  {fl}")
    # Any request that did not reach the forced length is not a throughput sample.
    short = [r for r in rows if r.get("finish_reason") not in (None, "length")]
    if short:
        print(f"   NOTE: {len(short)} request(s) did not finish at the forced length "
              f"(e.g. finish_reason={short[0].get('finish_reason')}) -- exclude them from rate comparisons")
    acct = [r for r in rows if r.get("accounting_note")]
    if acct:
        print(f"   NOTE: {len(acct)} request(s) carry an accounting mismatch against the server's own count")
    print()
