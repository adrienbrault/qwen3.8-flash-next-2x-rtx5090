#!/usr/bin/env python3
"""Summarise an R725/R731-style `vllm bench serve` unit (stdlib only).

Output: one README-layout table per (dataset, pass), one row per concurrency; an A/B table per dataset; a diagnostics
table; MTP acceptance per Spec-Bench category and task group (c1 cells); FLAGS; and, when --decide is given, a final
`DECISION:` line under the rule that R731 pre-registered (see r731-std-bench.sh).

Inputs
  --runs       runs.tsv written by the unit: tag, [pass], dataset, conc, serial_lo, serial_hi, warmups, catfile, rc,
               [container]. `container` is the per-cell container log (R731 boots a fresh server for every cell, so
               each cell has its own log); without that column every run reads --container (R725).
  --results    directory holding <tag>.json (vllm bench serve --save-result --save-detailed) and, from R731 on,
               <tag>.samples.tsv (the shim's sample manifest: i, sha, prompt_len, output_len, in send order)
  --container  fallback container log (R725: one boot for the whole unit)
  --boots      boots.tsv (R731): one row per boot with the launcher md5, image, EXL3 env digest and the GPU clock
               offsets read at boot and after the cell
  --sb-data    the Spec-Bench question.jsonl: maps a manifest prompt hash to its category
  --sb-dir     R725 only: directory holding the Spec-Bench split files named in runs.tsv `catfile`
  --sb-out     the forced Spec-Bench output length (every Spec-Bench output must equal it)

Columns (headline, review-r725 §5 layout)
  Requests, Output tok/s, Req/s   vLLM's completed / output_throughput / request_throughput
  TTFT, TPOT p50 / p99 (ms)       vLLM's own percentiles from the result JSON
  Per-stream tok/s                1000 / TPOT p50
  E2E p50 (s)                     vLLM's p50 e2el
  tau                             tokens per verify step = gen / (gen - accepted), summed over the cell's measured
                                  container lines (warm-ups dropped). Equivalently 1 + accepted / steps, where
                                  steps = gen - accepted counts the prefill token as a step (the exact per-request
                                  count is gen - accepted - 1; the bias is ~1 % at 256 tokens). R725's review and
                                  FINDINGS used this definition.
  depth                           draft tokens proposed per verify step = proposed / (gen - accepted): the MTP depth in
                                  effect (the policy drops 3 -> 2 at c >= 5)
Diagnostics
  step p50                        vLLM's ITL p50: per SSE frame, i.e. per verify step, not per token
  tau client                      sum(output_len - 1) / sum(frames after the first), from the client's ITL arrays: an
                                  independent client-side check of tau
  acc/prop                        accepted / proposed (NOT comparable across levels: the depth changes)
  cached                          sum(cached prompt tokens) / sum(prompt tokens) over the measured lines, and the
                                  number of measured requests with any cached token
The R725 columns dec-agg, @full and occ are gone (review-r725 §3/§5.9: @full was mis-specified, all three are house
metrics).

Checks (FLAGS): no result / rc != 0; failed requests; completed != num_prompts; container line count != requests +
warm-ups (foreign traffic, or lines missing); client sum(output_lens) != container sum(gen); a Spec-Bench output_len !=
sb-out; manifests that differ between cells of one dataset (the sample is not the same at every level); boots that
differ in launcher md5 / image / EXL3 env / keys / policy / cache; clock offsets != the expected ones; cache share above
--cache-max; the c1 category mapping unverifiable.
"""
import argparse, csv, json, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_container import parse  # noqa: E402

MT_BENCH = ("writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities")


def g(d, *keys, default=None):
    """First present key: vLLM names the 50th percentile both p50_* and median_*, and fields moved across versions."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def requests_of(res):
    """Per-request rows from --save-detailed arrays; a request is ok when it has no error and a positive latency."""
    st = g(res, "start_times", default=[]) or []
    tt = g(res, "ttfts", default=[]) or []
    lat = g(res, "latencies", "e2els", default=[]) or []
    out = g(res, "output_lens", default=[]) or []
    inp = g(res, "input_lens", default=[]) or []
    itl = g(res, "itls", default=[]) or []
    err = g(res, "errors", default=[]) or []
    n = min(len(st), len(tt), len(lat), len(out))
    rows = []
    for i in range(n):
        ok = (not (err[i] if i < len(err) else "")) and lat[i] and lat[i] > 0 and tt[i] is not None
        rows.append(dict(start=st[i], ttft=tt[i] or 0.0, lat=lat[i] or 0.0, out=out[i] or 0,
                         inp=inp[i] if i < len(inp) else None, frames=len(itl[i] or []) if i < len(itl) else None,
                         ok=bool(ok)))
    return rows


def read_manifest(path):
    if not os.path.exists(path):
        return None
    return [(r["sha"], int(r["prompt_len"]), int(r["output_len"]))
            for r in csv.DictReader(open(path), delimiter="\t")]


def fmt(x, nd=1, w=0):
    if x is None:
        return "-".rjust(w)
    return f"{x:{w}.{nd}f}"


_parsed = {}


def container_recs(path):
    if path not in _parsed:
        _parsed[path] = sorted(parse(path), key=lambda r: r["serial"]) if path and os.path.exists(path) else []
    return _parsed[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--container", default=None)
    ap.add_argument("--boots", default=None)
    ap.add_argument("--sb-data", default=None)
    ap.add_argument("--sb-dir", default=None)
    ap.add_argument("--sb-out", type=int, default=256)
    ap.add_argument("--want-gpc", default="0 0", help="expected core (GPC) offsets per card, space-separated")
    ap.add_argument("--want-mem", default="4500 4500", help="expected memory offsets per card, space-separated")
    ap.add_argument("--cache-max", type=float, default=0.01, help="max cached share of prompt tokens per cell")
    ap.add_argument("--spread-max", type=float, default=3.0, help="max A/B spread on output tok/s, percent")
    ap.add_argument("--spread-exempt", type=lambda v: [int(x) for x in v.split(",") if x], default=[],
                    help="concurrency levels reported but not gated by --spread-max (c1: R592 boot-to-boot noise)")
    ap.add_argument("--decide", default=None, help="passes that must all be complete, e.g. 'A B'; prints DECISION")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rundir = os.path.dirname(os.path.abspath(a.runs))
    runs = list(csv.DictReader(open(a.runs), delimiter="\t"))
    sha2cat = {}
    if a.sb_data and os.path.exists(a.sb_data):
        import hashlib
        for ln in open(a.sb_data):
            if ln.strip():
                r = json.loads(ln)
                sha2cat.setdefault(hashlib.sha256(r["turns"][0].encode()).hexdigest()[:16], set()).add(r["category"])

    out_rows, flags, integrity, cache_bad, cat_tables = [], [], [], [], []
    manifests = {}
    for run in runs:
        tag, ds, conc = run["tag"], run["dataset"], int(run["conc"])
        ps = run.get("pass") or "-"
        lo, hi, w = int(run["serial_lo"]), int(run["serial_hi"]), int(run.get("warmups") or 0)
        cpath = run.get("container") or ""
        cpath = (cpath if os.path.isabs(cpath) else os.path.join(rundir, cpath)) if cpath not in ("", "-") else a.container
        row = dict(tag=tag, dataset=ds, conc=conc, pass_=ps, rc=run.get("rc"))
        out_rows.append(row)
        try:
            res = json.load(open(os.path.join(a.results, f"{tag}.json")))
        except Exception as e:
            integrity.append(f"{tag}: no result JSON ({e.__class__.__name__}); rc {run.get('rc')}")
            continue
        if str(run.get("rc")) not in ("0", "None", ""):
            integrity.append(f"{tag}: client rc {run.get('rc')}")
        rows = requests_of(res)
        ok = [r for r in rows if r["ok"]]
        recs = container_recs(cpath)
        mine = [r for r in recs if lo < r["serial"] <= hi]
        meas = mine[w:]
        gen = sum(r["gen"] for r in meas)
        acc = sum(r["acc"] for r in meas)
        prop = sum(r["prop"] for r in meas)
        steps = gen - acc
        frames = sum(r["frames"] for r in ok if r["frames"] is not None)
        ptok = sum(r["prompt"] for r in meas)
        ctok = sum(r["cached"] for r in meas)
        completed = g(res, "completed", default=len(ok))
        failed = g(res, "failed", default=len(rows) - len(ok))
        tpot50 = g(res, "p50_tpot_ms", "median_tpot_ms")
        row.update(
            n_prompts=g(res, "num_prompts"), completed=completed, failed=failed, duration=g(res, "duration"),
            out_tps=g(res, "output_throughput"), total_tps=g(res, "total_token_throughput"),
            req_s=g(res, "request_throughput"),
            ttft_p50=g(res, "p50_ttft_ms", "median_ttft_ms"), ttft_p99=g(res, "p99_ttft_ms"),
            tpot_p50=tpot50, tpot_p99=g(res, "p99_tpot_ms"),
            per_stream=(1000.0 / tpot50 if tpot50 else None),
            itl_p50=g(res, "p50_itl_ms", "median_itl_ms"),
            e2el_p50_s=(g(res, "p50_e2el_ms", "median_e2el_ms") or 0) / 1000.0 or None,
            mean_in=(statistics.mean(r["inp"] for r in ok if r["inp"] is not None) if ok else None),
            mean_out=(statistics.mean(r["out"] for r in ok) if ok else None),
            tau=(gen / steps if steps > 0 else None), depth=(prop / steps if steps > 0 else None),
            tau_client=(sum(r["out"] - 1 for r in ok if r["frames"]) / frames if frames else None),
            acc_prop=(acc / prop if prop else None), mtp_n=len(meas),
            prompt_tok=ptok, cached_tok=ctok, cached_share=(ctok / ptok if ptok else None),
            cached_n=sum(1 for r in meas if r["cached"] > 0),
            client_out=sum(r["out"] for r in ok), container_gen=gen,
        )
        # checks
        if failed:
            integrity.append(f"{tag}: {failed} failed requests")
        if row["n_prompts"] is not None and completed != row["n_prompts"]:
            integrity.append(f"{tag}: {completed} completed of {row['n_prompts']} prompts")
        if len(mine) != completed + failed + w:
            integrity.append(f"{tag}: {len(mine)} container lines in serials ({lo},{hi}] vs {completed}+{failed} "
                             f"requests + {w} warm-ups (foreign traffic, or lines missing)")
        if row["client_out"] != gen:
            d = (row["client_out"] - gen) / max(gen, 1) * 100
            integrity.append(f"{tag}: client sum(output_lens) {row['client_out']} != container sum(gen) {gen} "
                             f"({d:+.2f} %)")
        if ds == "specbench":
            bad = [r["out"] for r in ok if r["out"] != a.sb_out]
            if bad:
                integrity.append(f"{tag}: {len(bad)}/{len(ok)} outputs != {a.sb_out} tokens (length forcing did not "
                                 f"hold; e.g. {bad[:5]})")
        if row["cached_share"] is not None and row["cached_share"] > a.cache_max:
            cache_bad.append(f"{tag}: cached {row['cached_share'] * 100:.2f} % of prompt tokens "
                             f"({row['cached_n']} requests) > {a.cache_max * 100:.2f} %")
        man = read_manifest(os.path.join(a.results, f"{tag}.samples.tsv"))
        if man is not None:
            manifests.setdefault(ds, []).append((tag, man))
        # per-category acceptance at c1, where the container's serial order is the send order
        cats = None
        if conc == 1 and ds == "specbench" and man is not None and sha2cat:
            cs = [sha2cat.get(s, {"?"}) for s, _, _ in man]
            if any(len(c) != 1 for c in cs):
                flags.append(f"{tag}: {sum(len(c) != 1 for c in cs)} manifest prompts map to no / several categories")
            cats = [sorted(c)[0] for c in cs]
        elif (run.get("catfile") or "-") != "-" and a.sb_dir:  # R725: the c1 split in file order
            cats = [json.loads(l)["category"] for l in open(os.path.join(a.sb_dir, run["catfile"])) if l.strip()]
        if cats is not None:
            inp = [r["inp"] for r in rows]
            seq_ok = len(meas) == len(cats) and [r["prompt"] for r in meas] == inp
            if not seq_ok:
                flags.append(f"{tag}: category mapping UNVERIFIED ({len(meas)} lines vs {len(cats)} prompts; "
                             f"prompt-token sequence {'differs' if len(meas) == len(cats) else 'n/a'})")
            by = {}
            for cat, r in zip(cats, meas):
                b = by.setdefault(cat, [0, 0, 0, 0])
                b[0] += r["acc"]; b[1] += r["prop"]; b[2] += 1; b[3] += r["gen"]
            cat_tables.append((tag, seq_ok, by))

    # the same sample at every level: every cell of a dataset must carry the identical manifest
    for ds, ms in manifests.items():
        ref_tag, ref = ms[0]
        for tag, m in ms[1:]:
            if m != ref:
                same = len(set(x[0] for x in m) & set(x[0] for x in ref))
                integrity.append(f"{tag}: sample differs from {ref_tag} ({len(m)} vs {len(ref)} requests, "
                                 f"{same} prompts shared)")
    seen_ds = {r["dataset"] for r in out_rows}
    if a.decide:
        for ds in seen_ds:
            have = {t for t, _ in manifests.get(ds, [])}
            miss = [r["tag"] for r in out_rows if r["dataset"] == ds and r["tag"] not in have and "out_tps" in r]
            if miss:
                integrity.append(f"{ds}: no sample manifest for {', '.join(miss)} (same-sample check impossible)")

    # boots: one configuration throughout, clocks as expected at boot and after every cell
    boots, boot_bad = [], []
    if a.boots and os.path.exists(a.boots):
        boots = list(csv.DictReader(open(a.boots), delimiter="\t"))
        for k in ("launcher_md5", "image", "image_id", "env_sha", "env_n", "keys_n", "policy", "cache", "window"):
            vals = {b.get(k) for b in boots}
            if len(vals) > 1:
                boot_bad.append(f"boots differ in {k}: {sorted(v or '' for v in vals)}")
        for b in boots:
            for k, want in (("gpc_boot", a.want_gpc), ("gpc_after", a.want_gpc),
                            ("mem_boot", a.want_mem), ("mem_after", a.want_mem)):
                v = b.get(k)
                if v not in (None, "", "-") and v != want:
                    boot_bad.append(f"{b['tag']}: {k} '{v}' != '{want}'")
                if v in (None, "", "-") and k.endswith("_boot"):
                    boot_bad.append(f"{b['tag']}: {k} not recorded")

    # ---- output
    if boots:
        b0 = boots[0]
        print("== configuration (first boot; FLAGS list any boot that differs) ==")
        for k in ("image", "image_id", "launcher_md5", "env_n", "env_sha", "keys_n", "window", "slots", "cache",
                  "policy", "gpc_boot", "mem_boot", "power_limit_w", "vram_free"):
            if k in b0:
                print(f"  {k:<14} {b0[k]}")
        print(f"  boots          {len(boots)}")

    passes = list(dict.fromkeys(r["pass_"] for r in out_rows))
    for ds in dict.fromkeys(r["dataset"] for r in out_rows):
        for ps in passes:
            rs = [x for x in out_rows if x["dataset"] == ds and x["pass_"] == ps]
            if not rs:
                continue
            print(f"\n== {ds}, pass {ps} ==")
            print("| Concurrency | Requests | Output tok/s | Req/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | "
                  "Per-stream tok/s (1000/TPOT p50) | E2E p50 (s) | tau | depth |")
            print("|---|---|---|---|---|---|---|---|---|---|")
            for r in rs:
                if "out_tps" not in r:
                    print(f"| {r['conc']} | (no result; rc {r.get('rc')}) | | | | | | | | |")
                    continue
                print(f"| {r['conc']} | {r['completed']} | {fmt(r['out_tps'], 1)} | {fmt(r['req_s'], 2)} | "
                      f"{fmt(r['ttft_p50'], 0)} / {fmt(r['ttft_p99'], 0)} | {fmt(r['tpot_p50'], 2)} / "
                      f"{fmt(r['tpot_p99'], 2)} | {fmt(r['per_stream'], 0)} | {fmt(r['e2el_p50_s'], 2)} | "
                      f"{fmt(r['tau'], 2)} | {fmt(r['depth'], 2)} |")

    # A/B: the two-boot replication, per level
    spreads, spread_bad = [], []
    if len(passes) >= 2:
        pa, pb = passes[0], passes[1]
        for ds in dict.fromkeys(r["dataset"] for r in out_rows):
            cells = {(r["pass_"], r["conc"]): r for r in out_rows if r["dataset"] == ds and "out_tps" in r}
            concs = sorted({c for _, c in cells})
            print(f"\n== {ds}: pass {pa} vs {pb} (spread = |A - B| / mean) ==")
            print(f"{'conc':>4} {'tok/s A':>8} {'B':>8} {'mean':>8} {'spread':>7} {'TPOT50 A':>8} {'B':>6} "
                  f"{'TTFT50 A':>8} {'B':>6} {'tau A':>6} {'B':>5}")
            for c in concs:
                A, B = cells.get((pa, c)), cells.get((pb, c))
                if not (A and B and A["out_tps"] and B["out_tps"]):
                    print(f"{c:>4}  (pass {pb if A else pa} missing)")
                    continue
                m = (A["out_tps"] + B["out_tps"]) / 2
                s = abs(A["out_tps"] - B["out_tps"]) / m * 100
                spreads.append((ds, c, s))
                if s > a.spread_max and c not in a.spread_exempt:
                    spread_bad.append(f"{ds} c{c}: A/B spread {s:.2f} % > {a.spread_max:g} % "
                                      f"({A['out_tps']:.1f} vs {B['out_tps']:.1f} tok/s)")
                print(f"{c:>4} {fmt(A['out_tps'], 1, 8)} {fmt(B['out_tps'], 1, 8)} {fmt(m, 1, 8)} {fmt(s, 2, 6)}% "
                      f"{fmt(A['tpot_p50'], 2, 8)} {fmt(B['tpot_p50'], 2, 6)} {fmt(A['ttft_p50'], 0, 8)} "
                      f"{fmt(B['ttft_p50'], 0, 6)} {fmt(A['tau'], 2, 6)} {fmt(B['tau'], 2, 5)}")

    print("\n== diagnostics (not for the README) ==")
    print(f"{'tag':<22} {'dur s':>6} {'mean in':>7} {'out':>5} {'tot tok/s':>9} {'step p50':>8} {'tau':>5} "
          f"{'tau cli':>7} {'acc/prop':>8} {'cached %':>8} {'hits':>4} {'lines':>5}")
    for r in out_rows:
        if "out_tps" not in r:
            continue
        print(f"{r['tag']:<22} {fmt(r['duration'], 0, 6)} {fmt(r['mean_in'], 0, 7)} {fmt(r['mean_out'], 0, 5)} "
              f"{fmt(r['total_tps'], 1, 9)} {fmt(r['itl_p50'], 2, 8)} {fmt(r['tau'], 2, 5)} "
              f"{fmt(r['tau_client'], 2, 7)} {fmt(r['acc_prop'], 3, 8)} "
              f"{fmt(r['cached_share'] and r['cached_share'] * 100, 2, 8)} {r['cached_n']:>4} {r['mtp_n']:>5}")

    for tag, seq_ok, by in cat_tables:
        print(f"\n== {tag}: MTP per Spec-Bench category (mapping {'verified' if seq_ok else 'UNVERIFIED'}) ==")
        print(f"{'category':<16} {'n':>3} {'tau':>5} {'acc/prop':>8} {'gen':>7}")
        grp = {}
        for cat, (ac, pr, n, gen) in by.items():
            print(f"{cat:<16} {n:>3} {fmt(gen / (gen - ac) if gen > ac else None, 2, 5)} "
                  f"{fmt(ac / pr if pr else None, 3, 8)} {gen:>7}")
            gk = "mt_bench" if cat in MT_BENCH else cat
            x = grp.setdefault(gk, [0, 0, 0, 0])
            x[0] += ac; x[1] += pr; x[2] += n; x[3] += gen
        print(f"-- task groups (MT-bench pooled) --")
        for gk, (ac, pr, n, gen) in grp.items():
            print(f"{gk:<16} {n:>3} {fmt(gen / (gen - ac) if gen > ac else None, 2, 5)} "
                  f"{fmt(ac / pr if pr else None, 3, 8)} {gen:>7}")

    allflags = boot_bad + integrity + cache_bad + flags
    print("\nFLAGS: " + ("none" if not allflags else ""))
    for f in allflags:
        print("  - " + f)

    decision = None
    if a.decide:
        want_passes = a.decide.split()
        reasons = []
        # completeness: every (dataset, conc) seen in any pass must have a result in every required pass
        grid = {(r["dataset"], r["conc"]) for r in out_rows}
        for ps in want_passes:
            miss = [f"{ds} c{c}" for ds, c in sorted(grid)
                    if not any(r["pass_"] == ps and r["dataset"] == ds and r["conc"] == c and "out_tps" in r
                               for r in out_rows)]
            if miss:
                reasons.append(f"pass {ps} incomplete ({', '.join(miss)})")
        if boot_bad:
            decision = "VOID " + "; ".join(boot_bad[:3])
        else:
            reasons += integrity[:3] + cache_bad[:3] + spread_bad
            if reasons:
                decision = "NOT-PUBLISHABLE " + "; ".join(reasons)
            else:
                ms = max(spreads, key=lambda x: x[2]) if spreads else None
                decision = ("PUBLISHABLE " + (f"(max A/B spread {ms[2]:.2f} % at {ms[0]} c{ms[1]}; every cell cached "
                                              f"<= {a.cache_max * 100:g} %)" if ms else ""))
        print(f"\nDECISION: {decision}")
    if a.json:
        json.dump(dict(runs=out_rows, boots=boots, flags=allflags, spreads=spreads, decision=decision,
                       categories={t: {"verified": v, "by": b} for t, v, b in cat_tables}),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
