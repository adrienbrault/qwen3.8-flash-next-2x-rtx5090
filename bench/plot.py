#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Draw the README's figures from the published raw records.

    uv run bench/plot.py            # writes docs/img/*.svg

Every figure reads `bench/results/<date>-<round>/`, so no figure can carry a number that is not
in this repository, and each prints what it drew so the values can be checked against the
round's write-up.
"""

import collections
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"
OUT = ROOT / "docs" / "img"

CODE, PROSE, PREFILL = "#0969da", "#cf222e", "#8250df"
plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 10,
    "axes.edgecolor": "#d8dee4",
    "axes.labelcolor": "#57606a",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.color": "#57606a",
    "ytick.color": "#57606a",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", metadata={"Title": caption})
    plt.close(fig)


def annotate(ax, xs, ys, color, fmt="{:.0f}", dy=7):
    """dy places a series' labels above (positive) or below (negative) its markers, so two series that
    read within a few tokens per second of each other do not print on top of one another."""
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", fontsize=8.5, color=color)


def decode_rates(path, arm):
    """Per '<conc>-<kind>' for one arm of a decode-curve round (tags '<ARM><boot>-c<conc>-<kind>', both boots pooled;
    R704 has the arms OLD and NEW, R719, R719b and R787a the arm NEW only):

    per_stream   median over requests of decode_tps = (tokens - 1) / (t_last - t_first), the streaming rate after
                 the first token;
    decode_agg   mean over rounds of the sum of the round's decode_tps (TTFT and straggler tails excluded);
    wall_agg     mean over rounds of all streams' tokens over the round's wall time (the end-to-end burst figure,
                 which was the README's headline until R704);
    ttft         median over requests of the time to the first token;
    overlap      mean over rounds of (min t_last - max t_first) / mean decode window: the share of the mean decode
                 window during which every stream of the round is decoding. It bounds how far decode_agg overstates
                 the rate the streams sustain together.

    These are the definitions of the R704 and R719 drivers' analysis step (R719b re-ran the R719 driver, R787a is a copy of
    it with provenance checks added), so the printed values reproduce their curve.tsv.
    """
    reqs, rounds = collections.defaultdict(list), collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if not r.get("ok") or not r["tag"][: len(arm)] == arm or not r["tag"][len(arm)].isdigit():
            continue
        shape = r["tag"].split("-", 1)[1]
        reqs[shape].append(r)
        rounds[(shape, r["tag"], r["run"])].append(r)
    if not reqs:
        raise ValueError(f"{path}: no ok records for arm {arm}")
    out = {}
    for shape, rs in reqs.items():
        rr = [v for (s, _, _), v in rounds.items() if s == shape]
        overlap = [(min(r["t_last_abs"] for r in v) - max(r["t_first_abs"] for r in v))
                   / st.mean(r["t_last_abs"] - r["t_first_abs"] for r in v) for v in rr]
        out[shape] = {
            "per_stream": st.median(r["decode_tps"] for r in rs),
            "decode_agg": st.mean(sum(r["decode_tps"] for r in v) for v in rr),
            "wall_agg": st.mean(sum(r["completion_tokens"] for r in v) / v[0]["round_wall_s"] for v in rr),
            "ttft": st.median(r["ttft_s"] for r in rs),
            "overlap": st.mean(overlap),
        }
    return out


R704 = RESULTS / "2026-09-24-r704-decode-curve-ab" / "records.jsonl"
R719 = RESULTS / "2026-09-24-r719-decode-curve" / "records.jsonl"
R719B = RESULTS / "2026-09-25-r719b-decode-curve" / "records.jsonl"
R580 = RESULTS / "2026-09-20-r580-decode-curve-try2" / "records.jsonl"
R580_PREFILL = R580.parent / "prefill.jsonl"
R554_DEPTH = RESULTS / "2026-09-19-r554-depth-decode" / "depth.jsonl"
# R787 (2026-09-27): the four figure inputs re-measured on the R785 daily (tabbyapi:rebase-dev-r3, pool 901,120, 42 env
# keys with the tiled HC prefill) with the instruments of R719b, R580 (prefill part), R554 and R731b.
R787A = RESULTS / "2026-09-27-r787a-decode-curve" / "records.jsonl"
R787B_PREFILL = RESULTS / "2026-09-27-r787b-prefill-curve" / "prefill.jsonl"
R787C_DEPTH = RESULTS / "2026-09-27-r787c-depth-decode" / "depth.jsonl"
# R813 (2026-09-30): R787a's decode curve (same driver, prompts, forced length and two boots) on the served image,
# tabbyapi:merge-tok-r1 with 41 env keys (the prefill merge and the asynchronous stash on).
R813A = RESULTS / "2026-09-30-r813-decode-curve" / "records.jsonl"


def figure_decode_scaling():
    """The served configuration's decode curve: R813 (tabbyapi:merge-tok-r1 at a 901,120-token pool, 41 env keys,
    memory clock +4500, core offset 0, stock power; two boots, 1 to 8 streams, each stream on its own prompt; R787a's
    driver). The chart draws the decode metrics only; the round-wall aggregate, TTFT and the overlap are printed for the
    write-up's table."""
    new = decode_rates(R813A, "NEW")
    conc = [c for c in range(1, 9) if f"c{c}-code" in new]
    agg = {k: [new[f"c{c}-{k}"]["decode_agg"] for c in conc] for k in ("code", "prose")}
    per = {k: [new[f"c{c}-{k}"]["per_stream"] for c in conc] for k in ("code", "prose")}

    # Two panels rather than a twin axis: the aggregate rises while the per-stream rate falls, and on one plot the
    # two pairs of lines cross between 2 and 4 streams, where their labels land on top of each other.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for a, series, title, ylabel in (
            (ax, agg, "Decode aggregate", "decode tokens per second, sum over streams"),
            (ax2, per, "Decode rate per stream", "decode tokens per second, one stream (median)")):
        for kind, color in (("code", CODE), ("prose", PROSE)):
            a.plot(conc, series[kind], marker="o", markersize=5, color=color, linewidth=2, label=kind)
        # At each x the higher of the two values is labelled above its marker and the lower one below.
        for i, x in enumerate(conc):
            hi = "code" if series["code"][i] >= series["prose"][i] else "prose"
            for kind, color in (("code", CODE), ("prose", PROSE)):
                annotate(a, [x], [series[kind][i]], color, dy=7 if kind == hi else -14)
        a.set_title(title)
        a.set_xlabel("concurrent streams")
        a.set_ylabel(ylabel)
        a.set_ylim(0, max(max(v) for v in series.values()) * 1.2)
        a.set_xticks(conc)
        a.grid(axis="y", color="#eaeef2")
        a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=9, loc="lower right" if a is ax else "upper right")
    fig.suptitle("Decode rate after the first token against concurrency, served configuration", fontsize=11,
                 fontweight="bold")
    print(f"decode scaling (R813, 2 boots x 3 rounds) at {conc}")
    print("  shape      per-stream   decode agg   round-wall agg   TTFT     overlap")
    for kind in ("code", "prose"):
        for c in conc:
            n = new[f"c{c}-{kind}"]
            print(f"  {kind:5} c{c}   {n['per_stream']:6.1f}       {n['decode_agg']:4.0f}         {n['wall_agg']:4.0f}"
                  f"             {n['ttft']:.2f} s   {n['overlap']:.3f}")
    ov = [new[f"c{c}-{k}"]["overlap"] for c in conc if c > 1 for k in ("code", "prose")]
    print(f"  overlap at 2-8 streams, per shape: {min(ov):.3f} to {max(ov):.3f}")
    save(fig, "decode-scaling.svg", "Decode rate after the first token against concurrency, sum over streams and per stream")


def print_r719():
    """R719's curve (stack-r3-rows32 with the draft-KV window, stock memory clock), printed for R719b's comparison table.
    R719 and R719b ran the same driver and prompts on the same image."""
    old, new = decode_rates(R719, "NEW"), decode_rates(R719B, "NEW")
    conc = [c for c in range(1, 9) if f"c{c}-code" in new]
    print(f"R719 -> R719b (2 boots x 3 rounds each) at {conc}")
    print("  shape      per-stream R719 -> R719b   decode agg R719 -> R719b   TTFT R719 / R719b")
    for kind in ("code", "prose"):
        for c in conc:
            o, n = old[f"c{c}-{kind}"], new[f"c{c}-{kind}"]
            print(f"  {kind:5} c{c}   {o['per_stream']:6.1f} -> {n['per_stream']:6.1f} ({n['per_stream'] / o['per_stream']:.3f}x)"
                  f"   {o['decode_agg']:4.0f} -> {n['decode_agg']:4.0f} ({n['decode_agg'] / o['decode_agg']:.3f}x)"
                  f"   {o['ttft']:.2f} / {n['ttft']:.2f} s")


def print_r787a():
    """R719b's curve (the R728 daily, stack-r3-rows32) against R787a's (the R785 daily, rebase-dev-r3), printed for
    R787a's comparison table. Same driver, prompts and tags."""
    old, new = decode_rates(R719B, "NEW"), decode_rates(R787A, "NEW")
    conc = [c for c in range(1, 9) if f"c{c}-code" in new and f"c{c}-code" in old]
    print(f"R719b -> R787a (2 boots x 3 rounds each) at {conc}")
    print("  shape      per-stream R719b -> R787a   decode agg R719b -> R787a   TTFT R719b / R787a")
    for kind in ("code", "prose"):
        for c in conc:
            o, n = old[f"c{c}-{kind}"], new[f"c{c}-{kind}"]
            print(f"  {kind:5} c{c}   {o['per_stream']:6.1f} -> {n['per_stream']:6.1f} ({n['per_stream'] / o['per_stream']:.3f}x)"
                  f"   {o['decode_agg']:4.0f} -> {n['decode_agg']:4.0f} ({n['decode_agg'] / o['decode_agg']:.3f}x)"
                  f"   {o['ttft']:.2f} / {n['ttft']:.2f} s")


def print_r813():
    """R787a's curve (tabbyapi:rebase-dev-r3, 2026-09-27) against R813's (the served tabbyapi:merge-tok-r1, 2026-09-30),
    printed for R813's comparison table. Same driver, prompts and tags; two sessions three days apart."""
    old, new = decode_rates(R787A, "NEW"), decode_rates(R813A, "NEW")
    conc = [c for c in range(1, 9) if f"c{c}-code" in new and f"c{c}-code" in old]
    print(f"R787a -> R813 (2 boots x 3 rounds each) at {conc}")
    print("  shape      per-stream R787a -> R813   decode agg R787a -> R813   TTFT R787a / R813")
    for kind in ("code", "prose"):
        for c in conc:
            o, n = old[f"c{c}-{kind}"], new[f"c{c}-{kind}"]
            print(f"  {kind:5} c{c}   {o['per_stream']:6.1f} -> {n['per_stream']:6.1f} ({n['per_stream'] / o['per_stream']:.3f}x)"
                  f"   {o['decode_agg']:4.0f} -> {n['decode_agg']:4.0f} ({n['decode_agg'] / o['decode_agg']:.3f}x)"
                  f"   {o['ttft']:.2f} / {n['ttft']:.2f} s")


def print_r704():
    """R704's two arms (stack-r2 against the configuration before R701), printed for its write-up's tables. R704
    sent one prompt to every stream of a round, R719 one prompt per stream, so the two rounds are not drawn together."""
    new, old = decode_rates(R704, "NEW"), decode_rates(R704, "OLD")
    conc = [c for c in range(1, 9) if f"c{c}-code" in new]
    print(f"R704 (2 boots x 3 rounds per arm) at {conc}")
    print("  shape      per-stream OLD -> NEW   decode agg OLD -> NEW   round-wall agg OLD -> NEW   TTFT OLD / NEW"
          "   overlap OLD / NEW")
    for kind in ("code", "prose"):
        for c in conc:
            o, n = old[f"c{c}-{kind}"], new[f"c{c}-{kind}"]
            print(f"  {kind:5} c{c}   {o['per_stream']:6.1f} -> {n['per_stream']:6.1f} ({n['per_stream'] / o['per_stream']:.3f}x)"
                  f"   {o['decode_agg']:4.0f} -> {n['decode_agg']:4.0f}   {o['wall_agg']:4.0f} -> {n['wall_agg']:4.0f}"
                  f"   {o['ttft']:.2f} / {n['ttft']:.2f} s   {o['overlap']:.3f} / {n['overlap']:.3f}")
    ov = [d[f"c{c}-{k}"]["overlap"] for d in (old, new) for c in conc if c > 1 for k in ("code", "prose")]
    print(f"  overlap at 2-8 streams, per shape and arm: {min(ov):.3f} to {max(ov):.3f}")


def need(path):
    """The figures draw one named round each. A missing input is an error, never a silent switch to an older round."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; the figure is drawn from that round only")
    return path


def depth_decode(path=R787C_DEPTH):
    """Decode rate at one stream on top of an already-prefilled context, per kind: R787c by default (the R554 probe on
    the R785 daily; its code targets aim at ~100k / ~200k tokens, where R554's code points landed at ~180k and past the
    window). R554's records are read only by print_depth_compare().

    Returns per kind a list of (prompt tokens, mean decode rate, mean tokens per decode step), the last being completion
    tokens / streamed frames: fn_bench streams one frame per verify step, so it carries the MTP draft acceptance."""
    rows = collections.defaultdict(list)
    for line in open(need(path)):
        r = json.loads(line)
        # c1 only: the file also holds a 4-stream arm, whose per-request rate is a different quantity.
        if r.get("decode_tps") and r.get("prompt_tokens") and r["tag"].startswith("c1-"):
            rows[(r["tag"].split("-")[1], r["prompt_tokens"])].append(
                (r["decode_tps"], r["completion_tokens"] / r["client_frames"] if r.get("client_frames") else None))
    if not rows:
        raise ValueError(f"{path}: no c1 decode records")
    out = collections.defaultdict(list)
    for (kind, toks), v in sorted(rows.items(), key=lambda kv: kv[0][1]):
        tps = [t for _, t in v if t]
        out[kind].append((toks, st.mean(d for d, _ in v), st.mean(tps) if tps else None))
    return out


PREFILL_TARGETS = (30000, 60000, 120000, 200000, 240000)


def prefill_points(path):
    """Cold prefill per target of an R580-protocol round (R580, R787b): the round asked for the filler budget that
    lands on each target, so its tags are the token counts it aimed at and the points are what the server counted
    (mean prompt tokens, mean prompt tokens / TTFT over the target's salted prompts). Every target must have records."""
    rows = collections.defaultdict(list)
    for line in open(need(path)):
        r = json.loads(line)
        if r.get("ttft_s") and r.get("prompt_tokens"):
            rows[r["tag"]].append((r["prompt_tokens"], r["prompt_tokens"] / r["ttft_s"]))
    keys = [f"pf-{t}" for t in PREFILL_TARGETS]
    missing = [k for k in keys if not rows[k]]
    if missing:
        raise ValueError(f"{path}: no records for {missing}")
    return ([st.mean([t for t, _ in rows[k]]) for k in keys], [st.mean([v for _, v in rows[k]]) for k in keys])


def figure_prefill():
    """Cold prefill from R787b and decode at depth from R787c, both on the R785 daily. The older rounds (R580, R554)
    are printed by print_prefill_compare() and print_depth_compare(), not drawn."""
    path = R787B_PREFILL
    toks, rate = prefill_points(path)
    depth = depth_decode()

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#d8dee4")
    handles = ax.plot(toks, rate, marker="o", color=PREFILL, linewidth=2, label="prefill rate")
    annotate(ax, toks, rate, PREFILL)
    # the series with the higher mean rate labels above its markers, the other below, so the two do not collide
    # Each point is labelled with its rate and its tokens per decode step: at depth the rate follows how much of the
    # continuation the MTP draft predicts, so the two numbers are read together.
    upper = max(("code", "prose"), key=lambda k: st.mean(v for _, v, _ in depth[k]))
    for kind, color in (("code", CODE), ("prose", PROSE)):
        dy = 7 if kind == upper else -24
        xs = [t for t, _, _ in depth[kind]]
        ys = [v for _, v, _ in depth[kind]]
        handles += ax2.plot(xs, ys, marker="s", markersize=4, linestyle="--", color=color, linewidth=1.6,
                            label=f"decode at depth, {kind} (label: t/s, tokens per step)")
        for i, (x, y, tps) in enumerate(depth[kind]):
            # the first point sits at the left edge: its label starts at the marker instead of centring on it
            ax2.annotate(f"{y:.0f}, {tps:.2f}/step" if tps else f"{y:.0f}", (x, y), textcoords="offset points",
                         xytext=(-4 if i == 0 else 0, dy), ha="left" if i == 0 else "center", fontsize=8, color=color)
    ax.set_title("Prompt length costs prefill time; the decode step time stays flat")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("prompt tokens per second, prefill")
    ax2.set_ylabel("tokens per second, decode at 1 stream")
    ax.set_ylim(0, max(rate) * 1.3)
    ax2.set_ylim(0, max(v for d in depth.values() for _, v, _ in d) * 1.6)
    ax.set_xticks(toks, [f"{round(t / 1000)}k" for t in toks])
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)
    ax.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=9, loc="lower right")
    print(f"prefill ({path.parent.name}):", [round(v) for v in rate], "t/s at", [round(t) for t in toks], "tokens")
    print("decode at depth (tokens, t/s, tokens per step):",
          {k: [(round(t), round(v), round(p, 2) if p else None) for t, v, p in d] for k, d in depth.items()})
    save(fig, "prefill.svg", "Cold prefill rate and decode rate at depth against prompt length")


R731B = RESULTS / "2026-09-25-r731b-std-bench-stock" / "results"
R787D = RESULTS / "2026-09-27-r787d-std-bench" / "results"
# R811 (2026-09-29): R787d's protocol on the served image (arm NEW, tabbyapi:merge-tok-r1) and on the previous one (arm
# OLD, tabbyapi:tokenize-offloop-r2), alternating, a fresh boot per cell. R811b (2026-09-30) re-ran the ShareGPT
# 4-stream cell, whose NEW A/B spread was 4.22 %; that cell is published as the mean of the four NEW boots (the rule was
# written down after R811 and before R811b ran).
R811 = RESULTS / "2026-09-29-r811-std-bench-ab"
R811B = RESULTS / "2026-09-29-r811b-sharegpt-c4"
R811B_CELL = ("sharegpt", 4)
SHAREGPT, SPECBENCH = "#1a7f37", "#9a6700"


def std_bench_runs(path):
    """Every `[AB]-<dataset>-c<conc>.json` of one standard-benchmark results directory, per (dataset, conc), in pass
    order, each with its τ from the run directory's summary.json (bench/std_bench_summary.py's container-log count)."""
    runs = collections.defaultdict(list)
    files = sorted(need(path).glob("[AB]-*-c*.json"))
    if not files:
        raise FileNotFoundError(f"{path} holds no [AB]-*-c*.json cell")
    summary = path.parent / "summary.json"
    tau = {r["tag"]: r["tau"] for r in json.load(open(summary))["runs"]} if summary.exists() else {}
    for f in files:
        d = json.load(open(f))
        d["_src"], d["_tau"] = f"{path.parent.parent.name}/{path.parent.name}/{f.stem}", tau.get(f.stem)
        runs[(d["dataset"], int(d["conc"]))].append(d)
    return runs


def std_bench_served_runs(arm="NEW"):
    """The served image's standard benchmark, one arm: R811's cells, with the ShareGPT 4-stream cell replaced by the four
    boots of R811 and R811b together. Both directories are read explicitly; either one missing, or R811b holding
    anything but that cell's two passes, raises."""
    runs = std_bench_runs(R811 / arm / "results")
    extra = std_bench_runs(R811B / arm / "results")
    if set(extra) != {R811B_CELL} or len(extra[R811B_CELL]) != 2:
        raise ValueError(f"{R811B / arm}: expected passes A and B of {R811B_CELL} only, found {sorted(extra)}")
    if len(runs) != 8 or any(len(v) != 2 for v in runs.values()):
        raise ValueError(f"{R811 / arm}: expected 8 cells of 2 passes")
    runs[R811B_CELL] = runs[R811B_CELL] + extra[R811B_CELL]
    return runs


def std_bench_cells(runs):
    """Per (dataset, conc), the mean over the cell's runs (two passes, or four boots): output tok/s is `vllm bench
    serve`'s output_throughput (all completion tokens over the run's wall time, prefill and request turnover
    included); per-stream is 1000 / TPOT p50, TPOT = (latency - TTFT) / (output tokens - 1) per request, which
    includes the time a request waits while other requests' prefill chunks run. Both as in bench/std_bench_summary.py."""
    out = {}
    for k, v in runs.items():
        per = []
        for d in v:
            tpot = [(lat - ttft) / (n - 1) for lat, ttft, n in zip(d["latencies"], d["ttfts"], d["output_lens"]) if n > 1]
            per.append(1.0 / st.median(tpot))
        out[k] = (st.mean(d["output_throughput"] for d in v), st.mean(per))
    return out


def std_bench(path):
    """One standard-benchmark results directory (R731b, R787d), passes A/B mean per cell."""
    return std_bench_cells(std_bench_runs(path))


def figure_std_bench():
    """The decode curve (R813: steady-state decode after the first token, all streams starting together, no prefill
    in the window) against the standard benchmark (R811 + R811b, arm NEW: closed loop, requests arriving as others
    finish, so their prefill chunks interleave with the running streams' decode). The two also differ in output length
    (1,024 forced tokens against ~210-256), which puts more of each request's life in TTFT and turnover in the standard
    benchmark."""
    fn = decode_rates(R813A, "NEW")
    fn_conc = [c for c in range(1, 9) if f"c{c}-code" in fn]
    sb = std_bench_cells(std_bench_served_runs())
    datasets = (("sharegpt", "ShareGPT V3", SHAREGPT), ("specbench", "Spec-Bench", SPECBENCH))
    sb_conc = sorted({c for (_, c) in sb})

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for a, idx, fn_key, title in ((ax, 0, "decode_agg", "Aggregate over streams"), (ax2, 1, "per_stream", "Per stream")):
        for kind, color in (("code", CODE), ("prose", PROSE)):
            ys = [fn[f"c{c}-{kind}"][fn_key] for c in fn_conc]
            a.plot(fn_conc, ys, marker="o", markersize=4, color=color, linewidth=2, label=f"decode only, {kind} (R813)")
            if kind == "code":
                annotate(a, fn_conc, ys, color, dy=7)
        for key, name, color in datasets:
            ys = [sb[(key, c)][idx] for c in sb_conc]
            a.plot(sb_conc, ys, marker="s", markersize=4, color=color, linewidth=2, linestyle="--",
                   label=f"{name}, prefill interleaved (R811)")
            # Labels only from 2 streams on the aggregate panel and from 4 on the per-stream panel (the values are in
            # the README tables): below that the dashed and solid series lie within a few label heights of each other.
            # On the per-stream panel both datasets label below their markers, ShareGPT one line lower, because the
            # decode-only labels sit above theirs.
            first = 2 if a is ax else 4
            if a is ax:
                dy = -14 if key == "sharegpt" else 7
            else:
                dy = -26 if key == "sharegpt" else -14
            annotate(a, sb_conc, [y if c >= first else None for c, y in zip(sb_conc, ys)], color, dy=dy)
        a.set_title(title)
        a.set_xlabel("concurrent streams")
        a.set_ylim(0, max(fn[f"c{c}-{k}"][fn_key] for c in fn_conc for k in ("code", "prose")) * 1.2)
        a.set_xticks(fn_conc)
        a.grid(axis="y", color="#eaeef2")
        a.set_axisbelow(True)
        a.legend(frameon=False, fontsize=8, loc="lower right" if a is ax else "upper right")
    ax.set_ylabel("tokens per second, sum over streams\n(standard benchmark: output tok/s, wall clock)")
    ax2.set_ylabel("tokens per second, one stream\n(standard benchmark: 1000 / TPOT p50)")
    fig.suptitle("Decode alone against the standard benchmark, served configuration", fontsize=11, fontweight="bold")
    print(f"standard benchmark (R811 arm NEW, passes A/B mean; ShareGPT c4 the mean of R811's and R811b's four boots)"
          f" at {sb_conc}")
    for key, name, _ in datasets:
        print(f"  {name:11}  output tok/s {[round(sb[(key, c)][0], 1) for c in sb_conc]}"
              f"   per-stream {[round(sb[(key, c)][1]) for c in sb_conc]}")
    save(fig, "std-bench.svg", "Decode alone against the standard benchmark, sum over streams and per stream")


def print_r787d():
    """R731b (the R728 daily) against R787d (the R785 daily), same protocol and sample, passes A/B mean per cell."""
    old, new = std_bench(R731B), std_bench(R787D)
    print("R731b -> R787d (passes A/B mean)")
    for key in ("sharegpt", "specbench"):
        for c in sorted(c for (k, c) in new if k == key):
            if (key, c) not in old:
                continue
            (oo, op), (no, np_) = old[(key, c)], new[(key, c)]
            print(f"  {key:9} c{c}   output tok/s {oo:6.1f} -> {no:6.1f} ({no / oo:.3f}x)"
                  f"   per-stream {op:5.1f} -> {np_:5.1f} ({np_ / op:.3f}x)")


def fmt_range(vals, nd):
    """'a' when every value prints the same at nd decimals, else 'min–max'."""
    lo, hi = f"{min(vals):,.{nd}f}", f"{max(vals):,.{nd}f}"
    return lo if lo == hi else f"{lo}–{hi}"


def print_std_bench_tables(runs, name):
    """The README's Standard benchmark tables from the raw records: per cell the mean over its runs, the p99 columns as
    the range over them, the A/B spread |A - B| / mean (with more than two runs, the range of output tok/s and the run
    count instead), per-stream the mean over the runs of 1000 / TPOT p50 (vLLM's own percentile, as
    bench/std_bench_summary.py), τ the mean of the runs' container-log τ."""
    print(f"standard benchmark tables ({name})")
    for key, title in (("sharegpt", "ShareGPT V3"), ("specbench", "Spec-Bench")):
        print(f"  {title}")
        for c in sorted(c for (k, c) in runs if k == key):
            v = runs[(key, c)]
            out = [d["output_throughput"] for d in v]
            spread = (f"{abs(out[0] - out[1]) / st.mean(out) * 100:.2f} %" if len(v) == 2
                      else f"{min(out):.1f}–{max(out):.1f} ({len(v)} boots)")
            tpot50 = st.mean(d["p50_tpot_ms"] for d in v)
            taus = [d["_tau"] for d in v]
            tau = f"{st.mean(taus):.2f}" if all(t is not None for t in taus) else "-"
            print(f"  | {c} | {st.mean(out):.1f} | {spread} | {st.mean(d['request_throughput'] for d in v):.2f}"
                  f" | {st.mean(d['p50_ttft_ms'] for d in v):,.0f} / {fmt_range([d['p99_ttft_ms'] for d in v], 0)}"
                  f" | {tpot50:.2f} / {fmt_range([d['p99_tpot_ms'] for d in v], 2)}"
                  f" | {st.mean(1000 / d['p50_tpot_ms'] for d in v):.0f}"
                  f" | {st.mean(d['p50_e2el_ms'] for d in v) / 1000:.2f} | {tau} |")
            if len(v) > 2:
                print("      runs: " + ", ".join(f"{d['_src']} {d['output_throughput']:.1f}" for d in v))


def print_r811():
    """R811 (+ R811b at ShareGPT 4 streams): the served image (NEW) against the previous one (OLD), alternating boots in
    one session. Per cell the mean over the boot pairs of NEW / OLD for output tok/s and for the mean TTFT."""
    new, old = std_bench_served_runs("NEW"), std_bench_served_runs("OLD")
    print("R811 NEW / OLD, mean of the per-pair ratios (ShareGPT c4: four pairs, R811 and R811b)")
    for key in ("sharegpt", "specbench"):
        for c in sorted(c for (k, c) in new if k == key):
            n, o = new[(key, c)], old[(key, c)]
            if [d["pass"] for d in n] != [d["pass"] for d in o]:
                raise ValueError(f"{key} c{c}: NEW and OLD runs are not paired by pass")
            out = [a["output_throughput"] / b["output_throughput"] for a, b in zip(n, o)]
            ttft = [a["mean_ttft_ms"] / b["mean_ttft_ms"] for a, b in zip(n, o)]
            print(f"  {key:9} c{c}   output tok/s {st.mean(out):.3f} ({' / '.join(f'{r:.3f}' for r in out)})"
                  f"   TTFT mean {st.mean(ttft):.3f} ({' / '.join(f'{r:.3f}' for r in ttft)})")


def print_prefill_compare():
    """R580's cold prefill (the R580 daily, 2026-09-20) against R787b's (the R785 daily), same protocol and targets."""
    (ot, orate), (nt, nrate) = prefill_points(R580_PREFILL), prefill_points(R787B_PREFILL)
    print("R580 -> R787b cold prefill (mean of 3 salted prompts per target)")
    for t, a, b, x, y in zip(PREFILL_TARGETS, ot, nt, orate, nrate):
        print(f"  target {t:>7,}   tokens {a:9,.0f} / {b:9,.0f}   t/s {x:7,.0f} -> {y:7,.0f} ({y / x:.3f}x)")


def print_depth_compare():
    """R554's decode at depth (1 stream) against R787c's. The code targets differ (R554's landed at ~180k tokens)."""
    for name, path in (("R554", R554_DEPTH), ("R787c", R787C_DEPTH)):
        d = depth_decode(path)
        print(f"{name} decode at depth, 1 stream (tokens, t/s, tokens per step):",
              {k: [(round(t), round(v), round(p, 2) if p else None) for t, v, p in pts] for k, pts in d.items()})


if __name__ == "__main__":
    figure_decode_scaling()
    figure_std_bench()
    print_std_bench_tables(std_bench_served_runs(), "R811 arm NEW, ShareGPT c4 from R811 and R811b")
    print_r811()
    print_r813()
    print_r787a()
    print_r787d()
    print_r719()
    print_r704()
    figure_prefill()
    print_prefill_compare()
    print_depth_compare()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
