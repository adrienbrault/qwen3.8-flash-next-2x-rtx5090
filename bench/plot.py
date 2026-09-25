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
    R704 has the arms OLD and NEW, R719 the arm NEW only):

    per_stream   median over requests of decode_tps = (tokens - 1) / (t_last - t_first), the streaming rate after
                 the first token;
    decode_agg   mean over rounds of the sum of the round's decode_tps (TTFT and straggler tails excluded);
    wall_agg     mean over rounds of all streams' tokens over the round's wall time (the end-to-end burst figure,
                 which was the README's headline until R704);
    ttft         median over requests of the time to the first token;
    overlap      mean over rounds of (min t_last - max t_first) / mean decode window: the share of the mean decode
                 window during which every stream of the round is decoding. It bounds how far decode_agg overstates
                 the rate the streams sustain together.

    These are the definitions of the R704 and R719 drivers' analysis step, so the printed values reproduce their
    curve.tsv.
    """
    reqs, rounds = collections.defaultdict(list), collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if not r.get("ok") or not r["tag"][: len(arm)] == arm or not r["tag"][len(arm)].isdigit():
            continue
        shape = r["tag"].split("-", 1)[1]
        reqs[shape].append(r)
        rounds[(shape, r["tag"], r["run"])].append(r)
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
R580 = RESULTS / "2026-09-20-r580-decode-curve-try2" / "records.jsonl"
R580_PREFILL = R580.parent / "prefill.jsonl"


def figure_decode_scaling():
    """The served configuration's decode curve: R719 (stack-r3-rows32, two boots, 1 to 8 streams, each stream on its
    own prompt). The chart draws the decode metrics only; the round-wall aggregate, TTFT and the overlap are printed
    for the write-up's table."""
    new = decode_rates(R719, "NEW")
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
    print(f"decode scaling (R719, 2 boots x 3 rounds) at {conc}")
    print("  shape      per-stream   decode agg   round-wall agg   TTFT     overlap")
    for kind in ("code", "prose"):
        for c in conc:
            n = new[f"c{c}-{kind}"]
            print(f"  {kind:5} c{c}   {n['per_stream']:6.1f}       {n['decode_agg']:4.0f}         {n['wall_agg']:4.0f}"
                  f"             {n['ttft']:.2f} s   {n['overlap']:.3f}")
    ov = [new[f"c{c}-{k}"]["overlap"] for c in conc if c > 1 for k in ("code", "prose")]
    print(f"  overlap at 2-8 streams, per shape: {min(ov):.3f} to {max(ov):.3f}")
    save(fig, "decode-scaling.svg", "Decode rate after the first token against concurrency, sum over streams and per stream")


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


def depth_decode():
    """R554 read decode rate at one stream on top of an already-prefilled context, per kind."""
    rows = collections.defaultdict(list)
    for line in open(RESULTS / "2026-09-19-r554-depth-decode" / "depth.jsonl"):
        r = json.loads(line)
        # c1 only: the file also holds a 4-stream arm, whose per-request rate is a different quantity.
        if r.get("decode_tps") and r.get("prompt_tokens") and r["tag"].startswith("c1-"):
            rows[(r["tag"].split("-")[1], r["prompt_tokens"])].append(r["decode_tps"])
    out = collections.defaultdict(list)
    for (kind, toks), v in sorted(rows.items(), key=lambda kv: kv[0][1]):
        out[kind].append((toks, st.mean(v)))
    return out


def figure_prefill():
    """R580 asked for the filler budget that lands on each target, so its tags are the token counts it aimed at
    and the points are what the server counted. Before it, the only prefill records were R574's, whose "120k"
    is really 90,008 tokens because fn_bench's --ctx is a budget at about 0.75 tokens per unit."""
    if R580_PREFILL.exists():
        path, keys = R580_PREFILL, [f"pf-{t}" for t in (30000, 60000, 120000, 200000, 240000)]
    else:
        path, keys = (RESULTS / "2026-09-19-r574-chunk4096" / "prefill.jsonl",
                      ["pf-S-30000", "pf-S-60000", "pf-S-120000"])
    rows = collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("ttft_s") and r.get("prompt_tokens"):
            rows[r["tag"]].append((r["prompt_tokens"], r["prompt_tokens"] / r["ttft_s"]))
    keys = [k for k in keys if rows[k]]
    toks = [st.mean([t for t, _ in rows[k]]) for k in keys]
    rate = [st.mean([v for _, v in rows[k]]) for k in keys]
    depth = depth_decode()

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#d8dee4")
    handles = ax.plot(toks, rate, marker="o", color=PREFILL, linewidth=2, label="prefill rate")
    annotate(ax, toks, rate, PREFILL)
    for kind, color, dy in (("code", CODE, 7), ("prose", PROSE, -14)):
        xs = [t for t, _ in depth[kind]]
        ys = [v for _, v in depth[kind]]
        handles += ax2.plot(xs, ys, marker="s", markersize=4, linestyle="--", color=color, linewidth=1.6,
                            label=f"decode at depth, {kind}")
        annotate(ax2, xs, ys, color, dy=dy)
    ax.set_title("Prompt length costs latency, not rate")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("prompt tokens per second, prefill")
    ax2.set_ylabel("tokens per second, decode at 1 stream")
    ax.set_ylim(0, max(rate) * 1.3)
    ax2.set_ylim(0, max(v for d in depth.values() for _, v in d) * 1.6)
    ax.set_xticks(toks, [f"{round(t / 1000)}k" for t in toks])
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)
    ax.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=9, loc="lower right")
    print("prefill:", [round(v) for v in rate], "t/s at", [round(t) for t in toks], "tokens")
    print("decode at depth:", {k: [(round(t), round(v)) for t, v in d] for k, d in depth.items()})
    save(fig, "prefill.svg", "Cold prefill rate and decode rate at depth against prompt length")


if __name__ == "__main__":
    figure_decode_scaling()
    print_r704()
    figure_prefill()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
