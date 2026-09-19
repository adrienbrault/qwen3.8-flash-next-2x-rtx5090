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

CODE, PROSE = "#0969da", "#cf222e"
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


def fn_bench_rates(path, arm):
    """Per '<conc>-<kind>': mean aggregate t/s per round, and mean per-stream t/s.

    Rounds are keyed by the full tag, so two boots of the same arm stay separate rounds.
    """
    tok, wall, per = collections.defaultdict(int), {}, collections.defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if not r.get("ok") or not r["tag"].startswith(arm):
            continue
        tok[(r["tag"], r["run"])] += r["completion_tokens"] or 0
        wall[(r["tag"], r["run"])] = r["round_wall_s"]
        per[r["tag"].split("-", 1)[1]].append(r["wall_tps"])
    agg = collections.defaultdict(list)
    for (tag, run), v in tok.items():
        agg[tag.split("-", 1)[1]].append(v / wall[(tag, run)])
    return {k: st.mean(v) for k, v in agg.items()}, {k: st.mean(v) for k, v in per.items()}


def merged(dicts, key):
    vals = [d[key] for d in dicts if key in d]
    return st.mean(vals) if vals else None


R570 = RESULTS / "2026-09-19-r570-promote-c5-policy" / "records.jsonl"
R571 = RESULTS / "2026-09-19-r571-promote-c5-policy-2" / "records.jsonl"
R580 = RESULTS / "2026-09-20-r580-decode-curve" / "records.jsonl"
CONC = [4, 5, 6, 7, 8]


def figure_decode_scaling():
    """R580 read 1 to 8 streams on one boot; before it existed the curve was stitched from R570 and R571,
    which only ran 4 to 8, so the fallback below draws the shorter x range from those two rounds."""
    if R580.exists():
        a, p = fn_bench_rates(R580, "S")
        conc = [c for c in range(1, 9) if f"c{c}-code" in a]
        agg = {k: [a[f"c{c}-{k}"] for c in conc] for k in ("code", "prose")}
        per = {k: [p[f"c{c}-{k}"] for c in conc] for k in ("code", "prose")}
        src = "R580"
    else:
        # The B arm is the policy served since R576, so it is the one that describes the daily.
        conc = CONC
        a570, p570 = fn_bench_rates(R570, "B")
        a571, p571 = fn_bench_rates(R571, "B")
        agg = {k: [merged([a570, a571], f"c{c}-{k}") for c in conc] for k in ("code", "prose")}
        per = {k: [merged([p570, p571], f"c{c}-{k}") for c in conc] for k in ("code", "prose")}
        src = "R570+R571"

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#d8dee4")
    handles = []
    for kind, color, dy in (("code", CODE, 7), ("prose", PROSE, -14)):
        handles += ax.plot(conc, agg[kind], marker="o", color=color, linewidth=2, label=f"{kind}, all streams")
        handles += ax2.plot(conc, per[kind], marker="s", markersize=4, linestyle="--", color=color,
                            linewidth=1.6, label=f"{kind}, one stream")
        annotate(ax, conc, agg[kind], color, dy=dy)
        annotate(ax2, conc, per[kind], color, dy=dy)
    ax.set_title("Decode rate against concurrency")
    ax.set_xlabel("concurrent streams")
    ax.set_ylabel("tokens per second, all streams")
    ax2.set_ylabel("tokens per second, one stream")
    ax.set_ylim(0, max(max(v) for v in agg.values()) * 1.25)
    ax2.set_ylim(0, max(max(v) for v in per.values()) * 1.25)
    ax.set_xticks(conc)
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)
    ax.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=9, ncol=2, loc="lower center")
    print(f"decode scaling ({src}) at {conc}")
    print("  aggregate:", {k: [round(v) for v in v2] for k, v2 in agg.items()})
    print("  per stream:", {k: [round(v) for v in v2] for k, v2 in per.items()})
    save(fig, "decode-scaling.svg", "Decode rate against concurrency, aggregate and per stream")


def figure_prefill():
    rows = collections.defaultdict(list)
    for line in open(RESULTS / "2026-09-19-r574-chunk4096" / "prefill.jsonl"):
        r = json.loads(line)
        if r.get("ttft_s") and r.get("prompt_tokens"):
            rows[r["tag"]].append((r["prompt_tokens"], r["prompt_tokens"] / r["ttft_s"]))
    keys = ["pf-S-30000", "pf-S-60000", "pf-S-120000"]
    toks = [st.mean([t for t, _ in rows[k]]) for k in keys]
    rate = [st.mean([v for _, v in rows[k]]) for k in keys]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(toks, rate, marker="o", color=CODE, linewidth=2)
    annotate(ax, toks, rate, CODE)
    ax.set_title("Cold prefill rate against prompt length")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("prompt tokens per second")
    ax.set_ylim(0, 12000)
    ax.set_xticks(toks, [f"{round(t / 1000)}k" for t in toks])
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)
    print("prefill:", [round(v) for v in rate], "at", [round(t) for t in toks])
    save(fig, "prefill.svg", "Cold prefill rate at three prompt lengths")


if __name__ == "__main__":
    figure_decode_scaling()
    figure_prefill()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
