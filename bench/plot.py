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

CODE, PROSE, CAND, SERVED = "#0969da", "#cf222e", "#1a7f37", "#9a6700"
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


def annotate(ax, xs, ys, color, fmt="{:.0f}"):
    for x, y in zip(xs, ys):
        if y is None:
            continue
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points", xytext=(0, 7),
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
CONC = [4, 5, 6, 7, 8]


def figure_decode_scaling():
    a570, p570 = fn_bench_rates(R570, "A")
    a571, p571 = fn_bench_rates(R571, "A")
    agg = {k: [merged([a570, a571], f"c{c}-{k}") for c in CONC] for k in ("code", "prose")}
    per = {k: [merged([p570, p571], f"c{c}-{k}") for c in CONC] for k in ("code", "prose")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, data, title, ylab, top in (
        (ax1, agg, "Aggregate decode", "tokens per second, all streams", 800),
        (ax2, per, "Per stream", "tokens per second, one stream", 160),
    ):
        for kind, color in (("code", CODE), ("prose", PROSE)):
            ax.plot(CONC, data[kind], marker="o", color=color, label=kind, linewidth=2)
            annotate(ax, CONC, data[kind], color)
        ax.set_title(title)
        ax.set_xlabel("concurrent streams")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, top)
        ax.set_xticks(CONC)
        ax.grid(axis="y", color="#eaeef2")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right" if ax is ax1 else "upper right")
    print("decode scaling, aggregate:", {k: [round(v) for v in v2] for k, v2 in agg.items()})
    print("decode scaling, per stream:", {k: [round(v) for v in v2] for k, v2 in per.items()})
    save(fig, "decode-scaling.svg", "Decode rate from 4 to 8 concurrent streams, aggregate and per stream")


def figure_c5_policy():
    a = [fn_bench_rates(R570, "A")[0], fn_bench_rates(R571, "A")[0]]
    b = [fn_bench_rates(R570, "B")[0], fn_bench_rates(R571, "B")[0]]
    served = [st.mean([v for v in (merged(a, f"c{c}-code"), merged(a, f"c{c}-prose")) if v]) for c in CONC]
    cand = [st.mean([v for v in (merged(b, f"c{c}-code"), merged(b, f"c{c}-prose")) if v]) for c in CONC]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = range(len(CONC))
    ax.bar([i - 0.19 for i in x], served, width=0.36, color=SERVED, label="[[4, 3], [8, 1]] (was served)")
    ax.bar([i + 0.19 for i in x], cand, width=0.36, color=CAND, label="[[4, 3], [5, 2], [8, 1]] (served since R576)")
    annotate(ax, [i - 0.19 for i in x], served, SERVED)
    annotate(ax, [i + 0.19 for i in x], cand, CAND)
    ax.set_title("Draft depth 2 at 5 streams: mean of code and prose")
    ax.set_xlabel("concurrent streams")
    ax.set_ylabel("tokens per second, all streams")
    ax.set_xticks(list(x), [str(c) for c in CONC])
    ax.set_ylim(0, 800)
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    print("c5 policy, served:", [round(v) for v in served], "candidate:", [round(v) for v in cand])
    save(fig, "c5-draft-policy.svg", "Aggregate decode under the served draft policy and the 5-stream policy")


def figure_acceptance():
    d = RESULTS / "2026-09-19-r572-mtp-accept-ablation"

    def positions(tag, which=0):
        reqs = [json.loads(x) for x in open(d / f"stats-{tag}.jsonl") if '"rec":"request"' in x.replace(" ", "")]
        reqs = [r for r in reqs if r.get("tokens") == 2048]
        r = reqs[which]
        return [r["pos"][k][0] / r["rounds"] for k in sorted(r["pos"])]

    served, fp16 = positions("S"), positions("K16")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = range(3)
    ax.bar([i - 0.19 for i in x], served, width=0.36, color=CODE, label="8-bit KV (served)")
    ax.bar([i + 0.19 for i in x], fp16, width=0.36, color=CAND, label="full-precision KV (diagnostic)")
    annotate(ax, [i - 0.19 for i in x], served, CODE, "{:.2f}")
    annotate(ax, [i + 0.19 for i in x], fp16, CAND, "{:.2f}")
    ax.set_title("MTP drafts accepted by position (code, greedy, 2,048 tokens)")
    ax.set_xlabel("draft position")
    ax.set_ylabel("share accepted")
    ax.set_xticks(list(x), ["first", "second", "third"])
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#eaeef2")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    print("acceptance served:", [round(v, 3) for v in served], "full precision:", [round(v, 3) for v in fp16])
    save(fig, "mtp-acceptance.svg", "Share of MTP drafts accepted at each position, 8-bit against full-precision KV")


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
    figure_c5_policy()
    figure_acceptance()
    figure_prefill()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.svg"))))
