#!/usr/bin/env python3
"""Per-request stats from a TabbyAPI log (docker logs output, rich-wrapped lines).

Tabby prints one metrics record per finished request, wrapped over several lines:
  #105 chat/completions: 402 tokens generated at 122.6 T/s · prompt 12,516 tokens, 55% cached, 5,604 new in 1.31 s
  (4,278 T/s) · queued 2.76 s, first token 4.07 s, total 7.35 s · draft 274/384 accepted (71%) · max_tokens reached
The cached field is a percentage ("55% cached") or "none cached". Usage: tabby_log_stats.py LOG [LOG...]
"""
import re, sys, statistics as st

HEAD = re.compile(r"^\d\d:\d\d:\d\d\.\d+ \w+:\s+#\d+ ")
REC = re.compile(r"([\d,]+) tokens generated at [\d.]+ T/s · prompt ([\d,]+) tokens, (none|\d+%) cached, ([\d,]+) new in ([\d.]+) s"
                 r".*?queued ([\d.]+) s, first token ([\d.]+) s, total ([\d.]+) s(?: · draft (\d+)/(\d+) accepted)?")
num = lambda s: int(s.replace(",", ""))


def records(path):
    buf = []
    for line in open(path, errors="replace"):
        if HEAD.match(line) and buf:
            yield " ".join(buf)
            buf = []
        buf.append(line.strip())
    if buf:
        yield " ".join(buf)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


for path in sys.argv[1:]:
    rows = []
    for rec in records(path):
        m = REC.search(re.sub(r"\s+", " ", rec))
        if not m:
            continue
        gen, prompt, new = num(m[1]), num(m[2]), num(m[4])
        rows.append(dict(gen=gen, prompt=prompt, new=new, cached=prompt - new, prefill=float(m[5]), queued=float(m[6]),
                         ttft=float(m[7]), total=float(m[8]), acc=int(m[9] or 0), drafted=int(m[10] or 0)))
    if not rows:
        print(f"{path}: no metric records")
        continue
    P = sum(r["prompt"] for r in rows)
    C = sum(r["cached"] for r in rows)
    D = sum(r["drafted"] for r in rows)
    print(f"{path}: requests {len(rows)} · generated {sum(r['gen'] for r in rows):,} · prompt {P:,}, cached {C:,} ({C / P * 100:.1f} %), "
          f"new {P - C:,} · prefill {sum(r['prefill'] for r in rows):.0f} s · cold (<10 % cached, >4k prompt) "
          f"{sum(1 for r in rows if r['cached'] < 0.1 * r['prompt'] and r['prompt'] > 4096)} · "
          f"queued p50/p90 {pct([r['queued'] for r in rows], .5):.2f}/{pct([r['queued'] for r in rows], .9):.2f} s · "
          f"ttft p50/p90 {pct([r['ttft'] for r in rows], .5):.2f}/{pct([r['ttft'] for r in rows], .9):.2f} s · "
          f"total mean {st.mean(r['total'] for r in rows):.2f} s · draft accept {sum(r['acc'] for r in rows) / D * 100 if D else 0:.1f} %")
