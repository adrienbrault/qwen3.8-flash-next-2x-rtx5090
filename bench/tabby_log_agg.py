#!/usr/bin/env python3
"""Server-side truth for the R586 SWE-bench run.

Aggregate decode is tokens divided by the wall seconds during which at least one stream was
decoding -- the union of per-request decode intervals, not a mean of per-request rates. A 0.8 s
reply and a 221 s reply count equally in a mean, which is how an earlier pass read 176 t/s where
the truth was 65.6. Tabby's log timestamps carry no date, so the day is rolled forward by hand.
"""
import re
import statistics as st
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/srv/qwen5090/results/2026-09-20-r586-swebench-full/docker-final.log"
HEAD = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+) \w+:\s+#(\d+) ")
REC = re.compile(
    r"([\d,]+) tokens generated at ([\d.]+) T/s . prompt ([\d,]+) tokens, (none|[\d.]+%) cached, "
    r"([\d,]+) new in ([\d.]+) s.*?first token ([\d.]+) s, total ([\d.]+) s"
    r"(?:.*?draft (\d+)/(\d+) accepted)?"
)
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


ivals, dsecs, rates, gens, prompts, cached, ttfts = [], [], [], [], [], [], []
acc_a = acc_d = 0
prev, day = -1.0, 0
for rec in records(LOG):
    h = HEAD.match(rec)
    if not h:
        continue
    t = int(h.group(1)) * 3600 + int(h.group(2)) * 60 + int(h.group(3)) + float("0." + h.group(4))
    if prev >= 0 and t < prev - 3600:
        day += 86400
    prev = t
    end = t + day
    m = REC.search(rec)
    if not m:
        continue
    n, rate = num(m.group(1)), float(m.group(2))
    if n <= 0 or rate <= 0:
        continue
    d = n / rate
    ivals.append((end - d, end))
    dsecs.append(d)
    rates.append(rate)
    gens.append(n)
    prompts.append(num(m.group(3)))
    cached.append(0.0 if m.group(4) == "none" else float(m.group(4).rstrip("%")))
    ttfts.append(float(m.group(7)))
    if m.group(9):
        acc_a += int(m.group(9))
        acc_d += int(m.group(10))

ivals.sort()
busy = 0.0
cs = ce = None
for s, e in ivals:
    if cs is None:
        cs, ce = s, e
    elif s <= ce:
        ce = max(ce, e)
    else:
        busy += ce - cs
        cs, ce = s, e
if cs is not None:
    busy += ce - cs
span = ivals[-1][1] - ivals[0][0]
toks = sum(gens)

print(f"window         : {span/3600:.2f} h, {len(gens):,} completed requests")
print(f"generated      : {toks:,} tokens")
print(f"decode-busy    : {busy/3600:.2f} h ({100*busy/span:.0f}% of the window)")
print()
print(f"AGGREGATE      : {toks/busy:,.1f} t/s   tokens / wall seconds with >=1 stream decoding")
print(f"PER STREAM     : {toks/sum(dsecs):,.1f} t/s   tokens / summed per-request decode seconds")
print(f"mean streams   : {sum(dsecs)/busy:.2f} decoding while anything decodes")
print()
g = sorted(gens)
print(f"generated tokens/request : median {st.median(g):.0f}  p90 {g[9*len(g)//10]}  p99 {g[99*len(g)//100]}  max {g[-1]}")
p = sorted(prompts)
print(f"prompt tokens/request    : median {st.median(p):,.0f}  p90 {p[9*len(p)//10]:,}  max {p[-1]:,}")
print(f"prefix cached            : median {st.median(cached):.0f}%")
r = sorted(rates)
print(f"per-request T/s          : p10 {r[len(r)//10]:.0f}  median {st.median(r):.0f}  p90 {r[9*len(r)//10]:.0f}")
if acc_d:
    print(f"MTP acceptance           : {acc_a:,}/{acc_d:,} = {100*acc_a/acc_d:.0f}%")

# Which requests actually consume the decode seconds?
order = sorted(range(len(gens)), key=lambda i: -dsecs[i])
tot_d = sum(dsecs)
for frac in (0.01, 0.05, 0.10):
    k = max(1, int(len(order) * frac))
    sel = order[:k]
    print(f"\nslowest {frac:.0%} by decode seconds ({k} requests): "
          f"{100*sum(dsecs[i] for i in sel)/tot_d:.0f}% of decode time, "
          f"{100*sum(gens[i] for i in sel)/toks:.0f}% of tokens, "
          f"median {st.median([gens[i] for i in sel]):.0f} tokens at {st.median([rates[i] for i in sel]):.0f} T/s")

slow = [i for i in range(len(gens)) if rates[i] < 50]
if slow:
    print(f"\nrequests under 50 T/s: {len(slow)} ({100*len(slow)/len(gens):.1f}%), "
          f"{100*sum(dsecs[i] for i in slow)/tot_d:.0f}% of decode time, "
          f"{100*sum(gens[i] for i in slow)/toks:.0f}% of tokens")
