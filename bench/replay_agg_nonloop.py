#!/usr/bin/env python3
"""Per-stream decode from a TabbyAPI container log, all requests / non-loop-stopped / loop-stopped, with MTP acceptance
(R785 review). Same record regex as tabby_log_agg.py; per-stream = tokens / summed per-request decode seconds.
  replay_agg_nonloop.py container.log [...]"""
import re,sys
HEAD=re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+) \w+:\s+#(\d+) ")
REC=re.compile(r"([\d,]+) tokens generated at ([\d.]+) T/s . prompt ([\d,]+) tokens, (none|[\d.]+%) cached, ([\d,]+) new in ([\d.]+) s.*?first token ([\d.]+) s, total ([\d.]+) s(?:.*?draft (\d+)/(\d+) accepted)?")
num=lambda s:int(s.replace(",",""))
def recs(path):
    buf=[]
    for line in open(path,errors="replace"):
        if HEAD.match(line) and buf: yield " ".join(buf); buf=[]
        buf.append(line.strip())
    if buf: yield " ".join(buf)
for p in sys.argv[1:]:
    rows=[]
    for r in recs(p):
        m=REC.search(r)
        if not m: continue
        n,rate=num(m.group(1)),float(m.group(2))
        if n<=0 or rate<=0: continue
        rows.append(dict(n=n,rate=rate,prompt=num(m.group(3)),loop="loop detected" in r,chat="chat/completions" in r,a=int(m.group(9) or 0),d=int(m.group(10) or 0)))
    def ps(s): return sum(x["n"] for x in s)/sum(x["n"]/x["rate"] for x in s)
    nl=[x for x in rows if not x["loop"]]; lp=[x for x in rows if x["loop"]]
    acc=lambda s: sum(x["a"] for x in s)/max(1,sum(x["d"] for x in s))
    print(f"{p}: n {len(rows)} (chat {sum(x['chat'] for x in rows)}) per-stream all {ps(rows):.1f} | non-loop n {len(nl)} {ps(nl):.1f} acc {acc(nl):.2f} | loop n {len(lp)} {ps(lp):.1f} acc {acc(lp):.2f} tokens-in-loop-reqs {sum(x['n'] for x in lp)} | max prompt {max(x['prompt'] for x in rows)}")
