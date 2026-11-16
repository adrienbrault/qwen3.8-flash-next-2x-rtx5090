#!/usr/bin/env python3
"""Per-request decode-step counters from a TabbyAPI container log of the served image.

The served image prints one `R828_COUNTER {json}` line per finished request (generated tokens, decode steps)
and one TabbyAPI metrics block ("#N ...: T tokens generated at R T/s · prompt P tokens ..."). This writes one
JSON line per request, in log order: serial, endpoint, generated tokens, decode steps and, when its metrics block
was logged intact, the prompt token count. Client-side rates on short answers are biased (the stream's last frame
arrives ~0.1 ms after the one before it, so one step's tokens are counted without its time); tokens per decode
step from these counters, divided by the measured step interval, is not.

    python3 server_steps.py container.log > server-steps.jsonl
"""
import json
import re
import sys

text = open(sys.argv[1], errors="replace").read()
prompts = {}
for m in re.finditer(r"#(\d+) [\w/]+(?: \(stream\))?: \d+ tokens generated at\s+[\d.,]+ T/s(.*?)(?=\n\S)", text, re.S):
    p = re.search(r"prompt ([\d,]+) tokens", " ".join(m.group(2).split()))
    if p:
        prompts[int(m.group(1))] = int(p.group(1).replace(",", ""))
for line in text.splitlines():
    if line.startswith("R828_COUNTER "):
        c = json.loads(line[len("R828_COUNTER "):])
        serial = int(re.match(r"#(\d+)", c["label"]).group(1))
        rec = dict(serial=serial, endpoint=c["label"].split(" ", 1)[1], generated=c["gen_tokens"], steps=c["decode_steps"])
        if serial in prompts:
            rec["prompt_tokens"] = prompts[serial]
        print(json.dumps(rec))
