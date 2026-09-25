"""Parse TabbyAPI per-request completion lines (container-*.log) into records:
serial, generated tokens, prompt tokens, cached tokens, new tokens, draft accepted/proposed.
Usage: parse_container.py <container.log> [--min-gen 1024]"""
import re, sys, json

def parse(path):
    txt = open(path, errors="replace").read()
    # join the wrapped continuation lines
    lines = txt.split("\n")
    joined, cur = [], None
    for ln in lines:
        if re.match(r"^\d\d:\d\d:\d\d\.\d+ ", ln) or ln.startswith(" -- "):
            if cur is not None:
                joined.append(cur)
            cur = ln
        elif cur is not None and ln.startswith(" " * 10):
            cur += " " + ln.strip()
        else:
            if cur is not None:
                joined.append(cur)
            cur = None
            joined.append(ln)
    if cur is not None:
        joined.append(cur)
    out = []
    for ln in joined:
        m = re.search(r"#(\d+) chat/completions \(stream\): ([\d,]+) tokens generated at ([\d.]+) T/s · prompt ([\d,]+) tokens, (.*?) · (?:queued [\d.]+ s, )?first token ([\d.]+) s, total ([\d.]+) s · draft ([\d,]+)/([\d,]+) accepted", ln)
        if not m:
            continue
        serial, gen, tps, prompt, cache_part, ft, tot, acc, prop = m.groups()
        prompt = int(prompt.replace(",", ""))
        mn = re.search(r"([\d,]+) new", cache_part)
        cached = prompt - int(mn.group(1).replace(",", "")) if mn else 0
        ts = ln[:12]
        out.append(dict(serial=int(serial), ts=ts, gen=int(gen.replace(",", "")), tps=float(tps), prompt=prompt,
                        cached=cached, new=prompt - cached, acc=int(acc.replace(",", "")),
                        prop=int(prop.replace(",", "")), cache_txt=cache_part))
    return out

if __name__ == "__main__":
    recs = parse(sys.argv[1])
    ming = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--min-gen" else 0
    for r in recs:
        if r["gen"] >= ming:
            print(json.dumps(r))
