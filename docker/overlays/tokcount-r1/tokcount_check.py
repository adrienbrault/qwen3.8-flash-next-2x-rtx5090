#!/usr/bin/env python3
"""tokcount r1 gate client + decision (R737). Stdlib only.

  run     3 concurrent streamed chat requests forced to exactly N tokens each (max_tokens = min_tokens = N,
          temperature 0, loop_detect_window 0 so the loop detector cannot end one early). finish_reason must be
          `length`: the engine's max_new_tokens limit is exact (job.py: the limit is rebased from the sequence at every
          requeue, not from the reported counter), so the true generated count is N BY CONSTRUCTION. As a second,
          independent client-side count the streamed reasoning + content is re-encoded through /v1/token/encode
          (add_bos false); that is approximate (the <think> markers are not streamed, re-tokenisation of greedy text
          can merge pieces) and is reported, not gated.
  decide  joins the rows with the container's per-request log lines and fn_greedy's comparison; prints the table and
          a final line `DECISION: PASS|FAIL|INCONCLUSIVE (...)`.

    tokcount_check.py run --url http://127.0.0.1:8022/v1 --model M --tag A --out longgen.jsonl [--ns 9000,13000,20000]
    tokcount_check.py decide <results dir> [--patched B --control A]
"""
import argparse, hashlib, json, os, re, sys, threading, time, urllib.request

TOPICS = [
    "the history and engineering of railway signalling, from time-interval working to ETCS level 3",
    "how a modern optimising compiler works, from lexing and parsing through SSA construction, loop optimisation, "
    "vectorisation, register allocation, instruction scheduling and link-time optimisation",   # R737 run 1: the shorter
    # wording tokenized to 59 like the railway topic and the distinct-length guard aborted the run
    "lithium-ion battery manufacturing",
]
ASK = ("Write an exhaustive, very long textbook chapter on {t}. Cover every sub-topic in depth, with numbered sections, "
       "worked examples and historical detail. Do not summarise and do not stop early; keep writing.")


def encode_len(url, text, timeout):
    body = json.dumps({"text": text, "add_bos_token": False}).encode()
    req = urllib.request.Request(url.rstrip("/").removesuffix("/v1") + "/v1/token/encode", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r).get("length")


def one(a, i, n, bar, out):
    prompt = ASK.format(t=TOPICS[i % len(TOPICS)])
    body = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n, "min_tokens": n,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}, "loop_detect_window": 0}
    req = urllib.request.Request(a.url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    rec = dict(tag=a.tag, i=i, n=n, error=None, finish_reason=None, usage=None)
    content, reasoning, usage, finish, tf, tl, frames = [], [], None, None, None, None, 0
    bar.wait()
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            for raw in r:
                ln = raw.decode("utf-8", "replace").strip()
                if not ln.startswith("data:"):
                    continue
                d = ln[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    c = json.loads(d)
                except ValueError:
                    continue
                usage = c.get("usage") or usage
                for ch in c.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    dl = ch.get("delta") or ch.get("message") or {}
                    x, y = dl.get("content"), dl.get("reasoning_content") or dl.get("reasoning")
                    if x or y:
                        tl = time.time(); tf = tf or tl; frames += 1
                        content.append(x or ""); reasoning.append(y or "")
    except Exception as e:
        rec["error"] = repr(e)[:300]
    text_r, text_c = "".join(reasoning), "".join(content)
    rec.update(t_start=round(t0, 3), t_first=tf and round(tf, 3), t_last=tl and round(tl, 3), frames=frames,
               finish_reason=finish, usage=usage, prompt_tokens=(usage or {}).get("prompt_tokens"),
               completion_tokens=(usage or {}).get("completion_tokens"),
               reasoning_chars=len(text_r), content_chars=len(text_c),
               sha=hashlib.sha256((text_r + "\n<|content|>\n" + text_c).encode()).hexdigest()[:16])
    try:
        rec["reencode_tokens"] = encode_len(a.url, text_r + text_c, 120) if rec["error"] is None else None
    except Exception as e:
        rec["reencode_tokens"] = None; rec["reencode_error"] = repr(e)[:200]
    if tf and tl and tl > tf:
        rec["client_decode_tps"] = round((n - 1) / (tl - tf), 1)
    out.append(rec)


def run(a):
    ns = [int(x) for x in a.ns.split(",")]
    # decide joins log lines to rows by prompt tokens: the prompts must tokenize to distinct lengths (same chat
    # template overhead for each, so distinct raw lengths suffice)
    plen = [encode_len(a.url, ASK.format(t=TOPICS[i % len(TOPICS)]), 60) for i in range(len(ns))]
    if len(set(plen)) != len(plen):
        print(f"ABORT: prompt token lengths not distinct {plen}; the log join would be ambiguous"); return 2
    bar = threading.Barrier(len(ns))
    out = []
    run_id = f"{time.time():.3f}"      # one id per invocation: decide gates on the last invocation of each tag
    th = [threading.Thread(target=one, args=(a, i, n, bar, out)) for i, n in enumerate(ns)]
    [t.start() for t in th]; [t.join() for t in th]
    with open(a.out, "a") as fh:
        for rec in sorted(out, key=lambda r: r["i"]):
            rec["run"] = run_id
            fh.write(json.dumps(rec) + "\n")
            print(f"LONGGEN {a.tag} n={rec['n']} finish={rec['finish_reason']} usage={rec['completion_tokens']} "
                  f"prompt={rec['prompt_tokens']} reencode={rec.get('reencode_tokens')} "
                  f"client_tps={rec.get('client_decode_tps')} err={rec['error']}")
    return 0 if all(r["error"] is None for r in out) else 1


LOGRX = re.compile(r"#(\d+) chat/completions \(stream\): ([\d,]+) tokens generated at ([\d.]+) T/s · prompt ([\d,]+) "
                   r"tokens.*?draft ([\d,]+)/([\d,]+) accepted")


def log_lines(path):
    """per-request completion lines of a container log (wrapped continuation lines joined), keyed by prompt tokens"""
    if not os.path.exists(path):
        return {}
    txt = re.sub(r"\n {10,}", " ", open(path, errors="replace").read())
    out = {}
    for m in LOGRX.finditer(txt):
        s, g, tps, p, acc, prop = m.groups()
        out.setdefault(int(p.replace(",", "")), []).append(
            dict(serial=int(s), gen=int(g.replace(",", "")), tps=float(tps), acc=int(acc.replace(",", "")),
                 prop=int(prop.replace(",", ""))))
    return out


def decide(a):
    R = a.dir
    rows = [json.loads(l) for l in open(os.path.join(R, "longgen.jsonl"))] if os.path.exists(os.path.join(R, "longgen.jsonl")) else []
    # R737 run 2: a TERMed invocation was re-run into the same results dir, and run() appends, so a tag can carry
    # several invocation blocks. container-<tag>.log is overwritten per boot, so only the LAST block of a tag joins its
    # own log lines: gate on it, print the superseded blocks as report-only. A block = rows sharing `run` (older rows
    # without it: a new block starts at i == 0, run() writes each invocation sorted by i).
    blocks = {}
    for r in rows:
        bl = blocks.setdefault(r["tag"], [])
        key = r.get("run")
        if not bl or (key is not None and bl[-1][0].get("run") != key) or (key is None and r.get("i") == 0):
            bl.append([])
        bl[-1].append(r)
    by = {t: bl[-1] for t, bl in blocks.items()}
    superseded = [(t, k, b) for t, bl in blocks.items() for k, b in enumerate(bl[:-1], 1)]
    notes, void = [], []
    print(f"{'arm':4} {'N':>6} {'finish':7} {'usage':>6} {'log':>6} {'log T/s':>8} {'client T/s':>10} {'acc/prop':>13} "
          f"{'reencode':>8} sha")
    res = {}
    for tag in (a.control, a.patched):
        L = log_lines(os.path.join(R, f"container-{tag}.log"))
        ok = []
        for r in sorted(by.get(tag, []), key=lambda r: r["n"]):
            lg = L.get(r.get("prompt_tokens"), [])
            lg = lg[-1] if len(lg) == 1 else None      # a prompt length seen twice cannot be joined: VOID the row
            valid = r["error"] is None and r["finish_reason"] == "length" and lg is not None
            if not valid:
                void.append(f"{tag} N={r['n']} (error={r['error']}, finish={r['finish_reason']}, log={'ok' if lg else 'unjoined'})")
            u = r.get("completion_tokens")
            ok.append(dict(n=r["n"], valid=valid, usage=u, log=lg and lg["gen"], sha=r.get("sha")))
            print(f"{tag:4} {r['n']:>6} {str(r['finish_reason']):7} {str(u):>6} {str(lg and lg['gen']):>6} "
                  f"{str(lg and lg['tps']):>8} {str(r.get('client_decode_tps')):>10} "
                  f"{(str(lg['acc']) + '/' + str(lg['prop'])) if lg else '-':>13} {str(r.get('reencode_tokens')):>8} {r.get('sha')}")
        res[tag] = ok
    A, B = res.get(a.control, []), res.get(a.patched, [])
    want = len(a.ns.split(","))
    b_exact = len(B) == want and all(x["valid"] and x["usage"] == x["n"] and x["log"] == x["n"] for x in B)
    b_wrong = [x for x in B if x["valid"] and (x["usage"] != x["n"] or x["log"] != x["n"])]
    a_repro = len(A) == want and all(x["valid"] and x["usage"] is not None and x["usage"] < x["n"] and x["log"] == x["usage"] for x in A)
    same = sum(1 for x, y in zip(sorted(A, key=lambda r: r["n"]), sorted(B, key=lambda r: r["n"])) if x["sha"] == y["sha"])
    print(f"long-generation text identity A vs B (report only, c{want} batch): {same}/{min(len(A), len(B))} identical")
    for t, k, b in superseded:
        last = {x["n"]: x.get("sha") for x in by[t]}
        print(f"SUPERSEDED (report only, not gated; its log lines were overwritten) {t} invocation {k}/{len(blocks[t])}: "
              + ", ".join(f"N={x['n']} finish={x['finish_reason']} usage={x.get('completion_tokens')} "
                          f"client_tps={x.get('client_decode_tps')} sha={x.get('sha')}"
                          f"{' (= gated)' if last.get(x['n']) == x.get('sha') else ' (!= gated)'}"
                          for x in sorted(b, key=lambda r: r["n"])))
    gl = ""
    gp = os.path.join(R, "greedy-compare.txt")
    if os.path.exists(gp):
        gl = next((l.strip() for l in open(gp) if l.startswith(f"GREEDY {a.patched} vs {a.control}:")), "")
    g_ok = gl.endswith("6 identical, IDENTICAL")
    print(f"fn_greedy: {gl or 'no comparison line'}")
    fo = os.path.join(R, "foreign.tsv")
    if os.path.exists(fo):
        print("foreign requests per arm (serials in our windows not sent by us):", open(fo).read().strip().replace("\n", "; "))
    if void:
        print("VOID rows: " + "; ".join(void))
    if b_wrong or (gl and not g_ok):
        why = []
        if b_wrong:
            why.append("patched count != N on " + ", ".join(f"N={x['n']} usage={x['usage']} log={x['log']}" for x in b_wrong))
        if gl and not g_ok:
            why.append("fn_greedy not 6/6 identical")
        print("DECISION: FAIL (" + "; ".join(why) + ")")
    elif b_exact and a_repro and g_ok:
        print(f"DECISION: PASS (patched usage = log = N on {want}/{want}; control undercounts {want}/{want}; fn_greedy 6/6)")
    else:
        why = []
        if not b_exact: why.append("patched rows incomplete/void")
        if not a_repro: why.append("control did not reproduce the undercount on every row")
        if not g_ok: why.append("no fn_greedy comparison")
        print("DECISION: INCONCLUSIVE (" + "; ".join(why) + ")")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--url", required=True); r.add_argument("--model", required=True)
    r.add_argument("--tag", required=True); r.add_argument("--out", required=True)
    r.add_argument("--ns", default="9000,13000,20000"); r.add_argument("--timeout", type=int, default=1500)
    d = sp.add_parser("decide")
    d.add_argument("dir"); d.add_argument("--patched", default="B"); d.add_argument("--control", default="A")
    d.add_argument("--ns", default="9000,13000,20000")
    a = ap.parse_args()
    sys.exit(run(a) if a.cmd == "run" else decide(a))
