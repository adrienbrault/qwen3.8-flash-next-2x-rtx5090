#!/usr/bin/env python3
"""
Box A/B client for the NVMe tier (round 3). Stdlib only; talks to a running TabbyAPI over HTTP and, for the log and
disk checks, to docker and du on the host. Every step writes its raw responses to --out.

  fill     --sizes 30000,120000 --salt S     for each size: one cold request, then the same prompt again (warm, VRAM
                                              hit); saves prompts + output hashes + cached tokens to <out>/fill.json
  wait-drained --container NAME              wait for a " -- nvme tier: drained" line newer than the start of the
                                              last request that prefilled new pages (<out>/mark.json)
  verify                                      after `docker rm -f` + relaunch: the same prompts once more; PASS when each
                                              output hash equals the pre-restart WARM hash and cached tokens equal the
                                              warm request's (the prefix and its checkpoint came back from disk)
  decode   --ntok 120000 --salt S             fresh salted prefix (admitted, drain pending), then c1 and c4 greedy
                                              decode immediately (pump in flight) and again after the drained line
  first-token --size 30000 --tag T            the stored fill prompt, max_tokens 1, top-5 first-token logprobs (FAIL 2
                                              control: tier-ON after restart vs a tier-OFF boot)
  churn    --dir HOSTDIR --cap-gb 2 --n 6     distinct salted 40k prompts; du -sb of the tier directory sampled every
                                              0.2 s and after each drain; PASS when the peak stays <= cap + 1 segment

Common: --url http://127.0.0.1:8022 --out DIR --docker "sudo docker" --container NAME
Exit code 0 = PASS, 1 = FAIL, 2 = setup error.
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request

WORDS = (
    "river stone lantern orbit copper meadow signal harbor quiet ember glacier violet canyon anchor thistle "
    "marble compass willow falcon cinder prairie beacon saffron tundra quartz juniper monsoon atlas velvet pebble "
    "horizon cobalt maple drift summit lagoon ember prism cedar garnet ripple beacon fjord meteor orchard sable "
    "tempest umber vessel wander zephyr basalt cradle dune estuary fathom grove hollow inlet jetty kelp ledge"
).split()


def post(url, path, body, timeout = 1800):
    req = urllib.request.Request(url + path, data = json.dumps(body).encode(), method = "POST",
                                 headers = {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout = timeout) as r:
        data = json.loads(r.read())
    return data, time.monotonic() - t0


def model_id(url):
    with urllib.request.urlopen(url + "/v1/model", timeout = 30) as r:
        return json.loads(r.read()).get("id")


def make_prompt(url, ntok, salt):
    rng = random.Random(f"{salt}:{ntok}")
    words = int(ntok / 1.4)
    for _ in range(4):
        text = f"[{salt}] " + " ".join(rng.choice(WORDS) for _ in range(words))
        rng = random.Random(f"{salt}:{ntok}")
        try:
            n = post(url, "/v1/token/encode", {"text": text, "add_bos_token": True})[0]["length"]
        except Exception:
            break
        if abs(n - ntok) < ntok * 0.02:
            break
        words = int(words * ntok / max(n, 1))
    return text + "\n\nIn one sentence, which word appears most often above?"


MODEL = [None]


def complete(url, prompt, max_tokens = 48):
    if MODEL[0] is None:
        MODEL[0] = model_id(url)
    # TabbyAPI returns `usage` (and prompt_tokens_details.cached_tokens) on non-streamed completions only when
    # stream_options.include_usage is set (endpoints/OAI/utils/completion.py:374, :85-89)
    body = {"model": MODEL[0], "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
            "stream": False, "stream_options": {"include_usage": True}}
    data, dt = post(url, "/v1/completions", body)
    text = data["choices"][0]["text"]
    usage = data.get("usage")
    if not usage:
        raise RuntimeError("no usage in the completion response: cached_tokens cannot be read")
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return {
        "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": cached,
        "seconds": round(dt, 3),
        "text": text,
    }


def docker_logs(args, since = None):
    """(epoch seconds, line) for each container log line, from docker's own timestamps."""
    cmd = args.docker.split() + ["logs", "--timestamps"] + (["--since", str(int(since) - 1)] if since else []) + \
        [args.container]
    r = subprocess.run(cmd, capture_output = True, text = True)
    out = []
    for line in (r.stdout + r.stderr).splitlines():
        ts, _, rest = line.partition(" ")
        try:
            t = datetime.datetime.strptime(ts[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo = datetime.timezone.utc).timestamp()
        except ValueError:
            t = 0.0
        out.append((t, rest))
    return out


def mark(args, t):
    """Remember the start of the last request that prefilled new pages: wait-drained needs a drained line after it."""
    json.dump({"since": t}, open(os.path.join(args.out, "mark.json"), "w"))


def cmd_fill(args):
    out = {"sizes": [], "salt": args.salt, "model": model_id(args.url)}
    for ntok in [int(x) for x in args.sizes.split(",")]:
        prompt = make_prompt(args.url, ntok, args.salt)
        mark(args, time.time())
        cold = complete(args.url, prompt)
        warm = complete(args.url, prompt)
        row = {"ntok": ntok, "prompt": prompt, "cold": cold, "warm": warm}
        out["sizes"].append(row)
        print(f"fill {ntok}: prompt {cold['prompt_tokens']} tok; cold {cold['hash']} cached {cold['cached_tokens']} "
              f"{cold['seconds']} s; warm {warm['hash']} cached {warm['cached_tokens']} {warm['seconds']} s", flush = True)
    out["t_end"] = time.time()
    json.dump(out, open(os.path.join(args.out, "fill.json"), "w"), indent = 1)
    return 0


def cmd_wait_drained(args):
    """PASS when a " -- nvme tier: drained" line is newer than the last marked cold request (mark.json)."""
    since = json.load(open(os.path.join(args.out, "mark.json")))["since"]
    t0 = time.time()
    while time.time() - t0 < args.timeout:
        lines = [(t, l) for t, l in docker_logs(args, since) if "nvme tier:" in l]
        done = [(t, l) for t, l in lines if t >= since and "nvme tier: drained" in l]
        if done:
            print(f"drained {done[-1][0] - since:.1f} s after the marked request: {done[-1][1]}")
            open(os.path.join(args.out, "drained.txt"), "a").write("\n".join(l for _, l in lines) + "\n")
            return 0
        time.sleep(1)
    print(f"no drained line within {args.timeout} s; last tier lines:\n" + "\n".join(
        l for _, l in [x for x in docker_logs(args) if "nvme tier:" in x[1]][-5:]))
    return 1


def cmd_first_token(args):
    """R526 FAIL 2 control: the stored 30k (or --size) fill prompt, max_tokens 1, top-5 logprobs of the first
    token. Run it on the tier-ON boot after the restart and on a tier-OFF boot: if the first-token margin between
    the top two candidates is small (< ~0.5 nats), an immediate EOS is a prefill-numerics near-tie, not the tier."""
    fill = json.load(open(args.fill or os.path.join(args.out, "fill.json")))
    rows = [r for r in fill["sizes"] if args.size in (0, r["ntok"])]
    if MODEL[0] is None:
        MODEL[0] = model_id(args.url)
    out = []
    for row in rows:
        body = {"model": MODEL[0], "prompt": row["prompt"], "max_tokens": 1, "temperature": 0, "top_k": 1,
                "stream": False, "logprobs": 5, "stream_options": {"include_usage": True}}
        data, dt = post(args.url, "/v1/completions", body)
        ch = data["choices"][0]
        top = ((ch.get("logprobs") or {}).get("top_logprobs") or [None])[0] or {}
        ranked = sorted(top.items(), key = lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
        cached = ((data.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        out.append({"ntok": row["ntok"], "text": ch.get("text"), "finish": ch.get("finish_reason"), "top5": ranked,
                    "margin": margin, "cached": cached, "tag": args.tag})
        print(f"first-token {row['ntok']} [{args.tag}]: {ch.get('text')!r} finish {ch.get('finish_reason')} cached "
              f"{cached}; top5 {[(t, round(v, 3)) for t, v in ranked]}; margin {margin}", flush = True)
    with open(os.path.join(args.out, "first_token.jsonl"), "a") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return 0


def cmd_verify(args):
    fill = json.load(open(os.path.join(args.out, "fill.json")))
    ok = True
    rows = []
    for row in fill["sizes"]:
        r = complete(args.url, row["prompt"])
        warm = row["warm"]
        same_hash = r["hash"] == warm["hash"]
        # Primary gate: greedy output equals the pre-restart warm output and the prefix came from disk. A cached
        # count within one page of the warm one is reported, not failed (the disk chain may end a page earlier).
        same_cached = r["cached_tokens"] == warm["cached_tokens"]
        near_cached = abs(r["cached_tokens"] - warm["cached_tokens"]) <= 256
        passed = same_hash and near_cached and r["cached_tokens"] > 0
        ok &= passed
        rows.append({"ntok": row["ntok"], "restored": r, "warm": warm, "cold": row["cold"]})
        note = "" if same_cached else f" (NOTE: cached differs from warm by {r['cached_tokens'] - warm['cached_tokens']})"
        print(f"verify {row['ntok']}: after restart {r['hash']} cached {r['cached_tokens']} ({r['seconds']} s latency) | "
              f"warm {warm['hash']} cached {warm['cached_tokens']} ({warm['seconds']} s) | cold {row['cold']['hash']} "
              f"({row['cold']['seconds']} s) -> {'PASS' if passed else 'FAIL'}{note}", flush = True)
    json.dump(rows, open(os.path.join(args.out, "verify.json"), "w"), indent = 1)
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def stream_decode(url, prompt, max_tokens, results, idx):
    body = {"model": MODEL[0] or model_id(url), "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
            "top_k": 1, "stream": True, "ignore_eos": True}
    req = urllib.request.Request(url + "/v1/completions", data = json.dumps(body).encode(), method = "POST",
                                 headers = {"Content-Type": "application/json"})
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout = 600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                n += 1
                if first is None:
                    first = time.monotonic()
    last = time.monotonic()
    results[idx] = (n, (last - first) if first else 0.0)


def decode_rate(url, streams, max_tokens, tag):
    prompts = [f"Write a long, detailed Python module implementing a {w} scheduler with tests. ({tag} {i})"
               for i, w in enumerate(["priority", "round-robin", "deadline", "fair-share"][:streams])]
    res = [None] * streams
    th = [threading.Thread(target = stream_decode, args = (url, p, max_tokens, res, i)) for i, p in enumerate(prompts)]
    [t.start() for t in th]
    [t.join() for t in th]
    # SSE chunks per second: with MTP a chunk carries several tokens, so this is only a relative during/after number
    per = [n / dt for n, dt in res if dt > 0]
    return {"streams": streams, "per_stream_chunks_per_s": [round(x, 1) for x in per],
            "aggregate_chunks_per_s": round(sum(per), 1)}


def cmd_decode(args):
    prompt = make_prompt(args.url, args.ntok, args.salt)
    mark(args, time.time())
    r = complete(args.url, prompt, max_tokens = 1)
    print(f"admitted {r['prompt_tokens']} tokens in {r['seconds']} s", flush = True)
    during = [decode_rate(args.url, 1, args.max_tokens, "d1"), decode_rate(args.url, 4, args.max_tokens, "d4")]
    rc = cmd_wait_drained(args)
    after = [decode_rate(args.url, 1, args.max_tokens, "a1"), decode_rate(args.url, 4, args.max_tokens, "a4")]
    out = {"during_drain": during, "after_drain": after, "drained": rc == 0}
    json.dump(out, open(os.path.join(args.out, "decode.json"), "w"), indent = 1)
    for d, a in zip(during, after):
        print(f"decode c{d['streams']}: during drain {d['aggregate_chunks_per_s']} chunks/s, after {a['aggregate_chunks_per_s']} chunks/s "
              f"({(d['aggregate_chunks_per_s'] / a['aggregate_chunks_per_s'] - 1) * 100 if a['aggregate_chunks_per_s'] else 0:+.1f} %)")
    return 0


def du_bytes(path, du = "sudo du"):
    """Apparent bytes of the tier directory (the tier's cap counts file sizes, which bounds the blocks written)."""
    r = subprocess.run(du.split() + ["-sb", path], capture_output = True, text = True)
    try:
        return int(r.stdout.split()[0])
    except (IndexError, ValueError):
        return -1


def cmd_churn(args):
    cap = int(args.cap_gb * 1024**3)
    segment = min(256 * 1024**2, cap // 8)
    peak = [0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], du_bytes(args.dir, args.du))
            time.sleep(0.2)
    th = threading.Thread(target = sampler)
    th.start()
    rows = []
    try:
        for i in range(args.n):
            p = make_prompt(args.url, args.ntok, f"{args.salt}-churn-{i}")
            mark(args, time.time())
            r = complete(args.url, p, max_tokens = 16)
            cmd_wait_drained(args)
            d = du_bytes(args.dir, args.du)
            rows.append({"i": i, "prompt_tokens": r["prompt_tokens"], "du": d})
            print(f"churn {i}: {r['prompt_tokens']} tok, du {d / 1024**3:.3f} GiB (cap {args.cap_gb} GiB)", flush = True)
    finally:
        stop.set()
        th.join()
    slack = 64 * 1024  # du -sb also counts directory entries and the lock file, which disk_bytes() does not
    ok = all(0 <= r["du"] <= cap + slack for r in rows) and peak[0] <= cap + segment
    json.dump({"rows": rows, "peak": peak[0], "cap": cap, "segment": segment}, open(
        os.path.join(args.out, "churn.json"), "w"), indent = 1)
    print(f"CHURN peak {peak[0] / 1024**3:.3f} GiB, after-drain max {max(r['du'] for r in rows) / 1024**3:.3f} GiB, "
          f"cap {args.cap_gb} GiB (+64 KiB after drain, +{segment / 1024**2:.0f} MiB transient) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices = ["fill", "wait-drained", "verify", "decode", "churn", "first-token"])
    ap.add_argument("--url", default = "http://127.0.0.1:8022")
    ap.add_argument("--out", required = True)
    ap.add_argument("--docker", default = "sudo docker")
    ap.add_argument("--container", default = "flashnext")
    ap.add_argument("--du", default = "sudo du", help = "du command (the tier directory is root-owned, mode 0700)")
    ap.add_argument("--timeout", type = float, default = 300)
    ap.add_argument("--sizes", default = "30000,120000")
    ap.add_argument("--salt", default = str(int(time.time()) % 100000))
    ap.add_argument("--ntok", type = int, default = 120000)
    ap.add_argument("--max-tokens", type = int, default = 1024)
    ap.add_argument("--dir", default = "/srv/qwen5090/fast/exl3-nvme-tier")
    ap.add_argument("--cap-gb", type = float, default = 2.0)
    ap.add_argument("--n", type = int, default = 6)
    ap.add_argument("--size", type = int, default = 30000, help = "first-token: fill size to probe (0 = all)")
    ap.add_argument("--tag", default = "", help = "first-token: label for the arm (e.g. on-restart, off)")
    ap.add_argument("--fill", default = None, help = "first-token: fill.json to take the prompt from (default <out>)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok = True)
    fn = {"fill": cmd_fill, "wait-drained": cmd_wait_drained, "verify": cmd_verify, "decode": cmd_decode,
          "churn": cmd_churn, "first-token": cmd_first_token}[args.cmd]
    try:
        return fn(args)
    except Exception as e:  # setup problems are not a verdict
        print(f"ERROR {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
