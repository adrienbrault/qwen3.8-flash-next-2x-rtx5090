#!/usr/bin/env python3
"""latchain r1 model-level parity: sha256 of every target (and draft) forward's output over a fixed decode run.

The fork flags (EXL3_LC_QSA_FORK, EXL3_LC_GDN_FORK) only change CUDA-graph node dependencies, so there is no
kernel-level test for them; this script is their bitwise check (and a whole-model check of the other flags). It
loads the model exactly like probes/r465/profile_decode_events.py (Config/Model/Cache/Generator, draft model first,
MTP depth, quantized cache, gpu split), enqueues `batch` greedy jobs on a fixed `context`-token prompt, and records
the digest of every model.forward / draft_model.forward output (logits or draft state, byte for byte) from the
first decode iteration on, plus the streamed token ids. Flags are read from the environment at process start,
so run one process per arm:

  lc_model_parity.py --model M --batch 4 --draft 3 --out off.json          (flags 0)
  lc_model_parity.py --model M --batch 4 --draft 3 --out off2.json         (flags 0 again: the A/A control)
  EXL3_LC_...=1 lc_model_parity.py --model M --batch 4 --draft 3 --out on.json
  lc_model_parity.py --compare off.json off2.json on.json

--compare treats the first file as the reference and prints, per other file,
  MODEL-PARITY <file>: forwards identical a/b, tokens identical yes|no
and a final "MODEL-PARITY-SUMMARY aa=IDENTICAL|DIFFERENT arms=IDENTICAL|DIFFERENT". If the A/A control differs,
logits-level identity is not a usable test on this build (some served kernel is run-to-run nondeterministic) and
the gate falls back to token-level evidence (P1 sequence hashes, fn_greedy).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def tensors_of(x):
    import torch
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from tensors_of(y)
    elif isinstance(x, dict):
        for k in sorted(x, key=str):
            yield from tensors_of(x[k])


def digest(x):
    h = hashlib.sha256()
    n = 0
    for t in tensors_of(x):
        t = t.detach()
        h.update(f"{tuple(t.shape)}|{t.dtype}|".encode())
        if t.numel():
            h.update(t.contiguous().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes())
        n += 1
    return h.hexdigest()[:32], n


def run(a):
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.constants import PAGE_SIZE

    b, depth, context = a.batch, a.draft, a.context
    lc_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("EXL3_LC_")}
    out_budget = (a.steps + a.bootstrap + 16) * (depth + 1)
    per_job = ((context + out_budget + depth + 1 + PAGE_SIZE - 1) // PAGE_SIZE + 1) * PAGE_SIZE
    cache_tokens = max(b * per_job, ((a.max_chunk_size + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE)
    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    draft_model = Model.from_config(config, component="mtp") if depth else None
    bits = [int(x) for x in a.cache_quant.split(",")]
    cache_kw = dict(layer_type=CacheLayer_quant, k_bits=bits[0], v_bits=bits[-1])
    cache = Cache(model, max_num_tokens=cache_tokens, max_batch_size=b, max_history=depth, **cache_kw)
    draft_cache = Cache(draft_model, max_num_tokens=cache_tokens, **cache_kw) if depth else None
    load_kw = dict(progressbar=False, max_batch_size=b, max_chunk_size=a.max_chunk_size,
                   use_per_device=[float(x) for x in a.gpu_split.split(",")])
    if draft_model:
        draft_model.load(**load_kw)
    model.load(**load_kw)
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=b,
                          max_chunk_size=a.max_chunk_size, draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=depth, dynamic_draft_tokens=False)

    recording = [False]
    calls = []

    def wrap(m, tag):
        orig = m.forward

        def fwd(*args, **kw):
            out = orig(*args, **kw)
            if recording[0]:
                d, n = digest(out)
                calls.append(f"{tag}:{n}:{d}")
            return out
        m.forward = fwd
    wrap(model, "target")
    if draft_model:
        wrap(draft_model, "draft")

    body = ("Explain the engineering tradeoffs in a reliable database. Discuss indexing, transactions, "
            "replication, testing, and recovery. Give concrete examples and continue the analysis.\n")
    prompts = []
    for slot in range(b):
        prefix = tokenizer.encode(f"Independent request {slot}:\n", add_bos=True)
        ids = tokenizer.encode(body)
        need = context - prefix.shape[-1]
        prompts.append(torch.cat([prefix, ids.repeat(1, (need + ids.shape[-1] - 1) // ids.shape[-1])[:, :need]], dim=1))
    jobs = [Job(input_ids=p.clone(), max_new_tokens=out_budget, sampler=GreedySampler(), stop_conditions=[],
                identifier=i, seed=1234 + i) for i, p in enumerate(prompts)]
    generator.enqueue(jobs)
    tokens = {i: [] for i in range(b)}
    seen = set()
    for _ in range(a.bootstrap):
        for r in generator.iterate():
            if r["stage"] == "error":
                raise RuntimeError(str(r))
            if r["stage"] == "streaming":
                seen.add(r["identifier"])
                if r.get("token_ids") is not None:
                    tokens[r["identifier"]] += r["token_ids"].view(-1).tolist()
        if len(seen) == b:
            break
    if len(seen) != b:
        raise RuntimeError("not all jobs reached decode")
    recording[0] = True
    for _ in range(a.steps):
        for r in generator.iterate():
            if r["stage"] == "error":
                raise RuntimeError(str(r))
            if r.get("eos"):
                raise RuntimeError("a job ended inside the window")
            if r["stage"] == "streaming" and r.get("token_ids") is not None:
                tokens[r["identifier"]] += r["token_ids"].view(-1).tolist()
    torch.cuda.synchronize()
    recording[0] = False
    for j in jobs:
        generator.cancel(j)
    tok_digest = hashlib.sha256(json.dumps(tokens, sort_keys=True).encode()).hexdigest()[:32]
    res = {"batch": b, "draft": depth, "context": context, "steps": a.steps, "lc_env": lc_env,
           "calls": calls, "tokens_sha": tok_digest, "tokens_per_job": {i: len(v) for i, v in tokens.items()}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"lc_model_parity: b{b} d{depth} ctx{context}: {len(calls)} forwards recorded, tokens {tok_digest}, "
          f"env {lc_env} -> {a.out}")


def compare(files):
    ref = json.loads(Path(files[0]).read_text())
    # a forward whose output holds no tensor hashes to a constant and would compare equal without measuring
    # anything: refuse the comparison instead (every target forward must contribute >= 1 tensor)
    for f in files:
        calls = json.loads(Path(f).read_text())["calls"]
        empty = [c for c in calls if c.startswith("target:0:")]
        if not calls or empty or not any(c.startswith("target:") for c in calls):
            print(f"MODEL-PARITY {f}: UNUSABLE ({len(calls)} forwards, {len(empty)} target forwards without tensors)")
            print("MODEL-PARITY-SUMMARY aa=DIFFERENT arms=UNUSABLE")
            return 1
    verdicts = []
    for f in files[1:]:
        o = json.loads(Path(f).read_text())
        n = min(len(ref["calls"]), len(o["calls"]))
        same = sum(x == y for x, y in zip(ref["calls"][:n], o["calls"][:n]))
        full = same == len(ref["calls"]) == len(o["calls"]) and len(ref["calls"]) > 0
        tok = ref["tokens_sha"] == o["tokens_sha"]
        first = next((i for i in range(n) if ref["calls"][i] != o["calls"][i]), None)
        print(f"MODEL-PARITY {f}: forwards identical {same}/{len(ref['calls'])} (arm has {len(o['calls'])}), "
              f"tokens identical {'yes' if tok else 'no'}"
              + (f", first differing forward #{first} ({ref['calls'][first].split(':')[0]})" if first is not None else ""))
        verdicts.append(full and tok)
    aa = verdicts[0] if verdicts else False
    arms = all(verdicts[1:]) if len(verdicts) > 1 else True
    print(f"MODEL-PARITY-SUMMARY aa={'IDENTICAL' if aa else 'DIFFERENT'} arms={'IDENTICAL' if arms else 'DIFFERENT'}")
    return 0 if aa and arms else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--draft", type=int, default=3)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--bootstrap", type=int, default=64)
    ap.add_argument("--max-chunk-size", type=int, default=2048)
    ap.add_argument("--cache-quant", default="8,8")
    ap.add_argument("--gpu-split", default="30,30")
    ap.add_argument("--compare", nargs="+", help="reference.json aa_control.json arm.json ...")
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(a.compare))
    assert a.model and a.out, "--model and --out required"
    t0 = time.time()
    run(a)
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
