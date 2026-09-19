#!/usr/bin/env python3
"""
Draft-chain identity + chain microbenchmark on the real model (operator-run, in the built image, served
container stopped: it loads the model itself; ~3-4 min including load).

Loads the checkpoint the way TabbyAPI does for the daily (MTP draft component first on its own autosplit, then
the trunk on the manual split, Q8 caches, 4 slots, draft policy [[4,3],[8,1]]), then runs the SAME fixed prompt
set, greedy, through one fresh Generator per arm:

  off     EXL3_MTP_DEVICE_DRAFT=0                                  (today's daily host chain)
  pruned  EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1
  full    EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1                 (r4 unpruned mirror, 1.27 GB; skipped
                                                                    unless the head card has room)

Arms are selected by flipping the module globals the code reads at call time (one model load for all arms).
Every other flag comes from the environment and must match the daily (the script prints and checks the
group-I trio). Per arm it records, for c1 (prompts one at a time) and c4 (four at once):
  - every draft window the chain returned (job ids, depth, drafted token IDs)
  - per-job output token IDs, accepted/rejected draft counts, draft_stats (position, window, accepted)
  - how often the verified token fell outside the pruned rows (host-fallback count)
and then a timed pass: wall time of each iterate_draftmodel_mtp_gen call with both cards synchronized
before and after (chain latency, c1 and c4), plus unsynchronized decode tok/s.

PASS = draft windows, output tokens and accept counts identical between off and pruned (and full, if run).
Exit status 0 = PASS. Writes <out>/chain-<arm>.json and <out>/chain-summary.json.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--model", default = "/models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab")
p.add_argument("--cache", type = int, default = 131072, help = "tokens per cache (daily: 786,432; smaller leaves room for the full arm)")
p.add_argument("--gpu-split", default = "30,30")
p.add_argument("--chunk", type = int, default = 2048)
p.add_argument("--max-new", type = int, default = 160)
p.add_argument("--bench-new", type = int, default = 384)
p.add_argument("--arms", default = "off,pruned,full")
p.add_argument("--out", default = "/tmp/chain-identity")
args = p.parse_args()
os.makedirs(args.out, exist_ok = True)

for k in ("EXL3_DRAFT_PINNED_STAGING", "EXL3_BATCH_VERIFY", "EXL3_MTP_HEAD_N"):
    print(f"{k}={os.environ.get(k)}")
if os.environ.get("EXL3_MTP_HEAD_N") in (None, "", "0"):
    sys.exit("EXL3_MTP_HEAD_N must be set (the pruned mirror needs the pruned head); pass the daily EXTRA_ENV")

from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
import exllamav3.generator.generator as gen_mod
import exllamav3.modules.embedding as emb_mod
import exllamav3.modules.embedding_pruned as ep

PROMPTS = [
    ("code", "Write a Python function that parses an ISO-8601 duration string such as P3DT4H5M into seconds, with tests."),
    ("prose", "Describe the history of the lighthouse at Alexandria in three paragraphs."),
    ("zh", "请用中文详细解释快速排序算法的原理，并给出时间复杂度分析。"),
    ("ja", "日本の四季について、それぞれの季節の特徴を詳しく説明してください。"),
    ("ru", "Объясни, как работает протокол TCP, включая установку соединения и управление перегрузкой."),
    ("math", "Solve step by step: a train leaves at 14:05 at 87 km/h, another at 14:50 at 112 km/h on the same track. When does the second catch up?"),
    ("json", "Return a JSON object describing three fictional cities with fields name, population, founded, landmarks (array)."),
    ("emoji", "Write a short upbeat product announcement for a coffee app, using plenty of emoji 🎉☕🚀 and hashtags."),
]


def chat(text):
    # ChatML framing with real special tokens, so generation starts with the reasoning-open special token
    # (an ID outside the pruned rows) and the out-of-set path is exercised by real verified tokens
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


config = Config.from_directory(args.model)
model = Model.from_config(config)
draft = Model.from_config(config, component = "mtp")
tokenizer = Tokenizer.from_config(config)
batch_args = {"max_batch_size": 4, "max_history": 3}
cache = Cache(model, max_num_tokens = args.cache, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8, **batch_args)
draft_cache = Cache(draft, max_num_tokens = args.cache, layer_type = CacheLayer_quant, k_bits = 8, v_bits = 8, **batch_args)
t_load = time.time()
draft.load(use_per_device = None)
model.load(use_per_device = [float(x) for x in args.gpu_split.split(",")], max_chunk_size = args.chunk, max_batch_size = 4)
print(f"loaded in {time.time() - t_load:.1f} s")
lm = model.modules[model.logit_layer_idx]
print(f"lm_head device {lm.device}; draft input layer device {draft.modules[0].device}; "
      f"embedding device {model.modules[0].embedding.weight.device}")

prompt_ids = [(name, tokenizer.encode(chat(t), encode_special_tokens = True)) for name, t in PROMPTS]


def set_arm(arm):
    gen_mod._MTP_DEVICE_DRAFT = arm in ("pruned", "full")
    emb_mod._EMBED_GPU = arm in ("pruned", "full")
    emb_mod._EMBED_GPU_PRUNED = arm == "pruned"
    gen_mod._EMBED_GPU_PRUNED = arm == "pruned"
    e = model.modules[0]
    e._gpu_mirror = None
    e._gpu_mirror_device = None
    e._pruned_mirror = None
    torch.cuda.empty_cache()


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def make_gen(record, timed):
    g = Generator(
        model = model, cache = cache, draft_model = draft, draft_cache = draft_cache, tokenizer = tokenizer,
        max_batch_size = 4, max_chunk_size = args.chunk, recurrent_cache_size = 512 * 1024**2,
        num_draft_tokens = 3, num_draft_tokens_by_batch = [(4, 3), (8, 1)], record_draft_stats = True,
    )
    orig = g.iterate_draftmodel_mtp_gen

    def wrapped(results):
        ids = [j.identifier for j in g.active_jobs if j.is_prefill_done()]
        if timed is not None:
            sync_all()
            t0 = time.perf_counter()
        r = orig(results)
        if timed is not None:
            sync_all()
            timed.append((len(ids), time.perf_counter() - t0))
        if r is not None and record is not None:
            record.append({"jobs": ids, "window": int(r.shape[1]), "draft": r[:len(ids)].tolist()})
        return r

    g.iterate_draftmodel_mtp_gen = wrapped
    return g


def run(g, batch, max_new):
    jobs = {}
    for name, ids in batch:
        j = Job(input_ids = ids, max_new_tokens = max_new, sampler = ArgmaxSampler(), identifier = name,
                stop_conditions = [], decode_special_tokens = True)
        jobs[name] = j
        g.enqueue(j)
    out = {n: [] for n, _ in batch}
    fin = {}
    t0 = time.perf_counter()
    while g.num_remaining_jobs():
        for r in g.iterate():
            if r.get("stage") != "streaming":
                continue
            tid = r.get("token_ids")
            if tid is not None:
                out[r["identifier"]] += tid.view(-1).tolist()
            if r.get("eos"):
                fin[r["identifier"]] = {
                    "new_tokens": r.get("new_tokens"),
                    "accepted": r.get("accepted_draft_tokens"),
                    "rejected": r.get("rejected_draft_tokens"),
                }
    dt = time.perf_counter() - t0
    res = {}
    for n in out:
        res[n] = {
            "tokens": out[n],
            "sha": hashlib.sha256(json.dumps(out[n]).encode()).hexdigest()[:16],
            **fin.get(n, {}),
            "draft_stats": [list(x) for x in jobs[n].draft_stats],
        }
    return res, dt


def arm_pass(arm):
    set_arm(arm)
    free0 = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    before = dict(ep.stats)
    rec = []
    g = make_gen(rec, None)
    rep = {"arm": arm, "pruned_embed": None}
    pe = getattr(g, "_pruned_embed", None)
    if pe is not None:
        rep["pruned_embed"] = {"rows": int(pe[0].numel()), "device": str(pe[1])}
    c1 = {}
    c1_time, c1_tok = 0.0, 0
    for item in prompt_ids:
        r, dt = run(g, [item], args.max_new)
        c1.update(r)
        c1_time += dt
        c1_tok += sum(len(v["tokens"]) for v in r.values())
    c4 = {}
    c4_time, c4_tok = 0.0, 0
    for k in range(0, len(prompt_ids), 4):
        r, dt = run(g, prompt_ids[k:k + 4], args.max_new)
        c4.update({f"c4:{n}": v for n, v in r.items()})
        c4_time += dt
        c4_tok += sum(len(v["tokens"]) for v in r.values())
    rep.update({
        "c1": c1, "c4": c4, "draft_windows": rec,
        "tok_s": {"c1": round(c1_tok / c1_time, 1), "c4": round(c4_tok / c4_time, 1)},
        "host_fallback": {k: ep.stats[k] - before[k] for k in ep.stats},
    })
    sync_all()
    free1 = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    rep["vram_free_mib_before_after"] = [[round(a / 2**20), round(b / 2**20)] for a, b in zip(free0, free1)]
    del g
    # timed pass: chain latency with both cards synchronized around each draft call
    timed = []
    g = make_gen(None, timed)
    run(g, [prompt_ids[0]], args.bench_new)
    t_c1 = [t for b, t in timed if b == 1]
    timed.clear()
    run(g, prompt_ids[:4], args.bench_new // 2)
    t_c4 = [t for b, t in timed if b == 4]
    del g

    def summ(ts):
        if not ts:
            return None
        ts = sorted(ts)
        return {"n": len(ts), "median_us": round(statistics.median(ts) * 1e6, 1), "p90_us": round(ts[int(0.9 * (len(ts) - 1))] * 1e6, 1)}

    rep["chain"] = {"c1": summ(t_c1), "c4": summ(t_c4)}
    with open(os.path.join(args.out, f"chain-{arm}.json"), "w") as f:
        json.dump(rep, f)
    print(f"[{arm}] tok/s c1 {rep['tok_s']['c1']} c4 {rep['tok_s']['c4']}; chain c1 {rep['chain']['c1']} c4 {rep['chain']['c4']}; "
          f"host fallback {rep['host_fallback']}; pruned_embed {rep['pruned_embed']}; VRAM free MiB {rep['vram_free_mib_before_after']}")
    return rep


arms = [a for a in args.arms.split(",") if a]
reports = {}
for arm in arms:
    if arm == "full":
        head_free = torch.cuda.mem_get_info(torch.device(lm.device))[0]
        need = model.modules[0].embedding.weight.numel() * 2 + (256 << 20)
        if head_free < need:
            print(f"[full] skipped: {head_free / 2**20:.0f} MiB free on {lm.device}, needs {need / 2**20:.0f}")
            continue
    reports[arm] = arm_pass(arm)


def key(rep):
    return (
        rep["draft_windows"],
        {n: (v["tokens"], v.get("accepted"), v.get("rejected"), v["draft_stats"]) for n, v in {**rep["c1"], **rep["c4"]}.items()},
    )


summary = {"arms": list(reports), "compare": {}, "ok": True}
ref = reports.get("off")
for arm, rep in reports.items():
    if arm == "off" or ref is None:
        continue
    kr, ka = key(ref), key(rep)
    same_windows = kr[0] == ka[0]
    same_jobs = kr[1] == ka[1]
    diff_jobs = [n for n in kr[1] if kr[1][n] != ka[1].get(n)]
    summary["compare"][arm] = {"draft_windows_identical": same_windows, "jobs_identical": same_jobs, "differing_jobs": diff_jobs,
                               "n_windows": len(ka[0])}
    summary["ok"] = summary["ok"] and same_windows and same_jobs
    print(f"[{arm} vs off] draft windows identical: {same_windows} ({len(ka[0])} windows); "
          f"tokens+accept counts identical: {same_jobs} {diff_jobs or ''}")
pr = reports.get("pruned")
if pr is not None:
    summary["pruned_host_fallback"] = pr["host_fallback"]
    if pr["pruned_embed"] is None:
        summary["ok"] = False
        print("FAIL: pruned arm did not activate the pruned mirror (see the EXL3_EMBED_GPU_PRUNED warning above)")
    if pr["host_fallback"].get("host_calls", 0) == 0:
        print("NOTE: no verified token fell outside the pruned rows; the out-of-set path was not exercised end to end")
summary["accept"] = {arm: {"c1": sum(v.get("accepted") or 0 for v in r["c1"].values()),
                           "c4": sum(v.get("accepted") or 0 for v in r["c4"].values())} for arm, r in reports.items()}
summary["chain"] = {arm: r["chain"] for arm, r in reports.items()}
summary["tok_s"] = {arm: r["tok_s"] for arm, r in reports.items()}
with open(os.path.join(args.out, "chain-summary.json"), "w") as f:
    json.dump(summary, f, indent = 1)
print("PASS" if summary["ok"] else "FAIL")
sys.exit(0 if summary["ok"] else 1)
