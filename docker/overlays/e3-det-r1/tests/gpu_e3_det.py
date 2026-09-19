#!/usr/bin/env python3
"""
GPU check for deterministic E3 grouped MoE prefill (EXL3_MOE_PREFILL_E3_DET, e3-det r1). Run inside the e3-det image
with the server stopped and both GPUs free, with the daily's EXTRA_ENV passed as -e flags (EXL3_MOE_PREFILL_E3=1 among
them; the DET flag itself is switched in-process, so do NOT pass EXL3_MOE_PREFILL_E3_DET), e.g.

  sudo docker run --rm --gpus all --ipc=host --shm-size=16g \
    -v /srv/qwen5090/models:/models:ro -v /srv/qwen5090/.exl3cache:/exl3-cache \
    -e TRITON_CACHE_DIR=/exl3-cache -e EXLLAMAV3_TUNE_CACHE=/exl3-cache \
    -e EXL3_HOST_GAP_REWIND=1 -e EXL3_HC_MIX_V2=1 -e EXL3_HC_MIX_V2_MIN_R=1 -e EXL3_LS_PREFILL_PIPELINE=1 \
    -e EXL3_MOE_COOP_V2=1 -e EXL3_SHARED_EXPERT_OVERLAP=1 -e EXL3_DRAFT_PINNED_STAGING=1 -e EXL3_BATCH_VERIFY=1 \
    -e EXL3_MTP_HEAD_N=65536 -e EXL3_MOE_PREFILL_E3=1 -e EXL3_HC_MIX_V2_INT8=1 \
    -v /srv/qwen5090/results/<dir>:/out --entrypoint python3 tabbyapi:e3-det-r1 \
    /opt/e3-det-r1/tests/gpu_e3_det.py --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab \
    --kernel --e2e --out /out/gpu_e3_det_p1.json

and then a second process for the cross-process check of the kernel phase:

    ... /opt/e3-det-r1/tests/gpu_e3_det.py --model ... --kernel --compare /out/gpu_e3_det_p1.json --out /out/gpu_e3_det_p2.json

One model load mirroring the served TabbyAPI path (as gpu_tip_replay.py: MTP draft, 8,8 caches, gpu_split 30,30, chunk
2048, 4 slots, draft 3 with policy [[4,3],[8,1]]; vision not loaded).

Phase K (--kernel), per routed-MoE layer with a K=2, a K=3 and a K=4 projection (the first eligible layer of each), at
512 / 1,024 / 2,048 rows, on one fixed input (CPU-seeded) and the layer's real router, shared experts detached:
  off     E3 off (served fused / batched-reconstruct tiers, FUSED_DET)
  atomic  E3 on, DET off (today's daily)
  det     E3 on, DET on
  * det: --repeats runs must be bitwise equal; 5 more runs with a concurrent GEMM on a side stream (different CTA
    placement/timing) must equal them too. The output's sha256 goes to the JSON (--compare checks it across processes).
  * atomic: the same number of repeats, distinct-output count reported (expected > 1; informational).
  * errors: det vs atomic (only the summation order differs, per contribution the arithmetic is the atomic kernel's:
    expected NRMSE ~1e-7, gate < 1e-5), det vs off and atomic vs off (tiling differences, ~1e-3, reported).
  * CUDA-event timing (--warmup, --iters) of off / atomic / det and peak-allocation deltas of atomic vs det.
Phase E (--e2e), like gpu_tip_replay phase X: a cold 8k and a cold 30k prompt (this image's exllamav3 sources), greedy,
--e2e-reps cold runs each with DET on, then the same with DET off (atomic E3). DET reps must all be identical; the
atomic reps' first differing token is reported (the baseline's non-determinism; it may happen to agree).

Exit code 0 PASS, 1 FAIL (reasons in the JSON "verdict").
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, CacheLayer_quant
from exllamav3.generator.sampler import GreedySampler
from exllamav3.modules import BlockSparseMLP
from exllamav3.modules import block_sparse_mlp as bsm
from exllamav3.ext import exllamav3_ext as ext


def parse(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--out", required = True)
    ap.add_argument("--kernel", action = "store_true")
    ap.add_argument("--e2e", action = "store_true")
    ap.add_argument("--compare", default = None, help = "JSON of an earlier --kernel run: det hashes must match")
    ap.add_argument("--cache-tokens", type = int, default = 65536)
    ap.add_argument("--split", default = "30,30")
    ap.add_argument("--kv-bits", default = "8,8")
    ap.add_argument("--slots", type = int, default = 4)
    ap.add_argument("--draft", type = int, default = 3)
    ap.add_argument("--rows", default = "512,1024,2048")
    ap.add_argument("--repeats", type = int, default = 20)
    ap.add_argument("--warmup", type = int, default = 5)
    ap.add_argument("--iters", type = int, default = 20)
    ap.add_argument("--e2e-lengths", default = "8000,30000")
    ap.add_argument("--e2e-reps", type = int, default = 3)
    ap.add_argument("--gen-tokens", type = int, default = 256)
    args = ap.parse_args(argv)
    assert args.kernel or args.e2e, "pass --kernel and/or --e2e"
    return args


# ---------------------------------------------------------------------------------------------------------------------
# Load (mirrors ExllamaV3Container.create() / create_cache() / load_model_sync() / create_generator())
# ---------------------------------------------------------------------------------------------------------------------

def load(args):
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    config.infer_params.moe_cpu_offload = 0
    config.infer_params.vision_pinned = False
    draft = Model.from_config(config, component = "mtp") if args.draft > 0 else None
    policy = [(4, args.draft), (8, 1)] if args.draft > 0 else None
    max_history = args.draft if args.draft > 0 else 0
    if policy:
        max_history = max(max_history, *(d for _, d in policy))
    k_bits, v_bits = (int(x) for x in args.kv_bits.split(","))
    cache = Cache(model, max_num_tokens = args.cache_tokens, layer_type = CacheLayer_quant,
                  k_bits = k_bits, v_bits = v_bits, max_batch_size = args.slots, max_history = max_history)
    draft_cache = Cache(draft, max_num_tokens = args.cache_tokens, layer_type = CacheLayer_quant,
                        k_bits = 8, v_bits = 8, max_batch_size = args.slots,
                        max_history = max_history) if draft is not None else None
    gpu_split = [float(x) for x in args.split.split(",")]
    if draft is not None:
        draft.load(reserve_per_device = None, use_per_device = None)
    model.load(tensor_p = False, reserve_per_device = None, use_per_device = gpu_split,
               max_chunk_size = 2048, max_batch_size = args.slots)
    gen = Generator(
        model = model, cache = cache, draft_model = draft, draft_cache = draft_cache, tokenizer = tokenizer,
        max_batch_size = args.slots, max_chunk_size = 2048, recurrent_cache_size = 4096 * 1024**2,
        cpu_cache_size = 0, num_draft_tokens = args.draft if args.draft > 0 else None,
        num_draft_tokens_by_batch = policy, dynamic_draft_tokens = False, ngram_match_min = 0,
    )
    return config, model, tokenizer, gen


# ---------------------------------------------------------------------------------------------------------------------
# Phase K
# ---------------------------------------------------------------------------------------------------------------------

def k_signature(layer):
    return (layer.multi_gate.K, layer.multi_up.K, layer.multi_down.K)


def eligible(layer):
    return (
        layer.device is not None and layer.device.type == "cuda"
        and layer.num_experts == 512 and layer.num_local_experts == 512
        and layer.num_experts_per_tok == 10 and layer.expert_size == 2560
        and layer.intermediate_size_padded == 640 and layer.support_fused
        and layer.fused_mode_buffers is not None and layer.multi_gate is not None
        and all(k in (2, 3, 4) for k in k_signature(layer))
        and layer.multi_gate.mul1 and layer.multi_up.mul1 and layer.multi_down.mul1
    )


def errors(ref, cand):
    d = (ref.double() - cand.double())
    rmse = d.square().mean().sqrt().item()
    rms = ref.double().square().mean().sqrt().item()
    return {"max_abs": d.abs().max().item(), "rmse": rmse, "nrmse": rmse / max(rms, 1e-30),
            "equal": bool(torch.equal(ref, cand)), "finite": bool(torch.isfinite(cand).all().item())}


def sha(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    v = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing = True)
        b = torch.cuda.Event(enable_timing = True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        v.append(a.elapsed_time(b))
    return {"median_ms": statistics.median(v), "min_ms": min(v), "max_ms": max(v), "samples_ms": v}


def set_mode(mode, thin = None):
    bsm.MOE_PREFILL_E3 = mode in ("atomic", "det")
    bsm.MOE_PREFILL_E3_DET = mode == "det"
    if thin is not None:
        bsm.MOE_PREFILL_E3_THIN_ROWS = thin


@torch.inference_mode()
def kernel_case(layer, rows, args, seed):
    dev = layer.device
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn((rows, 2560), generator = g) * 0.5).half().to(dev)
    saved = (bsm.MOE_PREFILL_E3, bsm.MOE_PREFILL_E3_DET, bsm.MOE_PREFILL_E3_THIN_ROWS,
             layer.shared_experts, layer.shared_gate)
    layer.shared_experts = None
    layer.shared_gate = None
    rec = {"rows": rows, "seed": seed}
    try:
        def run(mode):
            set_mode(mode)
            return layer.forward(x, {}).clone()

        sel, _ = layer.routing_fn(rows, layer.routing_cfg, x, {})
        counts = torch.bincount(sel.reshape(-1), minlength = 512)
        thin = bsm.MOE_PREFILL_E3_THIN_ROWS
        rec["routing"] = {"mean": counts.float().mean().item(), "max": int(counts.max()),
                          "experts_fat": int((counts > thin).sum()),
                          "assign_fat": int(counts[counts > thin].sum()), "assign": int(sel.numel())}

        off = run("off")
        det = [run("det") for _ in range(args.repeats)]
        torch.cuda.synchronize(dev)
        det_equal = all(torch.equal(det[0], d) for d in det[1:])
        # perturbed: a concurrent GEMM on a side stream while det runs
        side = torch.cuda.Stream(dev)
        # 4096^2 with a preallocated output: 64 MiB total. 8192^2 plus a fresh 128 MiB result per mm hit OOM at
        # the third case with the served-shape caches loaded (R531 try 1, 90 MiB free on cuda:0).
        big = torch.randn((4096, 4096), device = dev, dtype = torch.half)
        big_out = torch.empty_like(big)
        pert = []
        for _ in range(5):
            with torch.cuda.stream(side):
                for _ in range(16):
                    torch.mm(big, big, out = big_out)
            pert.append(run("det"))
        torch.cuda.synchronize(dev)
        del big, big_out
        pert_equal = all(torch.equal(det[0], p) for p in pert)
        atomic = [run("atomic") for _ in range(args.repeats)]
        torch.cuda.synchronize(dev)
        atomic_distinct = len({sha(a) for a in atomic})

        rec["det_sha256"] = sha(det[0])
        rec["det_repeats_equal"] = det_equal
        rec["det_perturbed_equal"] = pert_equal
        rec["atomic_distinct_outputs"] = atomic_distinct
        rec["err_det_vs_atomic"] = errors(atomic[0], det[0])
        rec["err_det_vs_off"] = errors(off, det[0])
        rec["err_atomic_vs_off"] = errors(off, atomic[0])
        rec["err_atomic_rep"] = errors(atomic[0], atomic[1])

        mem = {}
        for mode in ("atomic", "det"):
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            run(mode)
            torch.cuda.synchronize(dev)
            mem[mode] = torch.cuda.max_memory_allocated(dev) - base
        rec["peak_alloc_bytes"] = mem

        t = {m: timed(lambda m = m: run(m), args.warmup, args.iters) for m in ("off", "atomic", "det")}
        rec["timing"] = {m: {k: v for k, v in r.items() if k != "samples_ms"} for m, r in t.items()}
        rec["timing_samples_ms"] = {m: r["samples_ms"] for m, r in t.items()}
        rec["det_over_atomic"] = t["det"]["median_ms"] / t["atomic"]["median_ms"]
        rec["atomic_over_off"] = t["atomic"]["median_ms"] / t["off"]["median_ms"]
    finally:
        (bsm.MOE_PREFILL_E3, bsm.MOE_PREFILL_E3_DET, bsm.MOE_PREFILL_E3_THIN_ROWS,
         layer.shared_experts, layer.shared_gate) = saved
    return rec


def phase_k(model, args, record, reasons):
    layers = [m for m in model if isinstance(m, BlockSparseMLP) and eligible(m)]
    record["kernel"] = {"eligible_layers": len(layers), "thin_rows": bsm.MOE_PREFILL_E3_THIN_ROWS, "layers": []}
    picked = []
    for k in (2, 3, 4):
        lay = next((l for l in layers if k in k_signature(l) and l not in picked), None)
        if lay is None:
            lay = next((l for l in layers if k in k_signature(l)), None)
        if lay is None:
            # the 2.50 bpw pack's target layers are K=2 (12-36) and K=3 (0-11, 37-47); only the MTP layer is K=4
            # (R531 try 4). A missing K=4 is recorded, not a failure; K=2 and K=3 stay required.
            if k == 4:
                record["kernel"]["missing_k4"] = True
            else:
                reasons.append(f"no eligible layer with a K={k} projection")
            continue
        picked.append(lay)
        rows_list = [int(r) for r in args.rows.split(",")]
        lr = {"target_k": k, "layer": lay.key, "device": str(lay.device), "signature": list(k_signature(lay)),
              "cases": []}
        record["kernel"]["layers"].append(lr)
        with torch.cuda.device(lay.device):
            for rows in rows_list:
                c = kernel_case(lay, rows, args, seed = 1000 * k + rows)
                lr["cases"].append(c)
                print(json.dumps({"k": k, "layer": lay.key, "rows": rows, "det_equal": c["det_repeats_equal"],
                                  "perturbed_equal": c["det_perturbed_equal"],
                                  "atomic_distinct": c["atomic_distinct_outputs"],
                                  "nrmse_det_atomic": c["err_det_vs_atomic"]["nrmse"],
                                  "nrmse_det_off": c["err_det_vs_off"]["nrmse"],
                                  "ms_off_atomic_det": [round(c["timing"][m]["median_ms"], 4) for m in ("off", "atomic", "det")],
                                  "det_over_atomic": round(c["det_over_atomic"], 4),
                                  "peak_alloc": c["peak_alloc_bytes"]}), flush = True)
                tag = f"K{k} {lay.key} rows {rows}"
                if not c["det_repeats_equal"]:
                    reasons.append(f"{tag}: det repeats differ")
                if not c["det_perturbed_equal"]:
                    reasons.append(f"{tag}: det differs under a concurrent side-stream GEMM")
                if not (c["err_det_vs_atomic"]["finite"] and c["err_det_vs_off"]["finite"]):
                    reasons.append(f"{tag}: non-finite output")
                if not c["err_det_vs_atomic"]["nrmse"] < 1e-5:
                    reasons.append(f"{tag}: det vs atomic NRMSE {c['err_det_vs_atomic']['nrmse']:.3g} >= 1e-5 "
                                   "(expected summation-order noise only: slot/epilogue bug)")
                if c["peak_alloc_bytes"]["det"] > c["peak_alloc_bytes"]["atomic"] + (4 << 20):
                    reasons.append(f"{tag}: det peak allocation exceeds atomic by more than 4 MiB")
            torch.cuda.synchronize()
    if args.compare:
        with open(args.compare) as f:
            prev = json.load(f)
        mine = {(l["layer"], c["rows"]): c["det_sha256"] for l in record["kernel"]["layers"] for c in l["cases"]}
        theirs = {(l["layer"], c["rows"]): c["det_sha256"] for l in prev["kernel"]["layers"] for c in l["cases"]}
        cmp = {f"{k[0]}@{k[1]}": (mine.get(k) == theirs.get(k)) for k in sorted(set(mine) | set(theirs))}
        record["kernel"]["cross_process"] = {"against": args.compare, "equal": cmp}
        bad = [k for k, v in cmp.items() if not v]
        if bad:
            reasons.append(f"det output differs across processes: {bad}")
        print(f" -- cross-process det hashes: {'EQUAL' if not bad else 'DIFFER ' + str(bad)}", flush = True)


# ---------------------------------------------------------------------------------------------------------------------
# Phase E
# ---------------------------------------------------------------------------------------------------------------------

def corpus_ids(tokenizer, need):
    import exllamav3
    root = os.path.dirname(exllamav3.__file__)
    files = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".py"):
                files.append(os.path.join(d, f))
    parts = []
    for p in sorted(files):
        with open(p) as fh:
            parts.append(f"# file: {os.path.relpath(p, root)}\n" + fh.read())
    ids = tokenizer.encode("\n".join(parts), add_bos = False)[0]
    assert ids.numel() >= need, f"corpus has {ids.numel()} tokens, need {need}"
    return ids


def reset(gen):
    assert gen.num_remaining_jobs() == 0
    rc = gen.recurrent_cache
    if rc is not None:
        for k in list(rc.keys()):
            rc.pop(k)
        rc.update_total_size()
        if hasattr(rc, "tip_meta"):
            rc.tip_meta.clear()
    gen.pagetable.reset_page_table()


def run_job(gen, ids, max_new, stop):
    job = Job(input_ids = ids, max_new_tokens = max_new, sampler = GreedySampler(), stop_conditions = stop)
    gen.enqueue(job)
    eos = None
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "error":
                raise r["error"]
            if r.get("stage") == "streaming" and r.get("eos") and r.get("job") is job:
                eos = r
    assert eos is not None, "job ended without an EOS result"
    full = job.sequences[0].sequence_ids.torch()[0].clone()
    return full[ids.shape[-1]:], {"prompt_tokens": ids.shape[-1], "cached_tokens": eos.get("cached_tokens"),
                                   "gen_tokens": int(full.numel() - ids.shape[-1]),
                                   "eos_reason": eos.get("eos_reason"),
                                   "seconds": round(time.perf_counter() - t0, 3)}


def first_diff(a, b):
    n = min(a.numel(), b.numel())
    neq = (a[:n] != b[:n]).nonzero()
    if neq.numel():
        return int(neq[0])
    return None if a.numel() == b.numel() else n


def phase_e(config, tokenizer, gen, args, record, reasons):
    stop = list(config.eos_token_id_list) if getattr(config, "eos_token_id_list", None) else [config.eos_token_id]
    lengths = [int(v) for v in args.e2e_lengths.split(",")]
    ids = corpus_ids(tokenizer, max(lengths))
    header = tokenizer.encode("System: you are a coding agent. Read the repository below and summarize it.\n\n")[0]
    tail = tokenizer.encode("\n\nAssistant:")[0]
    record["e2e"] = {"gen_tokens": args.gen_tokens, "reps": args.e2e_reps, "prompts": []}
    # warm-up (autotune, pipeline setup) with DET on, discarded
    set_mode("det")
    reset(gen)
    run_job(gen, torch.cat([header, ids[:600], tail]).unsqueeze(0), 16, stop)
    for n in lengths:
        prompt = torch.cat([header, ids[:n], tail]).unsqueeze(0)
        pr = {"prompt_tokens": int(prompt.shape[-1])}
        for mode in ("det", "atomic"):
            set_mode(mode)
            outs, recs = [], []
            for _ in range(args.e2e_reps):
                reset(gen)
                o, r = run_job(gen, prompt, args.gen_tokens, stop)
                outs.append(o)
                recs.append(r)
            pr[mode] = {
                "runs": recs,
                "hashes": [hashlib.sha256(o.numpy().tobytes()).hexdigest()[:16] for o in outs],
                "first_diff_vs_rep0": [first_diff(outs[0], o) for o in outs[1:]],
            }
            pr[mode + "_out0"] = outs[0]
        pr["det_vs_atomic_first_diff"] = first_diff(pr.pop("det_out0"), pr.pop("atomic_out0"))
        record["e2e"]["prompts"].append(pr)
        print(json.dumps({"prompt_tokens": pr["prompt_tokens"],
                          "det_first_diff": pr["det"]["first_diff_vs_rep0"],
                          "atomic_first_diff": pr["atomic"]["first_diff_vs_rep0"],
                          "det_vs_atomic": pr["det_vs_atomic_first_diff"],
                          "det_seconds": [r["seconds"] for r in pr["det"]["runs"]],
                          "atomic_seconds": [r["seconds"] for r in pr["atomic"]["runs"]]}), flush = True)
        if any(d is not None for d in pr["det"]["first_diff_vs_rep0"]):
            reasons.append(f"e2e {pr['prompt_tokens']} tokens: DET reps differ {pr['det']['first_diff_vs_rep0']}")
        if any(r["cached_tokens"] for r in pr["det"]["runs"] + pr["atomic"]["runs"]):
            reasons.append(f"e2e {pr['prompt_tokens']} tokens: a 'cold' run reused cached tokens (reset failed)")


def main():
    args = parse()
    reasons = []
    assert hasattr(bsm, "MOE_PREFILL_E3_DET"), "block_sparse_mlp has no MOE_PREFILL_E3_DET: overlay not installed"
    assert hasattr(ext, "exl3_moe_prefill_e3_det") and hasattr(ext, "exl3_moe_prefill_e3_det_reduce"), \
        "extension lacks the e3-det symbols"
    record = {"env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
              "e3_env": bsm.MOE_PREFILL_E3, "det_env": bsm.MOE_PREFILL_E3_DET, "args": vars(args)}
    if os.environ.get("EXL3_MOE_PREFILL_E3_DET"):
        print(" -- note: EXL3_MOE_PREFILL_E3_DET is set in the environment; this script switches it in-process anyway",
              flush = True)
    t0 = time.time()
    config, model, tokenizer, gen = load(args)
    record["load_seconds"] = round(time.time() - t0, 1)
    try:
        if args.kernel:
            phase_k(model, args, record, reasons)
        if args.e2e:
            phase_e(config, tokenizer, gen, args, record, reasons)
    except Exception as ex:
        reasons.append(f"exception: {type(ex).__name__}: {ex}")
        raise
    finally:
        record["verdict"] = {"pass": not reasons, "reasons": reasons}
        with open(args.out, "w") as f:
            json.dump(record, f, indent = 1)
        print(f" -- gpu_e3_det: {'PASS' if not reasons else 'FAIL'} {reasons}", flush = True)
    sys.exit(0 if not reasons else 1)


if __name__ == "__main__":
    main()
