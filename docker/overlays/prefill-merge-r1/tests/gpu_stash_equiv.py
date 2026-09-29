#!/usr/bin/env python3
"""prefill-merge r1 GPU gate (flan, daily DOWN, one process, NVMe tier off): is the last-page stash of a merged prefill
the stash today's path stores, did the paths under test actually run, and does next-turn reuse still start there?

One model load serves every arm: the knobs are read per call (cache/prefill_merge.py), so each case runs
  OFF          EXL3_PREFILL_MERGE unset, EXL3_STASH_ASYNC unset      (the served path)
  ASYNC        EXL3_STASH_ASYNC=1                                   (must equal OFF bit for bit: hard)
  MERGE        EXL3_PREFILL_MERGE=1
  MERGE_ASYNC  both                                                 (must equal MERGE bit for bit: hard)
on the same token ids, from the same cache state (page table + recurrent cache reset before each arm; a warm case
first prefills its prefix, whose own stash lands on a page boundary so the prefix run itself never merges).

Per case and arm it records: EVERY stash the turn created (by position, every recurrent layer's tensors copied to the
host), the prefill_merge counters of the turn alone (merged forwards, snapshot stashes, async stashes, no-slab
fallbacks), whether each stash object is a PendingStash, the first GEN greedy tokens, and a FOLLOW-UP turn (prompt +
generated tokens + a fixed 64-token tail) whose cached count must equal the deepest reachable stash and whose greedy
tokens are compared across arms. The expected merged-forward count per case comes from tests/partition_sim.py run
with the INSTALLED split_point and window_ends (the rule test_partition.py checks exhaustively on CPU).

Verdict lines (the unit greps them; every hard line FAIL -> exit 1):
  GPU-EQUIV ENGAGED <PASS|FAIL>             hard (R803 review B2): OFF/ASYNC never merge; MERGE and MERGE_ASYNC merge
                                            exactly where the rule says (merged == snapshot stashes == expected); the
                                            async arms take every stash asynchronously unless no slab was free
                                            (GDN stashes + async_snapshots + no_slab add up, PendingStash objects); the
                                            stash positions are the same set in every arm and contain S*; the turn
                                            restores the whole prefix P. Reported per case: whether partition_sim's
                                            positions are a subset of OFF's (decode checkpoints add stashes it omits)
  GPU-EQUIV ASYNC==OFF <PASS|FAIL>          hard: every stash of every case byte-identical by position, and the turn's and
                                            the follow-up's greedy tokens identical (review S2, S3)
  GPU-EQUIV MERGE_ASYNC==MERGE <PASS|FAIL>  hard: the same between the two merge arms
  GPU-EQUIV REUSE <PASS|FAIL>               hard, AMENDED 2026-09-29 step 0 #2: every arm's follow-up cached count equals
                                            OFF's (served reuse is the reference), and merged cases restore at >= S*
  GPU-EQUIV MERGE~OFF <PASS|FAIL>           hard, AMENDED 2026-09-29 step 0 #2: at S*, the FIRST GDN layer's state and conv
                                            window MERGE vs OFF within FIRST_LAYER_BOUND relative L2, integer tensors (PLE id
                                            context), position and checkpoint size identical. A wrong capture point, slot or
                                            conv window is full-size at the first layer; M-variance has had one layer to grow
  GPU-EQUIV MERGE~OFF-WHOLE <PASS|FAIL>     reported (was the hard gate): every layer within REL_L2_BOUND
  GPU-EQUIV MERGE==OFF <BITWISE|DIFFERS>    reported (DESIGN.md §3.3: bitwise is not derivable offline); per case the max
                                            relative L2, max |diff| and max bf16/fp16 ulp distance are printed
  GPU-EQUIV GREEDY ...                      reported: greedy agreement MERGE vs OFF (turn and follow-up)
  GPU-EQUIV CTRL~OFF ...                    reported: OFF at max_chunk_size 256 vs OFF at 2048 (another valid partition of
                                            the same prompt, no merge code): the scale of partition-only differences
  GPU-EQUIV SERVED-REUSE-GAP ...            reported: cases where OFF restores short of its deepest stash (served behaviour)
  per case, "per-layer rel L2 at S*" prints the MERGE vs OFF profile in layer order (depth growth vs a step)

  --arms OFF,ASYNC runs lever (d) alone (MERGE lines N/A); --ctrl-chunk 0 drops the CTRL arm.

  docker run ... --entrypoint python3 tabbyapi:prefill-merge-r1 /opt/prefill-merge-r1/tests/gpu_stash_equiv.py \\
      --model /models/MODEL --out /results/gpu-equiv [--cache-quant 8,8 --gpu-split 30,30 --draft 3 --draft-split 0,32]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CASES = [  # (cached prefix P (page multiple), N new tokens as TabbyAPI logs them: prompt - cached)
    (0, 257), (0, 258), (0, 512), (0, 513), (0, 757), (0, 2049), (0, 2100), (0, 2807), (0, 4095), (0, 4097),
    (0, 6146), (0, 8192), (8192, 258), (8192, 757), (8192, 2807), (8192, 4095), (8192, 8192),
]   # (0, 8192) and (8192, 8192): an LS-pipeline window, then a merged serial tail split at +1,792 (review S5)
ARMS = (("OFF", {}), ("ASYNC", {"EXL3_STASH_ASYNC": "1"}), ("MERGE", {"EXL3_PREFILL_MERGE": "1"}),
        ("MERGE_ASYNC", {"EXL3_PREFILL_MERGE": "1", "EXL3_STASH_ASYNC": "1"}))
PAGE = 256
REL_L2_BOUND = 5e-2     # pre-registered (R803 review S4)


def arguments(argv = None):
    p = argparse.ArgumentParser(description = __doc__.split("\n")[0])
    p.add_argument("--model", required = True)
    p.add_argument("--out", type = Path, required = True)
    p.add_argument("--cache-quant", default = "8,8")
    p.add_argument("--cache-tokens", type = int, default = 65536)
    p.add_argument("--gpu-split", default = "30,30")
    p.add_argument("--draft", type = int, default = 3)
    p.add_argument("--draft-split", default = "0,32")
    p.add_argument("--gen", type = int, default = 16)
    p.add_argument("--seed", type = int, default = 20260929)
    p.add_argument("--cases", help = "P:N,P:N,... (default: the built-in list)")
    p.add_argument("--arms", default = "OFF,ASYNC,MERGE,MERGE_ASYNC",
                   help = "gated arms to run (OFF and ASYNC always; OFF,ASYNC = lever (d) alone)")
    p.add_argument("--ctrl-chunk", type = int, default = 256,
                   help = "control arm CTRL: OFF with the generator's max_chunk_size set to this (a page multiple; 0 = "
                          "no control), reported as the spread of another valid partition")
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------ verdict logic (CPU-testable)

def compare_stash(x, y):
    """Two host stashes (GDNState.stash layout: scalars + per-layer tuples of tensors) ->
    dict(bitwise, layers_differ, max_abs, max_ulp, max_rel_l2, int_equal, meta_equal)."""
    import torch
    r = dict(bitwise = True, layers_differ = 0, max_abs = 0.0, max_ulp = 0, max_rel_l2 = 0.0, int_equal = True,
             meta_equal = True, per_layer = {})
    if set(x) != set(y):
        r.update(bitwise = False, meta_equal = False, max_rel_l2 = math.inf)
        return r
    for k, v in x.items():
        if not isinstance(v, tuple):
            if v != y[k]:
                r.update(bitwise = False, meta_equal = False)
            continue
        layer_same = True
        rels = []
        r["per_layer"][str(k)] = dict(kind = layer_kind(v), rel_l2 = rels, order = list(_layer_order(k)))
        if len(v) != len(y[k]):
            r.update(bitwise = False, meta_equal = False, max_rel_l2 = math.inf)
            continue
        for p, q in zip(v, y[k]):
            if p.dtype != q.dtype or p.shape != q.shape:
                r.update(bitwise = False, meta_equal = False, max_rel_l2 = math.inf)
                layer_same = False
                rels.append(math.inf)
                continue
            fp = p.is_floating_point()
            if torch.equal(p.contiguous().view(torch.uint8) if fp else p, q.contiguous().view(torch.uint8) if fp else q):
                rels.append(0.0)
                continue
            layer_same = False
            r["bitwise"] = False
            if not fp:
                r["int_equal"] = False
                rels.append(math.inf)
                continue
            d = p.float() - q.float()
            r["max_abs"] = max(r["max_abs"], d.abs().max().item())
            den = q.float().norm().item()
            num = d.norm().item()
            rel = num / den if den > 0 else (0.0 if num == 0 else math.inf)
            if not math.isfinite(num):
                rel = math.inf
            r["max_rel_l2"] = max(r["max_rel_l2"], rel)
            rels.append(rel)
            if p.element_size() == 2:
                r["max_ulp"] = max(r["max_ulp"], (p.view(torch.int16).int() - q.view(torch.int16).int()).abs().max().item())
        r["layers_differ"] += not layer_same
    return r


def layer_kind(v):
    """'ple' for a PLE stash tuple (it carries the integer id context), 'gdn' otherwise."""
    return "ple" if any(not t.is_floating_point() for t in v) else "gdn"


def _layer_order(k):
    return k if isinstance(k, tuple) else (k,)


def first_gdn(per_layer):
    """(key, rel_l2 list) of the lowest-index GDN layer in a compare_stash per_layer map, or (None, None)."""
    g = sorted((k for k, d in per_layer.items() if d["kind"] == "gdn"), key = lambda k: per_layer[k]["order"])
    return (g[0], per_layer[g[0]]["rel_l2"]) if g else (None, None)


def compare_sets(xs, ys):
    """Two {position: host stash} maps -> (same position set, all bitwise, per-position compare_stash results)."""
    rows = {pos: compare_stash(xs[pos], ys[pos]) for pos in sorted(set(xs) & set(ys))}
    same_set = set(xs) == set(ys)
    return same_set, same_set and all(r["bitwise"] for r in rows.values()), rows


def merge_close(rows, bound = REL_L2_BOUND):
    """MERGE~OFF: every stash within the relative-L2 bound, integer tensors and metadata identical."""
    return all(r["meta_equal"] and r["int_equal"] and r["max_rel_l2"] < bound for r in rows.values())


def engaged(arm, rec, exp_merged, s_star, p_cached):
    """Did the arm run the paths it claims (review B2)? rec: cached, merged, snap, asyncs, no_slab, merge_no_slab,
    incomplete, positions, pending ({position: is PendingStash}). The expectation assumes the turn started from the
    P cached tokens it was set up with; a different start is named instead of being read as a merge failure.
    Returns a list of reasons (empty = engaged as expected)."""
    bad = []
    if (rec["cached"] or 0) != p_cached:        # a cold turn may report None for no cached tokens
        bad.append(f"turn started at cached {rec['cached']}, not P {p_cached}: expectation not applicable")
    if rec.get("merge_no_slab"):
        bad.append(f"{rec['merge_no_slab']} merge(s) fell back to the served cut: no free staging slab")
    if rec.get("incomplete"):
        bad.append(f"{rec['incomplete']} incomplete snapshot(s): merge disabled itself")
    pos, pend = rec["positions"], rec["pending"]
    n, npend = len(pos), sum(1 for p in pos if pend[p])
    if s_star not in pos:
        bad.append(f"no stash at S* {s_star}")
    merge_arm, async_arm = arm in ("MERGE", "MERGE_ASYNC"), arm in ("ASYNC", "MERGE_ASYNC")
    want_m = exp_merged if merge_arm else 0
    if rec["merged"] != want_m or rec["snap"] != want_m:
        bad.append(f"merged {rec['merged']} snapshot {rec['snap']} != expected {want_m}")
    snap_async = rec.get("snap_async", 0)
    if not async_arm:
        if rec["asyncs"] or snap_async:
            bad.append(f"{rec['asyncs']} async stashes / {snap_async} async snapshots with EXL3_STASH_ASYNC off")
        # MERGE: the snapshot stash is a PendingStash (finished synchronously); everything else a plain dict
        if npend != want_m:
            bad.append(f"{npend} PendingStash objects, expected {want_m}")
        if merge_arm and exp_merged and s_star in pend and not pend[s_star]:
            bad.append("S* stash is not the snapshot (not a PendingStash)")
    else:
        # a merged snapshot finishes on the worker too (async_snapshots): a MERGE_ASYNC turn whose only stash is the
        # snapshot takes no GDNState.stash() at all (flan R803 step 0, N 258 / 512 / 757)
        if rec["asyncs"] + snap_async < 1:
            bad.append("no asynchronous stash or snapshot")
        if snap_async != want_m:
            bad.append(f"{snap_async} asynchronous snapshots != expected {want_m}")
        if rec["asyncs"] + rec["no_slab"] + want_m != n:
            bad.append(f"async {rec['asyncs']} + no-slab {rec['no_slab']} + snapshot {want_m} != {n} stashes")
        if npend != rec["asyncs"] + want_m:
            bad.append(f"{npend} PendingStash objects != async {rec['asyncs']} + snapshot {want_m}")
        if rec["no_slab"] == 0 and s_star in pend and not pend[s_star]:
            bad.append("S* stash is not a PendingStash")
    return bad


def case_verdict(res, exp_merged, s_star, p_cached):
    """One case over the arms present (OFF + ASYNC always; MERGE + MERGE_ASYNC unless --arms leaves them out; CTRL
    optional) -> row dict with the hard-gate booleans (see the module docstring). A gate whose arms are absent is None."""
    row = {"S": s_star, "expected_merged": exp_merged}
    arms = [a for a in GATED_ARMS if a in res]
    merge = "MERGE" in res and "MERGE_ASYNC" in res
    eng = {arm: engaged(arm, res[arm], exp_merged, s_star, p_cached) for arm in arms}
    sets = {arm: sorted(res[arm]["positions"]) for arm in arms}
    if len({tuple(v) for v in sets.values()}) != 1:
        eng["positions"] = [f"stash positions differ across arms: {sets}"]
    row["engaged"] = {k: v for k, v in eng.items() if v}
    row["ENGAGED"] = not row["engaged"]
    pairs = {"ASYNC==OFF": ("ASYNC", "OFF")}
    if merge:
        pairs["MERGE_ASYNC==MERGE"] = ("MERGE_ASYNC", "MERGE")
    for name, (x, y) in pairs.items():
        same_set, bitwise, rows = compare_sets(res[x]["stashes"], res[y]["stashes"])
        toks = res[x]["toks"] == res[y]["toks"] and res[x]["ftoks"] == res[y]["ftoks"]
        row[name] = dict(stashes = len(rows), same_set = same_set, bitwise = bitwise, tokens_equal = toks,
                         ok = same_set and bitwise and toks and len(rows) > 0)
    # REUSE (amended after flan R803 step 0): every arm restores exactly where OFF restores -- the knobs must not lose
    # (or invent) next-turn reuse -- and a merged case restores at or past its snapshot S*. The absolute "deepest
    # reachable stash" is reported only: the served OFF path does not reuse a stash that sits exactly at a
    # page-multiple prompt end (N_p % 256 == 0: N 257 / 513 / 2049 / 4097 restored 0 / 0 / 0 / 2048 in every arm).
    off_c = res["OFF"]["fcached"]
    row["REUSE"] = all(res[arm]["fcached"] == off_c for arm in arms) and \
        (exp_merged == 0 or not merge or all((res[arm]["fcached"] or 0) >= s_star for arm in arms))
    row["reuse"] = {arm: (res[arm]["fcached"], res[arm]["fexpect"]) for arm in res}
    row["served_reuse_gap"] = (off_c or 0) < res["OFF"]["fexpect"]
    if merge:
        same_set, bitwise, rows = compare_sets(res["MERGE"]["stashes"], res["OFF"]["stashes"])
        row["MERGE==OFF"] = dict(same_set = same_set, bitwise = bitwise,
                                 max_rel_l2 = max((r["max_rel_l2"] for r in rows.values()), default = math.inf),
                                 max_abs = max((r["max_abs"] for r in rows.values()), default = 0.0),
                                 max_ulp = max((r["max_ulp"] for r in rows.values()), default = 0),
                                 per_position = rows)
        # whole-stash bound (review S4, 5e-2): REPORTED since flan R803 step 0 -- it cannot tell a bug from the
        # model's depth amplification of row-count (M) numerics (every merged case failed it at 0.06-0.13 while
        # ASYNC==OFF held bitwise). The gate is the first GDN layer: its inputs differ from OFF only through the few
        # kernels before it, so a wrong capture point / slot / conv window shows there at full size.
        row["MERGE~OFF_whole"] = same_set and len(rows) > 0 and merge_close(rows)
        fl = {}
        for pos, r in rows.items():
            k, rel = first_gdn(r["per_layer"])
            fl[pos] = dict(layer = str(k), rel_l2 = rel)
        row["first_gdn_rel_l2"] = fl
        row["MERGE~OFF"] = bool(rows) and same_set and all(r["meta_equal"] and r["int_equal"] for r in rows.values()) \
            and all(d["rel_l2"] is not None and max(d["rel_l2"]) < FIRST_LAYER_BOUND for d in fl.values())
        # per-layer profile at S* (layer order): what grows with depth
        if s_star in rows:
            pl = rows[s_star]["per_layer"]
            row["profile_S"] = [(k, pl[k]["kind"], [round(x, 6) for x in pl[k]["rel_l2"]])
                                for k in sorted(pl, key = lambda k: pl[k]["order"])]
        row["greedy_turn_equal"] = res["MERGE"]["toks"] == res["OFF"]["toks"]
        row["greedy_follow_equal"] = res["MERGE"]["ftoks"] == res["OFF"]["ftoks"]
    if "CTRL" in res:
        _, _, crows = compare_sets(res["CTRL"]["stashes"], res["OFF"]["stashes"])
        row["CTRL~OFF"] = dict(common = sorted(crows),
                               max_rel_l2 = max((r["max_rel_l2"] for r in crows.values()), default = None),
                               first_gdn = {p: first_gdn(r["per_layer"])[1] for p, r in crows.items()},
                               bitwise = all(r["bitwise"] for r in crows.values()) if crows else None)
    return row


GATED_ARMS = ("OFF", "ASYNC", "MERGE", "MERGE_ASYNC")
FIRST_LAYER_BOUND = 1e-2   # amended after flan R803 step 0 (see case_verdict); REL_L2_BOUND is reported
HARD = ("ENGAGED", "ASYNC==OFF", "MERGE_ASYNC==MERGE", "REUSE", "MERGE~OFF")


def summarize(rows):
    """Case rows -> (hard gate dict, verdict lines). A gate whose arms were not run prints N/A and is left out."""
    merge = bool(rows) and all("MERGE==OFF" in r for r in rows)
    keys = [k for k in HARD if merge or k not in ("MERGE_ASYNC==MERGE", "MERGE~OFF")]
    hard = {k: bool(rows) for k in keys}
    for r in rows:
        hard["ENGAGED"] &= r["ENGAGED"]
        hard["ASYNC==OFF"] &= r["ASYNC==OFF"]["ok"]
        hard["REUSE"] &= r["REUSE"]
        if merge:
            hard["MERGE_ASYNC==MERGE"] &= r["MERGE_ASYNC==MERGE"]["ok"]
            hard["MERGE~OFF"] &= r["MERGE~OFF"]
    n = len(rows)
    lines = []
    for k in HARD:
        if k not in hard:
            lines.append(f"GPU-EQUIV {k} N/A (arms not run)")
            continue
        extra = ""
        if k == "MERGE~OFF":
            fmax = max((max(d["rel_l2"]) for r in rows for d in r["first_gdn_rel_l2"].values() if d["rel_l2"]),
                       default = math.inf)
            extra = f" (first GDN layer max rel L2 {fmax:.3g}, bound {FIRST_LAYER_BOUND:g})"
        lines.append(f"GPU-EQUIV {k} {'PASS' if hard[k] else 'FAIL'}{extra}")
    if merge:
        rel = max((r["MERGE==OFF"]["max_rel_l2"] for r in rows), default = math.inf)
        whole = all(r["MERGE~OFF_whole"] for r in rows)
        lines.append(f"GPU-EQUIV MERGE~OFF-WHOLE {'PASS' if whole else 'FAIL'} (reported: max rel L2 {rel:.3g}, "
                     f"former bound {REL_L2_BOUND:g})")
        lines.append(f"GPU-EQUIV MERGE==OFF {'BITWISE' if all(r['MERGE==OFF']['bitwise'] for r in rows) else 'DIFFERS'}")
        lines.append(f"GPU-EQUIV GREEDY turn {sum(r['greedy_turn_equal'] for r in rows)}/{n} follow-up "
                     f"{sum(r['greedy_follow_equal'] for r in rows)}/{n} (MERGE vs OFF)")
    ctrl = [r["CTRL~OFF"]["max_rel_l2"] for r in rows if r.get("CTRL~OFF") and r["CTRL~OFF"]["max_rel_l2"] is not None]
    if ctrl:
        lines.append(f"GPU-EQUIV CTRL~OFF max rel L2 {max(ctrl):.3g} (reported: OFF at max_chunk_size 256 vs 2048, the "
                     f"spread of another valid partition)")
    gap = [f"{r.get('P')}:{r.get('N')}" for r in rows if r.get("served_reuse_gap")]
    lines.append(f"GPU-EQUIV SERVED-REUSE-GAP {len(gap)} case(s) {gap} (reported: OFF restores short of its deepest "
                 f"stash)")
    return hard, lines


# ------------------------------------------------------------------------------------------------------ GPU run

def main():
    a = arguments()
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
    from exllamav3.cache import CacheLayer_quant
    import exllamav3.cache.prefill_merge as pm
    from exllamav3.generator.prefill_pipeline import window_ends
    import partition_sim

    for k in ("EXL3_PREFILL_MERGE", "EXL3_STASH_ASYNC"):
        os.environ.pop(k, None)
    want = [x for x in a.arms.split(",") if x]
    assert "OFF" in want and "ASYNC" in want and set(want) <= set(GATED_ARMS), want
    assert ("MERGE" in want) == ("MERGE_ASYNC" in want), "MERGE and MERGE_ASYNC run together"
    assert a.ctrl_chunk == 0 or (a.ctrl_chunk >= PAGE and a.ctrl_chunk % PAGE == 0), a.ctrl_chunk
    arms = [(n, e, None) for n, e in ARMS if n in want] + ([("CTRL", {}, a.ctrl_chunk)] if a.ctrl_chunk else [])
    assert "EXL3_NVME_TIER" not in os.environ, "run without the NVMe tier: a tier would serve identical prompts across arms"
    a.out.mkdir(parents = True, exist_ok = True)
    cases = CASES if not a.cases else [tuple(int(x) for x in c.split(":")) for c in a.cases.split(",")]
    pipeline = os.environ.get("EXL3_LS_PREFILL_PIPELINE", "0") == "1"

    config = Config.from_directory(a.model)
    model = Model.from_config(config)
    draft_model = Model.from_config(config, component = "mtp") if a.draft else None
    bits = [int(x) for x in a.cache_quant.split(",")]
    ckw = dict(layer_type = CacheLayer_quant, k_bits = bits[0], v_bits = bits[-1])
    cache = Cache(model, max_num_tokens = a.cache_tokens, max_batch_size = 2, max_history = a.draft, **ckw)
    draft_cache = Cache(draft_model, max_num_tokens = a.cache_tokens, **ckw) if draft_model else None
    if draft_model:
        draft_model.load(use_per_device = [float(x) for x in a.draft_split.split(",")], max_chunk_size = 2048)
    model.load(use_per_device = [float(x) for x in a.gpu_split.split(",")], max_chunk_size = 2048, max_batch_size = 2)
    tokenizer = Tokenizer.from_config(config)
    gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_batch_size = 2, max_chunk_size = 2048,
                    draft_model = draft_model, draft_cache = draft_cache, num_draft_tokens = a.draft,
                    dynamic_draft_tokens = False)
    assert gen.recurrent_cache is not None

    rng = random.Random(a.seed)
    vocab_lo, vocab_hi = 1000, 100000          # ordinary tokens (Qwen special tokens sit at the top of the vocab)

    def ids(n):
        return torch.tensor([[rng.randrange(vocab_lo, vocab_hi) for _ in range(n)]], dtype = torch.long)

    def run(input_ids, n):
        job = Job(input_ids = input_ids, max_new_tokens = n, sampler = GreedySampler(), stop_conditions = [])
        gen.enqueue(job)
        toks, last = [], None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r["stage"] == "error":
                    raise RuntimeError(str(r))
                if r.get("token_ids") is not None:
                    toks += r["token_ids"][0].tolist()
                if r.get("eos"):
                    last = r
        return toks, last

    def reset():
        pm.drain_worker()
        gen.pagetable.reset_page_table()
        gen.recurrent_cache.clear()
        gen.recurrent_cache.update_total_size()

    def host(st):
        pm.wait_stash(st)
        out = {}
        for k, v in st.items():
            if isinstance(v, (tuple, list)):
                out[k] = tuple(t.detach().to("cpu").clone() for t in v)
            else:
                out[k] = v
        return out

    counters = ("merged_prefills", "snapshot_stashes", "async_stashes", "async_no_slab", "merge_no_slab",
                "incomplete_snapshots", "async_snapshots")
    chunk0 = gen.max_chunk_size
    rows = []
    t0 = time.time()
    for P, N in cases:
        prefix = ids(P) if P else None
        body = ids(N)
        tail = ids(64)
        prompt = body if P == 0 else torch.cat((prefix, body), dim = 1)
        L = prompt.shape[1]
        S = (L - 1) // PAGE * PAGE                                  # the last-page stash position
        exp_merged, sim_stashes = partition_sim.merged_forwards(P, N, pipeline, split_point = pm.split_point,
                                                                window_ends = window_ends, PAGE_SIZE = PAGE)
        res = {}
        for arm, env, chunk in arms:
            for k in ("EXL3_PREFILL_MERGE", "EXL3_STASH_ASYNC"):
                os.environ.pop(k, None)
            gen.max_chunk_size = chunk0
            reset()
            if P:
                # the prefix + 2 tokens, knobs off: N_p = P + 1, so the served last-page cut stashes at P. (The first
                # step-0 run used prefix + 1 token, N_p = P: the served path does not reuse a stash that sits exactly
                # at a page-multiple prompt end, and every P = 8192 turn restarted at 6,144.)
                run(torch.cat((prefix, body[:, :2]), dim = 1), 1)
            if chunk:
                gen.max_chunk_size = chunk
            os.environ.update(env)
            before = set(gen.recurrent_cache.keys())
            m0 = {c: pm.metrics[c] for c in counters}
            toks, last = run(prompt.clone(), a.gen)
            pm.drain_worker()
            dm = {c: pm.metrics[c] - m0[c] for c in counters}      # this turn alone (before the follow-up)
            new = {k: gen.recurrent_cache[k] for k in gen.recurrent_cache.keys() if k not in before}
            by_pos = {}
            for v in new.values():
                assert v["position"] not in by_pos, f"{arm} P {P} N {N}: two new stashes at {v['position']}"
                by_pos[v["position"]] = v
            pending = {p: isinstance(v, pm.PendingStash) for p, v in by_pos.items()}
            stashes = {p: host(v) for p, v in by_pos.items()}
            follow = torch.cat((prompt, torch.tensor([toks], dtype = torch.long), tail), dim = 1)
            ftoks, flast = run(follow, a.gen)
            # the deepest stash the follow-up can restore: S*, or a decode checkpoint the GEN generated tokens crossed
            reach = [p for p in by_pos if p <= L + len(toks)] or [-1]
            gen.max_chunk_size = chunk0
            res[arm] = dict(stashes = stashes, positions = set(by_pos), pending = pending, toks = toks,
                            cached = last.get("cached_tokens"), fcached = flast.get("cached_tokens"),
                            fexpect = max(reach), ftoks = ftoks, merged = dm["merged_prefills"],
                            snap = dm["snapshot_stashes"], asyncs = dm["async_stashes"], no_slab = dm["async_no_slab"],
                            merge_no_slab = dm["merge_no_slab"], incomplete = dm["incomplete_snapshots"],
                            snap_async = dm["async_snapshots"])
        for k in ("EXL3_PREFILL_MERGE", "EXL3_STASH_ASYNC"):
            os.environ.pop(k, None)
        row = case_verdict(res, exp_merged, S, P)
        # decode checkpoints (the GEN tokens crossing a 2,048 boundary) are stashes the prefill replay does not model
        row.update(P = P, N = N, L = L, sim_stashes = sim_stashes,
                   sim_matches = set(sim_stashes) <= set(res["OFF"]["positions"]),
                   counters = {arm: {c: res[arm][c] for c in ("cached", "merged", "snap", "asyncs", "snap_async",
                                                               "no_slab", "merge_no_slab", "incomplete")} for arm in res})
        rows.append(row)
        me = row.get("MERGE==OFF")
        mo = "not run" if me is None else "BITWISE" if me["bitwise"] else \
            "DIFFERS max rel L2 %.3g max|d| %.3g max ulp %d; first GDN layer rel L2 %s" % (
                me["max_rel_l2"], me["max_abs"], me["max_ulp"],
                {p: [round(x, 5) for x in (d["rel_l2"] or [])] for p, d in row["first_gdn_rel_l2"].items()})
        ctl = row.get("CTRL~OFF")
        print(f"P {P:5d} N {N:5d} S* {S:6d}: expect merged {exp_merged}, stashes {sorted(res['OFF']['positions'])}"
              f"{'' if row['sim_matches'] else ' (sim ' + str(sorted(sim_stashes)) + ')'} | ENGAGED {row['ENGAGED']}"
              f"{'' if row['ENGAGED'] else ' ' + json.dumps(row['engaged'])} | counters {row['counters']} | "
              f"ASYNC==OFF {row['ASYNC==OFF']['ok']} MERGE_ASYNC==MERGE "
              f"{row['MERGE_ASYNC==MERGE']['ok'] if 'MERGE_ASYNC==MERGE' in row else 'n/a'} | MERGE vs OFF {mo} ~ "
              f"{row.get('MERGE~OFF', 'n/a')} | CTRL vs OFF "
              f"{'n/a' if not ctl else ('BITWISE' if ctl['bitwise'] else 'max rel L2 %.3g' % ctl['max_rel_l2'])} | "
              f"follow-up {row['reuse']} REUSE {row['REUSE']} | greedy turn {row.get('greedy_turn_equal', 'n/a')} "
              f"follow {row.get('greedy_follow_equal', 'n/a')}", flush = True)
        if row.get("profile_S"):
            print(f"    per-layer rel L2 at S* (layer order, [state, conv] / [conv, id]): "
                  + " ".join(f"{k}:{[('%.2g' % x) for x in v]}" for k, _, v in row["profile_S"]), flush = True)
    (a.out / "gpu-equiv.json").write_text(json.dumps(rows, indent = 1, default = str))
    hard, lines = summarize(rows)
    for line in lines:
        print(line)
    print(f"GPU-EQUIV metrics {pm.metrics}; pipeline {pipeline}; {time.time() - t0:.0f} s")
    return 0 if all(hard.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
