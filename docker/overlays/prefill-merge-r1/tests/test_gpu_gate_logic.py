#!/usr/bin/env python3
"""prefill-merge r1: the GPU gate's verdict logic (tests/gpu_stash_equiv.py case_verdict / summarize) on CPU fixtures.

Review R803 B2/S2/S3/S4: every hard line must be able to FAIL. A passing four-arm fixture, then one mutation per
check: merge not engaged, merge engaged where the rule says no, async never taken, S* not a PendingStash, stash
positions differing, a non-S* stash differing under ASYNC, async tokens differing, MERGE stash outside the relative-L2
bound, PLE id context differing, the follow-up not reusing, and an empty run.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: E402

import torch  # noqa: E402
import gpu_stash_equiv as G  # noqa: E402

S_STAR = 2560
P_CACHED = 0
POS = [2048, 2560]


def _noisy(t, noise):
    if not noise:
        return t
    return t + noise * torch.randn(t.shape, generator = torch.Generator().manual_seed(99)) * t.norm() / t.numel() ** 0.5


def stash(pos, seed, noise = 0.0, deep_noise = None):
    """Two GDN layers (keys (0, 0) and (40, 0), as get_all_recurrent_layers() names them) + the PLE layer (-3, 0).
    noise perturbs every GDN layer's recurrent plane; deep_noise only the deep layer's."""
    g = torch.Generator().manual_seed(seed * 7919 + pos)
    d = {"position": pos, "checkpoint_size": 1000}
    for li in (0, 40):
        rec = torch.randn(1, 4, 8, 8, generator = g)
        conv = torch.randn(16, 4, generator = g)
        n = deep_noise if (li == 40 and deep_noise is not None) else noise
        d[(li, 0)] = (_noisy(rec, n).bfloat16(), conv.bfloat16())
    d[(-3, 0)] = (torch.randn(8, 3, generator = g).half(), torch.arange(pos % 97, pos % 97 + 12))
    return d


def arm(stashes, pending, merged = 0, snap = 0, asyncs = 0, no_slab = 0, snap_async = 0, toks = (1, 2, 3),
        ftoks = (4, 5)):
    return dict(stashes = stashes, positions = set(stashes), pending = pending, toks = list(toks), ftoks = list(ftoks),
                cached = P_CACHED, fcached = S_STAR, fexpect = S_STAR, merged = merged, snap = snap, asyncs = asyncs,
                no_slab = no_slab, merge_no_slab = 0, incomplete = 0, snap_async = snap_async)


def fixture(noise = 1e-3, deep_noise = None):
    off = {p: stash(p, 1) for p in POS}
    mer = {p: stash(p, 1, noise if p == S_STAR else 0.0, deep_noise if p == S_STAR else None) for p in POS}
    return {
        "OFF": arm(off, {p: False for p in POS}),
        "ASYNC": arm(copy.deepcopy(off), {p: True for p in POS}, asyncs = 2),
        "MERGE": arm(mer, {2048: False, S_STAR: True}, merged = 1, snap = 1),
        "MERGE_ASYNC": arm(copy.deepcopy(mer), {p: True for p in POS}, merged = 1, snap = 1, asyncs = 1, snap_async = 1),
    }


def verdict(res, exp = 1):
    row = G.case_verdict(res, exp, S_STAR, P_CACHED)
    hard, lines = G.summarize([row])
    return row, hard, lines


def failed(hard):
    return sorted(k for k, v in hard.items() if not v)


def test_pass_fixture():
    row, hard, lines = verdict(fixture())
    assert failed(hard) == [], (hard, row["engaged"])
    assert "GPU-EQUIV MERGE==OFF DIFFERS" in lines, lines
    assert 0 < row["MERGE==OFF"]["max_rel_l2"] < 5e-2, row["MERGE==OFF"]["max_rel_l2"]
    assert "GPU-EQUIV MERGE~OFF-WHOLE PASS" in " ".join(lines), lines
    import json
    json.dumps([row], default = str)                   # gpu-equiv.json must serialize (tuple layer keys -> str)
    # an unperturbed merge is reported BITWISE and still passes
    _, hard, lines = verdict(fixture(noise = 0.0))
    assert failed(hard) == [] and "GPU-EQUIV MERGE==OFF BITWISE" in lines


def test_engaged_fails():
    r = fixture(); r["MERGE"].update(merged = 0, snap = 0)                 # merge silently off
    assert failed(verdict(r)[1]) == ["ENGAGED"]
    r = fixture(); r["OFF"].update(merged = 1)                             # merge where the knob is off
    assert failed(verdict(r)[1]) == ["ENGAGED"]
    r = fixture()                                                          # merge where the rule says no
    assert "ENGAGED" in failed(verdict(r, exp = 0)[1])
    r = fixture(); r["ASYNC"].update(asyncs = 0, no_slab = 2, pending = {p: False for p in POS})   # all fell back
    assert failed(verdict(r)[1]) == ["ENGAGED"]
    r = fixture(); r["ASYNC"]["pending"][S_STAR] = False; r["ASYNC"]["asyncs"] = 2    # S* not a PendingStash
    assert failed(verdict(r)[1]) == ["ENGAGED"]
    r = fixture(); r["MERGE_ASYNC"]["snap_async"] = 0                      # snapshot finished synchronously
    assert failed(verdict(r)[1]) == ["ENGAGED"]
    r = fixture(); r["MERGE"]["snap_async"] = 1                            # async snapshot with the knob off
    assert failed(verdict(r)[1]) == ["ENGAGED"]
    r = fixture(); r["MERGE"]["pending"] = {p: False for p in POS}         # snapshot not stored
    assert failed(verdict(r)[1]) == ["ENGAGED"]
    r = fixture()                                                          # a stash missing in one arm
    del r["MERGE_ASYNC"]["stashes"][2048]; r["MERGE_ASYNC"]["positions"] = {S_STAR}
    r["MERGE_ASYNC"]["pending"] = {S_STAR: True}; r["MERGE_ASYNC"]["asyncs"] = 0
    f = failed(verdict(r)[1])
    assert "ENGAGED" in f and "MERGE_ASYNC==MERGE" in f, f
    r = fixture(); r["MERGE"]["cached"] = 256                              # the turn did not start at P (named)
    row, hard, _ = verdict(r)
    assert failed(hard) == ["ENGAGED"] and "not P 0" in str(row["engaged"]), row["engaged"]
    r = fixture(); r["MERGE_ASYNC"].update(merged = 0, snap = 0, merge_no_slab = 1, asyncs = 2,
                                           pending = {p: True for p in POS})   # slab exhaustion: named fallback
    row, hard, _ = verdict(r)
    assert failed(hard) == ["ENGAGED"] and "no free staging slab" in str(row["engaged"]), row["engaged"]
    r = fixture(); r["MERGE"]["incomplete"] = 1
    row, hard, _ = verdict(r)
    assert failed(hard) == ["ENGAGED"] and "incomplete" in str(row["engaged"]), row["engaged"]


def test_snapshot_only_merge_async_engaged():
    # flan R803 step 0: N 258 / 512 / 757 -- the snapshot is the turn's only stash, so MERGE_ASYNC takes no
    # GDNState.stash(); its snapshot finishing on the worker (async_snapshots) is the async engagement
    only = [S_STAR]
    off = {S_STAR: stash(S_STAR, 1)}
    mer = {S_STAR: stash(S_STAR, 1, 1e-3)}
    r = {"OFF": arm(off, {S_STAR: False}),
         "ASYNC": arm(copy.deepcopy(off), {S_STAR: True}, asyncs = 1),
         "MERGE": arm(mer, {S_STAR: True}, merged = 1, snap = 1),
         "MERGE_ASYNC": arm(copy.deepcopy(mer), {S_STAR: True}, merged = 1, snap = 1, snap_async = 1)}
    assert only and failed(verdict(r)[1]) == []
    r["MERGE_ASYNC"]["snap_async"] = 0
    assert failed(verdict(r)[1]) == ["ENGAGED"]


def test_async_equality_fails():
    r = fixture(); r["ASYNC"]["stashes"][2048] = stash(2048, 2)            # a non-S* stash differs (S3)
    assert failed(verdict(r)[1]) == ["ASYNC==OFF"]
    r = fixture(); r["ASYNC"]["ftoks"] = [4, 6]                            # follow-up tokens differ (S2)
    assert failed(verdict(r)[1]) == ["ASYNC==OFF"]
    r = fixture(); r["MERGE_ASYNC"]["toks"] = [1, 2, 4]
    assert failed(verdict(r)[1]) == ["MERGE_ASYNC==MERGE"]


def test_merge_bound_fails():
    r = fixture(noise = 0.2)                                               # the FIRST GDN layer already 0.2 off
    row, hard, lines = verdict(r)
    assert failed(hard) == ["MERGE~OFF"], failed(hard)
    assert 0.1 < row["MERGE==OFF"]["max_rel_l2"] < 0.4, row["MERGE==OFF"]["max_rel_l2"]
    # depth amplification (flan R803 step 0: 0.06-0.13 over the whole stash): first layer at 1e-3, a deep layer at
    # 0.15 -> the gate (first GDN layer) passes, the former whole-stash bound is reported as failing
    r = fixture(noise = 1e-3, deep_noise = 0.15)
    row, hard, lines = verdict(r)
    assert failed(hard) == [], (failed(hard), row["first_gdn_rel_l2"])
    assert any(l.startswith("GPU-EQUIV MERGE~OFF-WHOLE FAIL") for l in lines), lines
    assert row["first_gdn_rel_l2"][S_STAR]["layer"] == "(0, 0)" and max(row["first_gdn_rel_l2"][S_STAR]["rel_l2"]) < 1e-2
    assert [k for k, _, _ in row["profile_S"]] == ["(-3, 0)", "(0, 0)", "(40, 0)"], row["profile_S"]
    r = fixture()
    ple = r["MERGE"]["stashes"][S_STAR][(-3, 0)]
    r["MERGE"]["stashes"][S_STAR][(-3, 0)] = (ple[0], ple[1] + 1)          # PLE id context differs
    r["MERGE_ASYNC"]["stashes"] = copy.deepcopy(r["MERGE"]["stashes"])
    assert failed(verdict(r)[1]) == ["MERGE~OFF"]
    r = fixture(); r["MERGE"]["stashes"][S_STAR]["checkpoint_size"] = 999
    r["MERGE_ASYNC"]["stashes"] = copy.deepcopy(r["MERGE"]["stashes"])
    assert failed(verdict(r)[1]) == ["MERGE~OFF"]


def test_reuse_and_empty():
    r = fixture(); r["MERGE"]["fcached"] = 2048                            # MERGE loses reuse that OFF keeps
    assert failed(verdict(r)[1]) == ["REUSE"]
    r = fixture()                                                          # every arm short of S* on a merged case
    for a in r.values():
        a["fcached"] = 2048
    assert failed(verdict(r)[1]) == ["REUSE"]
    r = fixture()                                                          # the served gap: all arms equal, non-merged
    for a in r.values():
        a["fcached"] = 0
    row, hard, lines = verdict(r, exp = 0)
    assert "REUSE" not in failed(hard) and row["served_reuse_gap"], row
    assert any(l.startswith("GPU-EQUIV SERVED-REUSE-GAP 1") for l in lines), lines
    hard, lines = G.summarize([])
    assert failed(hard) == ["ASYNC==OFF", "ENGAGED", "REUSE"], hard
    assert "GPU-EQUIV MERGE~OFF N/A (arms not run)" in lines, lines


def test_async_only_arms():
    # lever (d) alone (--arms OFF,ASYNC): the merge gates are N/A, the async gates still decide
    r = fixture()
    del r["MERGE"], r["MERGE_ASYNC"]
    row, hard, lines = verdict(r, exp = 1)
    assert sorted(hard) == ["ASYNC==OFF", "ENGAGED", "REUSE"] and failed(hard) == [], (hard, row["engaged"])
    assert "GPU-EQUIV MERGE_ASYNC==MERGE N/A (arms not run)" in lines
    r["ASYNC"]["stashes"][2048] = stash(2048, 2)
    assert failed(verdict(r, exp = 1)[1]) == ["ASYNC==OFF"]


def test_ctrl_reported():
    r = fixture()
    r["CTRL"] = arm({p: stash(p, 1, 0.05) for p in POS}, {p: False for p in POS})
    row, hard, lines = verdict(r)
    assert failed(hard) == [] and 0 < row["CTRL~OFF"]["max_rel_l2"] < 0.2, row["CTRL~OFF"]
    assert any(l.startswith("GPU-EQUIV CTRL~OFF max rel L2") for l in lines), lines


def test_compare_stash_units():
    a = stash(512, 3)
    b = copy.deepcopy(a)
    r = G.compare_stash(a, b)
    assert r["bitwise"] and r["max_rel_l2"] == 0.0
    rec = b[(0, 0)][0].clone()
    rec.view(torch.int16)[0, 0, 0, 0] += 1                                 # one bf16 ulp
    b[(0, 0)] = (rec, b[(0, 0)][1])
    r = G.compare_stash(a, b)
    assert not r["bitwise"] and r["max_ulp"] == 1 and r["layers_differ"] == 1 and r["max_rel_l2"] < 1e-2, r


if __name__ == "__main__":
    _harness.run_tests(globals())
