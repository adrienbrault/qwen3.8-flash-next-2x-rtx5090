#!/usr/bin/env python3
"""prefill-merge r1: the Job rule (cache/prefill_merge.split_point) over the served prefill loop.

simulate() replays Job.prefill's chunking for a solo job (job.py:1231-1234 chunk end, :1316-1330 last-page cut or the
merged split, :1483-1496 stashes) plus generator.recurrent_checkpoint() after each round (job.py:1581-1639, default
intervals 2,048 / 32,768), with and without the LS prefill pipeline (prefill_pipeline.window_ends / checkpoint, the
served functions). It checks, for every N in the brief's classes (exhaustively for N_p 1..8,192) and two cached starts:
  OFF    serial forwards F = ceil(A/2048) + [N_p % 256 != 0]  (A = floor(N_p/256)*256; F = 1 when A = 0)
  MERGE  F = ceil(N_p/2048), no forward > 2,048 rows, the split offset a positive multiple of 256 inside the forward,
         and the SAME set of stash positions as OFF (only the last-page stash changes provenance).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: E402,F401  (stubs; PM_TREE)

from exllamav3.cache.prefill_merge import split_point  # noqa: E402
from exllamav3.generator.prefill_pipeline import window_ends  # noqa: E402
from exllamav3.constants import PAGE_SIZE  # noqa: E402
import partition_sim as _sim  # noqa: E402

CHUNK = _sim.CHUNK
CLASSES = [64, 256, 257, 258, 512, 513, 757, 1024, 1536, 2000, 2048, 2049, 2100, 2807, 3000, 4095, 4097, 4400,
           6144, 6145, 6146, 8192, 12288]


def simulate(cached, n_new, merge, pipeline = False):
    """Serial forwards [(start, end, split_abs or None)], windows [[ends]], stash positions (absolute)."""
    return _sim.simulate(cached, n_new, merge, pipeline, split_point = split_point, window_ends = window_ends,
                         PAGE_SIZE = PAGE_SIZE)


def f_off(n_p):
    a = n_p // PAGE_SIZE * PAGE_SIZE
    return 1 if a == 0 else math.ceil(a / CHUNK) + (1 if n_p % PAGE_SIZE else 0)


def check(cached, n_new, pipeline = False):
    n_p = n_new - 1
    off, won, soff = simulate(cached, n_new, False, pipeline)
    mer, wmer, smer = simulate(cached, n_new, True, pipeline)
    assert won == wmer, (n_new, won, wmer)
    assert sorted(soff) == sorted(smer), (cached, n_new, soff, smer)
    for s, e, sp in mer:
        assert e - s <= CHUNK, (n_new, s, e)
        if sp is not None:
            assert s < sp < e and (sp - s) % PAGE_SIZE == 0 and sp % PAGE_SIZE == 0, (n_new, s, e, sp)
    for s, e, sp in off:
        assert e - s <= CHUNK
    if not pipeline and n_p > 0:
        assert len(off) == f_off(n_p), (cached, n_new, len(off), f_off(n_p), off)
        assert len(mer) == math.ceil(n_p / CHUNK), (cached, n_new, len(mer), mer)
    # every serial token is prefilled exactly once, in order
    for fw in (off, mer):
        pos = cached
        for s, e, _ in fw:
            assert s >= pos
            pos = e
        assert pos == cached + n_p or n_p == 0
    return off, mer


def test_classes_cold_and_warm():
    for cached in (0, 27136):
        for n in CLASSES:
            check(cached, n)


def test_exhaustive_serial():
    for cached in (0, 27136):
        for n in range(2, 8194):
            check(cached, n)


def test_pipeline_windows_unchanged():
    # the pipeline never ends a window at a non-2,048 chunk, so merge only touches the serial tail
    for cached in (0, 27136):
        for n in CLASSES + list(range(6100, 6300, 7)) + [30000, 70001]:
            check(cached, n, pipeline = True)


def test_brief_examples():
    # R802 classes: forwards OFF -> MERGE
    expect = {258: (2, 1), 757: (2, 1), 2100: (2, 2), 2807: (3, 2), 4095: (3, 2), 4097: (2, 2), 513: (1, 1),
              257: (1, 1), 2049: (1, 1), 6144: (4, 3), 6145: (3, 3)}
    for n, (fo, fm) in expect.items():
        off, mer = check(27136, n)
        assert (len(off), len(mer)) == (fo, fm), (n, len(off), len(mer))
    # N 2,807 from S* = 27,136: [27136, 29184) then one merged forward [29184, 29942) split at 29696
    _, mer = check(27136, 2807)
    assert mer == [(27136, 29184, None), (29184, 29942, 29696)], mer


def test_split_point_rule():
    assert split_point(0, 756, 756) == 512
    assert split_point(0, 512, 512) is None            # page multiple: no leftover
    assert split_point(0, 2048, 2099) is None          # remainder does not fit: served cut at 2,048
    assert split_point(0, 200, 200) is None            # no full page
    assert split_point(512, 700, 700) is None          # last page = start: nothing before the split
    assert split_point(0, 1024, 1100) is None          # forward stopped early (shared page): not the prompt end


if __name__ == "__main__":
    _harness.run_tests(globals())
