"""rebase165: checkpoint placement. The served grid ((pos - cached_base) % interval == 0, R823 interval/tail) under the
served prefill/checkpoint flags, upstream 2ece09d1's anchor rule (pos - last >= interval) otherwise. No torch."""
import types
import unittest

from _iso import load

cp = load("exllamav3.cache.checkpoint_policy")

# The daily launcher's checkpoint/prefill keys (ref/launch-flashnext-tabby.sh EXTRA_ENV)
DAILY = {
    "EXL3_LS_PREFILL_PIPELINE": "1", "EXL3_PREFILL_MERGE": "1", "EXL3_STASH_ASYNC": "1",
    "EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP": "4096", "EXL3_RECURRENT_CHECKPOINT_TAIL_PP": "12288",
    "EXL3_RECURRENT_CHECKPOINT_INFORWARD": "1", "EXL3_PREFILL_WHOLE_PROMPT": "1", "EXL3_PREFILL_RESUMABLE": "1",
}


def gen(grid, near=1024, pp=4096, tail=0, chunk=2048):
    return types.SimpleNamespace(recurrent_checkpoint_grid=grid, recurrent_checkpoint_interval=near,
                                 recurrent_checkpoint_interval_pp=pp, recurrent_checkpoint_tail_pp=tail,
                                 max_chunk_size=chunk)


def served_rule(g, pos, prompt_end, cached_base):
    # served job.py is_checkpoint_boundary (served:exllamav3/generator/job.py)
    return (pos - cached_base) % cp.interval(g, pos, prompt_end) == 0


def upstream_rule(g, pos, prompt_end, cached_base, last):
    # v1.6.5 job.py is_checkpoint_boundary (after the page-alignment test)
    step = g.recurrent_checkpoint_interval if pos >= prompt_end - g.max_chunk_size * 2 else \
        g.recurrent_checkpoint_interval_pp
    anchor = cached_base if last is None else last
    return pos - anchor >= step


class GridEnabled(unittest.TestCase):

    def test_unset_environment_follows_upstream(self):
        self.assertFalse(cp.grid_enabled({}))

    def test_daily_environment_keeps_the_served_grid(self):
        self.assertTrue(cp.grid_enabled(DAILY))

    def test_each_served_key_alone_selects_the_grid(self):
        for k in ("EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP", "EXL3_RECURRENT_CHECKPOINT_TAIL_PP"):
            self.assertTrue(cp.grid_enabled({k: "4096"}), k)
        for k in ("EXL3_RECURRENT_CHECKPOINT_INFORWARD", "EXL3_LS_PREFILL_PIPELINE", "EXL3_PREFILL_WHOLE_PROMPT",
                  "EXL3_PREFILL_MERGE"):
            self.assertTrue(cp.grid_enabled({k: "1"}), k)
            self.assertFalse(cp.grid_enabled({k: "0"}), k)

    def test_force(self):
        self.assertFalse(cp.grid_enabled(dict(DAILY, EXL3_RECURRENT_CHECKPOINT_GRID="0")))
        self.assertTrue(cp.grid_enabled({"EXL3_RECURRENT_CHECKPOINT_GRID": "1"}))
        with self.assertRaises(ValueError):
            cp.grid_enabled({"EXL3_RECURRENT_CHECKPOINT_GRID": "yes"})


class IsBoundary(unittest.TestCase):

    def positions(self):
        for prompt_end in (3000, 9000, 40000, 70001):
            for cached_base in (0, 256, 4096, 8192):
                for pos in range(cached_base, prompt_end + 1024, 256):
                    yield prompt_end, cached_base, pos

    def test_grid_matches_served_rule(self):
        for tail in (0, 12288):
            g = gen(True, tail=tail)
            for prompt_end, cached_base, pos in self.positions():
                for last in (None, cached_base, pos - 256):
                    self.assertEqual(cp.is_boundary(g, pos, prompt_end, cached_base, last),
                                     served_rule(g, pos, prompt_end, cached_base),
                                     (tail, prompt_end, cached_base, pos, last))

    def test_anchor_matches_upstream_rule(self):
        g = gen(False)
        for prompt_end, cached_base, pos in self.positions():
            for last in (None, cached_base, max(cached_base, pos - 1024), max(cached_base, pos - 4096)):
                self.assertEqual(cp.is_boundary(g, pos, prompt_end, cached_base, last),
                                 upstream_rule(g, pos, prompt_end, cached_base, last))

    def test_grid_positions_are_page_aligned(self):
        # served had no page test; the 1.6.5 caller returns False off-page first. Grid hits must be on pages
        g = gen(True, tail=12288)
        for prompt_end, cached_base, pos in self.positions():
            for off in (1, 17, 255):
                p = pos + off
                if cp.is_boundary(g, p, prompt_end, cached_base, None):
                    self.fail((p, prompt_end, cached_base))

    def test_tail_reads_need_interval(self):
        import os
        old = dict(os.environ)
        try:
            os.environ.pop("EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP", None)
            os.environ["EXL3_RECURRENT_CHECKPOINT_TAIL_PP"] = "12288"
            with self.assertRaises(ValueError):
                cp.read_tail(2048)
            os.environ["EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP"] = "4096"
            self.assertEqual(cp.read_tail(2048), 12288)
        finally:
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
