"""rebase165: Generator construction with a stub model/cache (the upstream tests/generator/test_checkpoint_intervals.py
pattern), CPU only. Imports the whole package from EXL3_SRC (real extension when importable, else a no-op stub; nothing
launches). Checks the served constructor additions against the environment: unset = upstream behaviour; the daily
launcher's prefill/checkpoint keys = served grid, full-chunk scheduling and the R823 intervals; refused variables fail
at construction."""
import os
import unittest
import torch
from types import SimpleNamespace
from unittest.mock import Mock, patch

from _iso import have_torch, import_package

DAILY = {
    "EXL3_LS_PREFILL_PIPELINE": "1", "EXL3_PREFILL_MERGE": "1", "EXL3_STASH_ASYNC": "1",
    "EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP": "4096", "EXL3_RECURRENT_CHECKPOINT_TAIL_PP": "12288",
    "EXL3_RECURRENT_CHECKPOINT_INFORWARD": "1", "EXL3_PREFILL_WHOLE_PROMPT": "1", "EXL3_PREFILL_RESUMABLE": "1",
}
SERVED_KEYS = tuple(DAILY) + ("EXL3_PREFILL_FAIR", "EXL3_RECURRENT_CHECKPOINT_GRID", "EXL3_PROMPT_LOOKUP",
                              "EXL3_MTP_HEAD_N")

gen_mod = None
if have_torch():
    try:
        import_package()
        import exllamav3.generator.generator as gen_mod
    except Exception as e:  # noqa: BLE001
        print(f"(skipping: cannot import the package: {e!r})")
        gen_mod = None


@unittest.skipIf(gen_mod is None, "integrated exllamav3 package not importable")
class Construct(unittest.TestCase):

    def setUp(self):
        self._env = dict(os.environ)
        for k in SERVED_KEYS:
            os.environ.pop(k, None)
        self._nvme = gen_mod._NVME_TIER
        gen_mod._NVME_TIER = ""

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        gen_mod._NVME_TIER = self._nvme

    def make(self, **kwargs):
        model = SimpleNamespace(config = SimpleNamespace(vocab_size = 256), caps = {"recurrent_states": True})
        cache = SimpleNamespace(num_slots = 4, max_history = 3, reset_states = Mock(), recurrent_layers = {})
        with patch("exllamav3.generator.generator.PageTable", return_value = SimpleNamespace(max_pages = 16)), \
             patch("exllamav3.generator.generator.ThreadPoolExecutor"):
            return gen_mod.Generator(model, cache, tokenizer = None, **kwargs)

    def test_unset_environment_is_upstream(self):
        g = self.make(max_chunk_size = 2048)
        self.assertEqual((g.shared_chunk_size, g.prefill_cooldown), (1024, 4))
        self.assertIs(g.recurrent_checkpoint_grid, False)
        self.assertIsNone(g.pipeline)
        self.assertFalse(g.rolling_checkpoint)
        self.assertEqual(g.recurrent_checkpoint_interval_pp, 32768)
        self.assertEqual(g.recurrent_checkpoint_tail_pp, 0)
        self.assertEqual(g.max_num_draft_tokens, g.max_draft_tokens)

    def test_daily_environment(self):
        os.environ.update(DAILY)
        g = self.make(max_chunk_size = 2048)
        self.assertEqual((g.shared_chunk_size, g.prefill_cooldown), (2048, 0))
        self.assertIs(g.recurrent_checkpoint_grid, True)
        self.assertEqual(g.recurrent_checkpoint_interval_pp, 4096)
        self.assertEqual(g.recurrent_checkpoint_tail_pp, 12288)

    def test_grid_rounds_interval_up_to_chunk_only_under_the_grid(self):
        g = self.make(max_chunk_size = 2048, recurrent_checkpoint_interval_pp = 2304)
        self.assertEqual(g.recurrent_checkpoint_interval_pp, 2304)        # upstream: kept as given
        os.environ["EXL3_RECURRENT_CHECKPOINT_GRID"] = "1"
        g = self.make(max_chunk_size = 2048, recurrent_checkpoint_interval_pp = 2304)
        self.assertEqual(g.recurrent_checkpoint_interval_pp, 4096)        # served: ceil to a chunk multiple

    def test_fair_opt_in(self):
        os.environ.update(DAILY)
        os.environ["EXL3_PREFILL_FAIR"] = "1"
        g = self.make(max_chunk_size = 2048)
        self.assertEqual((g.shared_chunk_size, g.prefill_cooldown), (1024, 4))

    def test_nvme_tier_refused(self):
        gen_mod._NVME_TIER = "/nvme-tier"
        with self.assertRaisesRegex(ValueError, "EXL3_NVME_TIER"):
            self.make()

    def test_ngram_policy_capacity(self):
        empty = torch.empty
        # This checks capacity selection, with pageable staging in the CPU-only fixture.
        def cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty(*args, **kwargs)
        with patch.object(torch, "empty", side_effect=cpu_empty):
            g = self.make(num_draft_tokens = 2, ngram_match_min = 2, num_draft_tokens_by_batch = [[4, 3], [8, 2]])
        self.assertEqual((g.max_draft_tokens, g.max_num_draft_tokens), (3, 3))
        self.assertEqual(g.num_draft_tokens_by_batch, ((4, 3), (8, 2)))

    def test_prompt_lookup_counters_complete_without_adaptive_state(self):
        from exllamav3.generator.prompt_lookup import COUNTERS
        from exllamav3.generator.job import Job
        job = SimpleNamespace(prompt_lookup_adaptive = None, decode_steps = 0, lookup_checks = 0, lookup_hits = 0,
                              lookup_proposed = 0, lookup_accepted = 0)
        out = Job.prompt_lookup_counters(job)
        self.assertEqual(set(out), set(COUNTERS))


if __name__ == "__main__":
    unittest.main()
