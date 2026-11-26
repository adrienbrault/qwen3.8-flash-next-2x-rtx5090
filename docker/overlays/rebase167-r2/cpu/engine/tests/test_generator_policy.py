"""rebase165: generator policy pieces extracted by AST from the checkout (no torch, no model).

- prefill scheduling defaults: served prefill flags -> full chunks, no cooldown (2ece09d1 fair scheduler neutral);
  explicit arguments and EXL3_PREFILL_FAIR=1 keep upstream's scheduler; unset environment = upstream defaults
- draft depth: batch-indexed policy (num_draft_tokens_by_batch) indexed by decode-ready job count, then the
  1.6.5 row ceiling (draft_max_rows)
- verification capacity covers the policy maximum and the hybrid window
"""
import ast
import types
import unittest

from _iso import functions, text

GEN = "exllamav3/generator/generator.py"
PAGE_SIZE = 256


def scheduler_defaults(environ, max_chunk_size=2048, shared_chunk_size=None, prefill_cooldown=None):
    ns = functions(GEN, ["served_prefill_schedule"], ns={"os": types.SimpleNamespace(environ=environ)})
    init = [n for n in ast.parse(text(GEN)).body if isinstance(n, ast.ClassDef) and n.name == "Generator"][0]
    init = [n for n in init.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"][0]
    body = init.body
    start = [i for i, n in enumerate(body) if isinstance(n, ast.If) and "served_prefill_schedule" in ast.unparse(n.test)]
    assert len(start) == 1
    # The served block, upstream's two defaults and the asserts that follow
    stmts = body[start[0]:start[0] + 5]
    src = "\n".join(ast.unparse(s) for s in stmts)
    assert "assert 0 < shared_chunk_size" in src and "assert prefill_cooldown >= 0" in src, src
    loc = dict(ns, PAGE_SIZE=PAGE_SIZE, max_chunk_size=max_chunk_size, shared_chunk_size=shared_chunk_size,
               prefill_cooldown=prefill_cooldown)
    exec(compile(src, GEN, "exec"), loc)
    return loc["shared_chunk_size"], loc["prefill_cooldown"]


class Scheduler(unittest.TestCase):

    def test_unset_environment_is_upstream(self):
        self.assertEqual(scheduler_defaults({}), (1024, 4))
        self.assertEqual(scheduler_defaults({}, max_chunk_size=2304), (1280, 4))

    def test_served_prefill_flags_neutralize_the_fair_scheduler(self):
        for env in ({"EXL3_LS_PREFILL_PIPELINE": "1"}, {"EXL3_PREFILL_WHOLE_PROMPT": "1"},
                    {"EXL3_LS_PREFILL_PIPELINE": "1", "EXL3_PREFILL_WHOLE_PROMPT": "1"}):
            self.assertEqual(scheduler_defaults(env), (2048, 0), env)

    def test_fair_opt_in_and_explicit_arguments_win(self):
        daily = {"EXL3_LS_PREFILL_PIPELINE": "1", "EXL3_PREFILL_WHOLE_PROMPT": "1"}
        self.assertEqual(scheduler_defaults(dict(daily, EXL3_PREFILL_FAIR="1")), (1024, 4))
        self.assertEqual(scheduler_defaults(daily, shared_chunk_size=512), (512, 0))
        self.assertEqual(scheduler_defaults(daily, prefill_cooldown=2), (2048, 2))

    def test_iterate_skips_sharing_at_full_chunk_and_no_cooldown(self):
        # iterate() only defers/shrinks prefill when shared < max_chunk or cooldown > 0
        src = ast.unparse(ast.parse(text(GEN)))
        self.assertIn("self.shared_chunk_size < self.max_chunk_size or self.prefill_cooldown > 0", src)


def drafter(policy, fallback=3, draft_max_rows=0, jobs=1, rows_per_job=1, hybrid=0):
    ns = functions(GEN, ["_get_draft_depth", "_round_draft_depth", "row_limited"], cls="Generator",
                   ns={"logger": types.SimpleNamespace(debug=lambda *a: None)})
    job = types.SimpleNamespace(is_prefill_done=lambda: True, sequences=[object()] * rows_per_job)
    pending = types.SimpleNamespace(is_prefill_done=lambda: False, sequences=[object()])
    g = types.SimpleNamespace(num_draft_tokens_by_batch=policy, num_draft_tokens=fallback,
                              draft_max_rows=draft_max_rows, active_jobs=[job] * jobs + [pending] * 2,
                              _draft_window_last=None, hybrid_draft_tokens=hybrid)
    for k in ("_get_draft_depth", "_round_draft_depth", "row_limited"):
        setattr(g, k, types.MethodType(ns[k], g))
    return g


class DraftDepth(unittest.TestCase):
    DAILY = ((4, 3), (8, 2))

    def test_daily_policy_by_decode_ready_jobs(self):
        for jobs, depth in ((1, 3), (4, 3), (5, 2), (8, 2), (9, 3), (16, 3)):
            g = drafter(self.DAILY, jobs=jobs)
            self.assertEqual(g._round_draft_depth(), depth, jobs)

    def test_no_policy_is_the_fallback(self):
        self.assertEqual(drafter(None, fallback=4, jobs=6)._round_draft_depth(), 4)

    def test_row_ceiling_applies_after_the_policy(self):
        g = drafter(self.DAILY, jobs=4, draft_max_rows=12)
        self.assertEqual(g.row_limited(g._round_draft_depth()), 2)      # 12 // 4 - 1
        g = drafter(self.DAILY, jobs=4, draft_max_rows=0)
        self.assertEqual(g.row_limited(g._round_draft_depth()), 3)

    def test_capacity_covers_policy_maximum_and_hybrid(self):
        src = ast.unparse(ast.parse(text(GEN)))
        self.assertIn("self.max_draft_tokens = max(self.max_num_draft_tokens, self.hybrid_draft_tokens)", src)
        self.assertIn("self.max_num_draft_tokens = max(self.num_draft_tokens, *(depth for _, depth in rules))", src)
        self.assertIn("max_q_size = max(self.max_draft_tokens + 1, max_q_size)", src)


class Refusals(unittest.TestCase):

    def test_nvme_tier_refused(self):
        src = text(GEN)
        self.assertIn('_NVME_TIER = os.environ.get("EXL3_NVME_TIER", "")', src)
        self.assertIn("EXL3_NVME_TIER is no longer supported", src)

    def test_mtp_kv_window_refused(self):
        src = text("exllamav3/cache/cache.py")
        self.assertIn("EXL3_MTP_KV_WINDOW is not supported by this build", src)
        for name in ("draft_window_rows", "release_draft_window_slot", "MTPWindowSlots"):
            self.assertNotIn(name, text(GEN))


class NativeRequests(unittest.TestCase):

    def test_block_tables_are_whole_tensors(self):
        # NATIVE R5: BCAttn.step asserts a 16-byte aligned block_table (upstream 2f2b3e62); a row slice
        # (table[a:b]) can start mid-allocation. Every "block_table" value in the generator stack is a whole tensor
        for rel in (GEN, "exllamav3/generator/job.py", "exllamav3/generator/prefill_pipeline.py"):
            for n in ast.walk(ast.parse(text(rel))):
                if isinstance(n, ast.Dict):
                    for k, v in zip(n.keys, n.values):
                        if isinstance(k, ast.Constant) and k.value == "block_table":
                            if isinstance(v, ast.Subscript):
                                self.assertEqual(ast.unparse(v), "params['block_table']", (rel, ast.unparse(v)))

    def test_sparse_head_and_pruned_head_are_exclusive(self):
        src = text(GEN)
        self.assertIn("sparse_draft_head and EXL3_MTP_HEAD_N are mutually exclusive", src)

    def test_pruned_branch_precedes_upstream_head(self):
        fn = [n for n in ast.walk(ast.parse(text("exllamav3/architecture/qwen4_exp_mtp.py")))
              if isinstance(n, ast.FunctionDef) and n.name == "sample_from_state"][0]
        src = ast.unparse(fn)
        self.assertLess(src.index("_MTP_HEAD_N"), src.index("lm_head_argmax"))


if __name__ == "__main__":
    unittest.main()
