"""CPU tests for the TABBY side of rebase165 (no GPU, no network, no model).

Run with plain unittest (pytest is not in the image):
    TABBY_DIR=/app EXL3_TREE=<dir holding exllamav3/> python3 test_tabby_rebase165.py

Covers:
  * encode once / tokenize off-loop on top of TabbyAPI dfd3848's encode cache: the context-length check's encode is
    reused by generate_gen (one tokenizer call per request), a cache hit never hops to the tokenize worker, a long
    miss does, a short miss does not; TABBY_ENCODE_ONCE=0 bypasses the cache; multimodal prompts are never cached;
    the LRU bound holds; the cache bookkeeping is thread-safe; cache_trace sees each real encode with its call site
  * Cache kwargs: recurrent_dtype only for recurrent_state_fp32 (so Tabby never forces fp16 over the engine's
    EXL3_GDN_STATE_BF16 selector); max_history covers the daily's draft policy [[4, 3], [8, 2]] with depth 3
  * stream_generate call sites: no filter_trigger left anywhere (fc25e6a removed it), every caller's keywords are
    accepted by ExllamaV3Container.stream_generate, and tool_choice forcing maps its trigger to reasoning_phase
  * the daily's generated config.yml (launcher heredoc, tune268r2 values) validates, and no key of the model /
    draft_model / memory / sampling sections is silently ignored; the batch draft policy parses; reasoning tags
    left at "auto" resolve to <think> / </think> for qwen3_coder when the template or tokenizer confirms them
  * loop-think backstop survives on the request while upstream's reasoning-phase sampler copy is built
"""
import ast
import asyncio
import glob
import os
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _bootstrap  # noqa: E402


def _read(path):
    with open(path) as f:
        return f.read()


TREE, TABBY = _bootstrap.install()
_bootstrap.check_tree(TREE)
os.chdir(TABBY)

import torch  # noqa: E402

import common.model  # noqa: E402,F401  (import order the TabbyAPI tests use)
import backends.exllamav3.model as model_module  # noqa: E402
from backends.exllamav3.model import ExllamaV3Container  # noqa: E402
from common import tokenize_offloop  # noqa: E402


class FakeTokenizer:
    """Counts encodes; ids = one token per character."""

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def encode(self, text, add_bos=False, encode_special_tokens=True, embeddings=None):
        with self.lock:
            self.calls += 1
        n = len(text) + (1 if add_bos else 0)
        return torch.arange(n, dtype=torch.long).unsqueeze(0)


def make_container():
    c = ExllamaV3Container()
    c.tokenizer = FakeTokenizer()
    c.hf_model = SimpleNamespace(add_bos_token=lambda: False)
    c.bos_token_id = None
    c.max_seq_len = 1 << 20
    c.cache = SimpleNamespace(max_num_tokens=1 << 21)
    c.generator = SimpleNamespace(generator=SimpleNamespace(max_draft_tokens=3))
    return c


def params():
    return SimpleNamespace(add_bos_token=None)


class TraceRecorder:
    def __init__(self):
        self.events = []

    def __enter__(self):
        mod = types.ModuleType("exllamav3.cache_trace")
        mod.encoded = lambda ids, path: self.events.append((int(ids.shape[-1]), path))
        import exllamav3
        self._patches = [
            patch.dict(sys.modules, {"exllamav3.cache_trace": mod}),
            patch.object(exllamav3, "cache_trace", mod, create=True),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()


class EncodeOnceTests(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.pop(k, None) for k in (
            "TABBY_ENCODE_ONCE", "EXL3_TOKENIZE_OFFLOOP", "EXL3_TOKENIZE_OFFLOOP_MIN_CHARS")}

    def tearDown(self):
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def run_generate_encode(self, c, prompt, embeddings=None):
        return asyncio.run(c.encode_prompt_offloop(prompt, False, embeddings or [], trace_path="generate_gen"))

    def test_check_then_generate_encodes_once_and_never_hops_on_hit(self):
        c = make_container()
        prompt = "x" * 20000  # above the 12,000-char hop threshold
        with TraceRecorder() as tr:
            c.validate_context_length(prompt, params())
            hops = tokenize_offloop.hops
            ids = self.run_generate_encode(c, prompt)
        self.assertEqual(c.tokenizer.calls, 1)
        self.assertEqual(tokenize_offloop.hops, hops, "a cache hit must not pay the executor hop")
        self.assertEqual(ids.shape[-1], 20000)
        self.assertEqual(tr.events, [(20000, "validate_context_length")])

    def test_long_miss_hops_short_miss_inline(self):
        c = make_container()
        with TraceRecorder() as tr:
            hops = tokenize_offloop.hops
            self.run_generate_encode(c, "y" * 13000)
            self.assertEqual(tokenize_offloop.hops, hops + 1)
            self.run_generate_encode(c, "y" * 13000)  # now a hit: inline
            self.assertEqual(tokenize_offloop.hops, hops + 1)
            self.run_generate_encode(c, "short prompt")
            self.assertEqual(tokenize_offloop.hops, hops + 1)
        self.assertEqual(c.tokenizer.calls, 2)
        self.assertEqual([p for _, p in tr.events], ["generate_gen", "generate_gen"])

    def test_offloop_disabled_never_hops(self):
        os.environ["EXL3_TOKENIZE_OFFLOOP"] = "0"
        c = make_container()
        with TraceRecorder():
            hops = tokenize_offloop.hops
            self.run_generate_encode(c, "z" * 50000)
        self.assertEqual(tokenize_offloop.hops, hops)

    def test_encode_once_off_encodes_twice_and_caches_nothing(self):
        os.environ["TABBY_ENCODE_ONCE"] = "0"
        c = make_container()
        with TraceRecorder() as tr:
            c.validate_context_length("hello world", params())
            self.run_generate_encode(c, "hello world")
        self.assertEqual(c.tokenizer.calls, 2)
        self.assertEqual(len(c.encode_cache), 0)
        self.assertEqual([p for _, p in tr.events], ["validate_context_length", "generate_gen"])

    def test_multimodal_never_cached(self):
        c = make_container()
        emb = [object()]
        with TraceRecorder():
            self.run_generate_encode(c, "img prompt", embeddings=emb)
            self.run_generate_encode(c, "img prompt", embeddings=emb)
        self.assertEqual(c.tokenizer.calls, 2)
        self.assertEqual(len(c.encode_cache), 0)

    def test_lru_bound(self):
        c = make_container()
        with TraceRecorder():
            for i in range(model_module.ENCODE_CACHE_ENTRIES + 5):
                c.encode_prompt(f"p{i}", add_bos=False)
        self.assertEqual(len(c.encode_cache), model_module.ENCODE_CACHE_ENTRIES)
        self.assertIsNone(c.cached_prompt_ids(("p0", False, True)))
        last = model_module.ENCODE_CACHE_ENTRIES + 4
        self.assertIsNotNone(c.cached_prompt_ids((f"p{last}", False, True)))

    def test_cache_bookkeeping_is_thread_safe(self):
        c = make_container()
        errors = []

        def worker(seed):
            try:
                for i in range(3000):
                    text = f"k{(i * 7 + seed) % 5}"
                    c.encode_prompt(text, add_bos=False)
                    c.cached_prompt_ids((text, False, True))
            except Exception as ex:  # noqa: BLE001
                errors.append(ex)

        with TraceRecorder(), patch.object(model_module, "ENCODE_CACHE_ENTRIES", 2):
            threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(c.encode_cache), 2)

    def test_add_bos_follows_bos_token_id(self):
        # 999b546: a model without a BOS token gets none even when add_bos_token is on
        c = make_container()
        c.hf_model = SimpleNamespace(add_bos_token=lambda: True)
        with TraceRecorder():
            c.validate_context_length("abc", params())
        self.assertIn(("abc", False, True), c.encode_cache)
        c.bos_token_id = 1
        with TraceRecorder():
            c.validate_context_length("abc", params())
        self.assertIn(("abc", True, True), c.encode_cache)


class CreateCacheTests(unittest.TestCase):
    def make(self, fp32):
        c = make_container()
        c.draft_model = None
        c.ngram_match_min = 0
        c.draft_num_tokens = 3
        c.hybrid_draft_tokens = 0
        c.max_batch_size = 8
        c.cache_size = 819200
        c.draft_num_tokens_by_batch = [(4, 3), (8, 2)]
        c.recurrent_state_fp32 = fp32
        return c

    def kwargs(self, fp32, mode="8,8"):
        calls = []
        with patch.object(model_module, "Cache", lambda model, **kw: calls.append(kw) or kw):
            self.make(fp32).create_cache(mode, object())
        return calls[0]

    def test_default_leaves_recurrent_dtype_to_the_engine(self):
        kw = self.kwargs(False)
        self.assertNotIn("recurrent_dtype", kw)
        self.assertEqual(kw["max_history"], 3)
        self.assertEqual(kw["max_batch_size"], 8)
        self.assertEqual((kw["k_bits"], kw["v_bits"]), (8, 8))
        self.assertIs(kw["layer_type"], model_module.CacheLayer_quant)

    def test_fp32_passes_float(self):
        self.assertIs(self.kwargs(True)["recurrent_dtype"], torch.float)


class StreamGenerateSurfaceTests(unittest.TestCase):
    FILES = [
        "endpoints/OAI/utils/chat_completion.py",
        "endpoints/OAI/utils/tool_choice.py",
        "endpoints/OAI/utils/completion.py",
        "backends/exllamav3/model.py",
    ]
    # endpoints/Kobold/utils/generation.py passes abort_event=, which stream_generate has not taken since before
    # 53da791: an upstream Kobold bug, unrelated to the served paths (the daily serves the OAI endpoints only)

    def test_no_filter_trigger_left(self):
        hits = []
        for path in glob.glob("**/*.py", recursive=True):
            if path.startswith("tests"):
                continue
            tree = ast.parse(_read(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword) and node.arg == "filter_trigger":
                    hits.append(f"{path}:{node.value.lineno}")
                if isinstance(node, ast.arg) and node.arg == "filter_trigger":
                    hits.append(f"{path}:{node.lineno}")
        self.assertEqual(hits, [])

    def test_callers_match_signature(self):
        import inspect
        accepted = set(inspect.signature(ExllamaV3Container.stream_generate).parameters) - {"self"}
        self.assertIn("reasoning_phase", accepted)
        seen = 0
        for path in self.FILES:
            tree = ast.parse(_read(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and node.func.attr == "stream_generate":
                    seen += 1
                    bad = [k.arg for k in node.keywords if k.arg and k.arg not in accepted]
                    self.assertEqual(bad, [], f"{path}:{node.lineno}")
        self.assertGreaterEqual(seen, 4)

    def test_forced_generation_maps_trigger_to_reasoning_phase(self):
        from endpoints.OAI.utils.tool_choice import forced_tool_generation

        calls = []

        class MC:
            active_job_ids = {}

            def stream_generate(self, request_id, prompt, params, disconnect_handler=None,
                                mm_embeddings=None, reasoning_phase=None, label=None):
                calls.append(reasoning_phase)

                async def gen():
                    yield {"finish_reason": "stop", "eos_reason": "max_new_tokens", "prompt_tokens": 1,
                           "gen_tokens": 0}
                return gen()

        class P:
            max_tokens = 10
            grammar_string = "start: call"

            def model_copy(self, deep=False):
                return P()

        p = P()

        async def drain(reasoning_end):
            return [c async for c in forced_tool_generation(
                MC(), "r", "prompt", p, None, None, reasoning_end=reasoning_end, tool_start="<tool_call>",
                reasoning_cap=None, reasoning_message="", call_grammar=None, label="t")]

        asyncio.run(drain("</think>"))
        asyncio.run(drain(None))
        self.assertEqual(calls, [True, None])


class DailyConfigTests(unittest.TestCase):
    """The config block launch-flashnext-tabby.sh writes (tune268r2 defaults)."""

    DAILY = {
        "model": {
            "model_dir": "/models",
            "model_name": "qwen3.8-flash-next-exl3-2.68bpw-selfcal",
            "backend": "exllamav3",
            "max_seq_len": 262144,
            "cache_size": 819200,
            "cache_mode": "8,8",
            "max_batch_size": 8,
            "tensor_parallel": False,
            "tensor_parallel_backend": "native",
            "gpu_split": [30, 30],
            "gpu_split_auto": False,
            "cpu_moe_offload_layers": 0,
            "cpu_moe_split_experts": 0,
            "ngram_ram": False,
            "chunk_size": 2048,
            "output_chunking": True,
            "vision": True,
            "reasoning": True,
            "reasoning_budget_tokens": 32768,
            "tool_format": "qwen3_coder",
        },
        "draft_model": {
            "draft_mode": "mtp",
            "draft_num_tokens": 3,
            "draft_num_tokens_by_batch": [[4, 3], [8, 2]],
            "draft_gpu_split": [0, 32],
            "draft_cache_mode": "Q8",
            "dynamic_draft": False,
        },
        "memory": {"sysmem_recurrent_cache": 4096, "sysmem_kv_cache": 0},
        "sampling": {"override_preset": "qwen38_thinking"},
    }

    def test_validates_and_nothing_is_ignored(self):
        from common.config_models import TabbyConfigModel

        cfg = TabbyConfigModel.model_validate(self.DAILY)
        for section, values in self.DAILY.items():
            fields = type(getattr(cfg, section)).model_fields
            unknown = [k for k in values if k not in fields]
            self.assertEqual(unknown, [], f"{section}: keys the schema would silently ignore")
        self.assertEqual(cfg.draft_model.draft_num_tokens_by_batch, [(4, 3), (8, 2)])
        self.assertEqual(cfg.draft_model.sparse_draft_head, 0, "upstream sparse head must stay off by default")
        self.assertEqual(cfg.model.reasoning_start_token, "auto", "upstream default changed: re-check the daily")

    def test_reasoning_tags_auto_for_qwen3_coder(self):
        from endpoints.OAI.utils.tools import detect_reasoning_tags

        template = "{% if enable_thinking %}<think>\n{% endif %}...</think>..."
        self.assertEqual(detect_reasoning_tags(template, lambda t: False, "qwen3_coder")[0], ("<think>", "</think>"))
        tok = {"<think>", "</think>"}
        self.assertEqual(detect_reasoning_tags("", lambda t: t in tok, "qwen3_coder")[0], ("<think>", "</think>"))
        # Neither confirms: reasoning parsing is switched off (the risk the launcher pin avoids)
        self.assertIsNone(detect_reasoning_tags("", lambda t: False, "qwen3_coder")[0])


class LoopThinkTests(unittest.TestCase):
    def test_backstop_survives_reasoning_copy(self):
        from common.sampling import BaseSamplerRequest

        req = BaseSamplerRequest(loop_detect_window=800)
        self.assertEqual(req.get_stop_on_loop(), (800, 2))
        req._loop_backstop = (1600, 4)
        self.assertEqual(req.get_stop_on_loop(), (1600, 4))
        req.reasoning_params()  # upstream reasoning-phase copy (None when nothing is overridden)
        self.assertEqual(req.get_stop_on_loop(), (1600, 4))
        self.assertEqual(req.model_copy(deep=True).get_stop_on_loop(), (1600, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
