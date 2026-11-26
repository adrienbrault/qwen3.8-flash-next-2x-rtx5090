"""CPU regression tests for the served MTP batched-verification mask fix, ported to the rebase165 tree.

Run:  EXL3_SRC=<exllamav3 checkout> python3 -B test_maskfix.py -v   (or out/py/run-cpu-tests.sh <checkout>)

Ported from ref/served-maskfix/test_maskfix.py. Differences: the "today" predicate is the served
draft_overlap.py (fixtures/draft_overlap_served.py) instead of git HEAD; the merged consumption branch also
holds upstream 1.6.5's batchable speculative readback, which every run disables (_NO_BATCH_READBACK = True)
so the oracle is the serial loop and round 4 is the only batched path under test; the provenance tests check
the predicate diff against the served fixture instead of git state.

No torch, numpy, CUDA or model. The tests execute, unmodified and extracted by AST from the checkout:
  * the consumption branch of Generator.iterate_gen (generator.py, the `else:` arm that
    builds round4_verify), once with _BATCH_VERIFY off (the serial oracle) and once on;
  * the real Job.prepare_logit_mask, Job.receive_logits and Job.receive_sample (job.py);
  * verification_batch_mode / draw_sampling_seed from draft_overlap.py, served (today)
    and in the checkout (patched).
Tensors, the greedy sampler, the tokenizer pieces and the banned-string matcher (ext) are pure-Python
stand-ins. Greedy only: the "sampled" batch mode changes the RNG stream by design and is not an
identity oracle.
"""
import ast
import copy
from collections import defaultdict
from pathlib import Path
import random
import os
import types
import unittest

SRC = Path(os.environ.get("EXL3_SRC") or Path(__file__).resolve().parents[3] / "wt-py")
SERVED_DOV = Path(__file__).resolve().parent / "fixtures" / "draft_overlap_served.py"
GEN = "exllamav3/generator/generator.py"
JOB = "exllamav3/generator/job.py"
DOV = "exllamav3/generator/draft_overlap.py"
NEG_INF = float("-inf")
V = 100                   # fake vocabulary
A, B, AD, EOS = 10, 50, 51, 99
PIECES = [f"<{i}>" for i in range(V)]
PIECES[A], PIECES[B], PIECES[AD], PIECES[EOS] = "a", "b", "ad", "</s>"


def head_text(rel):
    assert rel == DOV
    return SERVED_DOV.read_text()


def work_text(rel):
    return (SRC / rel).read_text()


def functions(text, name):
    return [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]


def one_function(text, name):
    found = functions(text, name)
    assert len(found) == 1, (name, len(found))
    return found[0]


def predicates(text):
    ns = {}
    exec(compile(text, DOV, "exec"), ns)
    return ns


# --------------------------------------------------------------------------------------------
# Minimal nested-list tensor
# --------------------------------------------------------------------------------------------

class T:
    readbacks = 0

    def __init__(self, data, cuda=False, dtype="float"):
        self.data, self.is_cuda, self.dtype = data, cuda, dtype
        self.device = types.SimpleNamespace(type="cuda" if cuda else "cpu", index=0 if cuda else None)

    @property
    def shape(self):
        out, x = [], self.data
        while isinstance(x, list):
            out.append(len(x))
            x = x[0] if x else 0
        return tuple(out)

    def __getitem__(self, index):
        def get(data, idx):
            if not idx:
                return data
            i, *rest = idx
            if isinstance(i, slice):
                return [get(v, rest) for v in data[i]]
            return get(data[i], rest)   # int index keeps a view on the row list
        return T(get(self.data, list(index) if isinstance(index, tuple) else [index]), self.is_cuda, self.dtype)

    def __setitem__(self, index, value):
        rows, cols = index
        for row in self.data[rows]:
            for c in cols:
                row[c] = value

    def flat(self):
        def walk(x):
            return sum((walk(v) for v in x), []) if isinstance(x, list) else [x]
        return walk(self.data)

    def reshape(self, *shape):
        values = self.flat()
        if shape == (-1,):
            return T(values, self.is_cuda, self.dtype)

        def build(dims):
            if not dims:
                return values.pop(0)
            return [build(dims[1:]) for _ in range(dims[0])]
        return T(build(list(shape)), self.is_cuda, self.dtype)

    view = reshape

    def to(self, target, **kwargs):
        # A device copy is a real copy: later rewrites of the pinned host mask must not alias it.
        if target == "long":
            return T(copy.deepcopy(self.data), self.is_cuda, "long")
        return T(copy.deepcopy(self.data), target.type == "cuda", self.dtype)

    def eq(self, other):
        return T([a == b for a, b in zip(self.flat(), other.flat())], self.is_cuda)

    def copy_(self, other, **kwargs):
        self.data[:] = other.flat()
        return self

    def is_pinned(self):
        return not self.is_cuda

    def cpu(self):
        T.readbacks += int(self.is_cuda)
        return T(copy.deepcopy(self.data))

    def zero_(self):
        for row in self.data:
            row[:] = [0.0] * len(row)
        return self

    def clone(self):
        return T(copy.deepcopy(self.data), self.is_cuda, self.dtype)

    def item(self):
        values = self.flat()
        assert len(values) == 1
        return values[0]


class Held:
    """SeqTensor stand-in for held_tokens/held_probs/... and sequence_ids."""

    def __init__(self, items=None):
        self.items = list(items or [])

    def append(self, x):
        self.items.append(x.item() if isinstance(x, T) else x)

    def clone(self, drop=None):
        return Held(self.items[:len(self.items) - drop] if drop and drop <= len(self.items) else self.items)

    def slice(self, a, b):
        return Held(self.items[a:b])

    def torch(self):
        return T(list(self.items))

    def clear(self):
        self.items.clear()

    def truncate(self, n):
        del self.items[n:]

    def __len__(self):
        return len(self.items)

    def __bool__(self):
        return bool(self.items)


def partial_strings_match(buf, strings, _buffer):
    """ext.partial_strings_match stand-in: full match position, -2 partial suffix match, -1 none."""
    text = buf.decode("utf-32-le")
    full = [text.find(s) for s in strings if text.find(s) >= 0]
    if full:
        return min(full)
    for k in range(1, len(text) + 1):
        if any(s.startswith(text[-k:]) and k < len(s) for s in strings):
            return -2
    return -1


# --------------------------------------------------------------------------------------------
# Harness: real served consumption branch + real Job methods
# --------------------------------------------------------------------------------------------

def build_job_class(counters):
    job_src = work_text(JOB)
    methods = [one_function(job_src, n) for n in ("prepare_logit_mask", "receive_logits", "receive_sample")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
                        + methods, type_ignores=[])
    ns = dict(
        torch=types.SimpleNamespace(int32="int32", half="half", long="long",
                                    # 1.6.5 first-token NaN/Inf guard (receive_logits): finite stand-ins
                                    isfinite=lambda t: types.SimpleNamespace(all=lambda: True)),
        NonFiniteLogits=RuntimeError,
        np=types.SimpleNamespace(uint8="uint8", frombuffer=lambda b, dtype=None: b),
        ext=types.SimpleNamespace(partial_strings_match=partial_strings_match),
        PAGE_SIZE=256, time=types.SimpleNamespace(time=lambda: 0.0),
        tensor_hash_checksum=lambda *a: None, random_hash=lambda: None,
        draw_sampling_seed=predicates(work_text(DOV))["draw_sampling_seed"],
    )
    exec(compile(ast.fix_missing_locations(module), JOB, "exec"), ns)

    class Sampler:
        batch_verify_mode = "greedy"
        reqs_past_ids = False

        def forward(self, logits, past_ids, seed, tokenizer, logit_mask=None):
            counters["sampler_calls"] += 1
            mask = logit_mask.data[0] if logit_mask is not None else [0.0] * V   # (1, V) broadcast
            rows = logits.data[0]
            out = [max(range(V), key=lambda t: row[t] + mask[t]) for row in rows]
            counters["widths"].append(len(out))
            return T([out], True)

    class Job:
        prepare_logit_mask = ns["prepare_logit_mask"]
        receive_logits = ns["receive_logits"]
        receive_sample = ns["receive_sample"]

        def __init__(self, generator, new_tokens=5, min_new_tokens=0, banned_strings=(), checkpoint=None,
                     held_text=""):
            self.generator, self.sampler = generator, Sampler()
            self.name_policy = types.SimpleNamespace(fast=False)
            self.rng = random.Random(1234)
            self.new_tokens, self.min_new_tokens = new_tokens, min_new_tokens
            self.max_new_tokens, self.max_rq_tokens = 10_000, 10_000
            self.rq_margin = 0   # 1.6.5 requeue margin (job.py), irrelevant at max_rq_tokens 10_000
            self.forced_ids, self.forced_sample, self.filters, self.filters_suspended = None, False, [], False
            self.filter_futures, self.logit_masks = [], []
            self.prefix_token = None
            self.stop_tokens = {EOS}
            self.stop_tokens_list = [EOS]
            self.stop_strings_utf32_offsets = None
            self.banned_strings = [s.lower() for s in banned_strings]
            self.banned_strings_utf32_offsets = self.banned_strings or None   # consumed by the fake ext
            self.banned_strings_utf32_buffer = None
            self.checkpoint = checkpoint
            self.checkpoint_rewound = False
            self.device_logit_mask, self.pinned_logit_mask = None, None
            self.logits_device = types.SimpleNamespace(type="cuda")
            self.current_device_ids = None
            self.return_probs, self.return_top_tokens, self.return_logits = False, 0, False
            self.decode_special_tokens = True
            self.held_text = held_text
            self.held_tokens, self.held_probs, self.held_k_tokens = Held(), Held(), Held()
            self.held_k_probs, self.held_logits = Held(), Held()
            self.full_completion = ""
            self.loop_detector = None
            self.recurrent_state = None
            seq = types.SimpleNamespace(sequence_ids=Held(), kv_position=10, allocated_pages=[], input_ids=[0])
            self.sequences = [seq]
            self.serial_number, self.identifier = 0, None
            for name in ("time_last_token", "time_enqueued", "time_first_prefill", "time_enqueue",
                         "time_prefill", "time_first_token", "time_generate"):
                setattr(self, name, 0.0)
            self.rq_cached, self.cached_pages, self.cached_tokens = None, 0, 0
            self.rq_new_tokens, self.rq_prompt_tokens = 0, 1
            self.accepted_draft_tokens = self.rejected_draft_tokens = self.lookup_accepted = 0
            self.prompt_lookup_sam = None
            self.prompt_lookup_adaptive = types.SimpleNamespace(metrics=defaultdict(int))
            self.draft_stats = []

        def get_pinned_logit_mask(self):
            # One reused pinned host buffer per job, as job.py:338
            if self.pinned_logit_mask is None:
                self.pinned_logit_mask = T([[0.0] * V], dtype="half")
            return self.pinned_logit_mask

        def prepare_sampling_past_ids(self):
            pass

        def hash_deferred_pages(self):
            # 1.6.5: pages a held checkpoint deferred are hashed at EOS/requeue (no pages here)
            pass

    return Job


def consumption_branch():
    fn = one_function(work_text(GEN), "iterate_gen")
    branch = [n for n in fn.body if isinstance(n, ast.If) and any(
        isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "round4_verify" for t in x.targets)
        for x in n.orelse)]
    assert len(branch) == 1
    return branch[0].orelse


def run_window(batch_verify, predicate_src, logits_row, depth, max_depth=None, **job_kw):
    """One MTP verification window (q = depth + 1 positions, every proposal = A)."""
    counters = dict(sampler_calls=0, waits=0, widths=[])
    Job = build_job_class(counters)
    tokenizer = types.SimpleNamespace(get_id_to_piece_list=lambda special: PIECES)
    gen = types.SimpleNamespace(
        tokenizer=tokenizer, draft_model=True, mtp_draft=True,
        max_num_draft_tokens=depth if max_depth is None else max_depth,
        _staging=lambda key, rows, cols, dtype: T([[0] * cols for _ in range(rows)]),
        verify_readback_events={}, prompt_lookup_enabled=False, _prompt_lookup_round={},
        record_draft_stats=False,
        model=types.SimpleNamespace(prefetch_tokens=lambda ids: None),
    )
    job = Job(gen, **job_kw)
    gen.active_jobs = [job]
    job.prepare_logit_mask()                         # generator.py:1615
    q = depth + 1
    proposals = T([[A] * depth])

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            counters["waits"] += 1

    def reject_remainder(job_, j_, i_, states_):
        n = q - 1 - i_
        job_.rejected_draft_tokens += n
        return n

    preds = predicates(predicate_src)
    ns = dict(
        self=gen, batch_logits=T([[list(logits_row) for _ in range(q)]], True), draft_tokens=proposals,
        logit_mapping=[0, 1], results=[], completed_jobs=[], requeuing_jobs=[], accepted_lengths=[],
        rewound_jobs=set(), batch_states=None, j=0, reject_remainder=reject_remainder,
        torch=types.SimpleNamespace(long="long", cat=lambda ts: T(sum((t.flat() for t in ts), []), True),
                                    cuda=types.SimpleNamespace(current_stream=lambda _: None, Event=Event)),
        _BATCH_VERIFY=batch_verify, DECODE_OVERLAP=False, _NO_BATCH_READBACK=True,
        verification_batch_mode=preds["verification_batch_mode"],
        common_verification_batch_key=preds["common_verification_batch_key"],
        draw_sampling_seed=preds["draw_sampling_seed"],
    )
    T.readbacks = 0
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(consumption_branch()), type_ignores=[])),
                 GEN, "exec"), ns)
    def plain(v):
        if isinstance(v, T):
            return ("tensor", v.flat())
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        return v
    stream = [{k: plain(v) for k, v in r.items() if k != "job"} for r in ns["results"]]
    return dict(
        ids=list(job.sequences[0].sequence_ids.items),
        accepted=job.accepted_draft_tokens, rejected=job.rejected_draft_tokens,
        eos=any(r.get("eos") for r in ns["results"]), stream=stream,
        new_tokens=job.new_tokens, checkpoint=copy.deepcopy(
            None if job.checkpoint is None else {k: (v.items if isinstance(v, Held) else v)
                                                 for k, v in job.checkpoint.items()}),
        rng=job.rng.getstate(), batched=counters["waits"] == 1 and T.readbacks == 0,
        widths=counters["widths"],
    )


def row(scores):
    r = [-1000.0] * V
    for t, s in scores.items():
        r[t] = s
    return r


EOS_FIRST = row({EOS: 2.0, A: 1.0})        # EOS ranks first, then "a"
B_FIRST = row({B: 2.0, A: 1.0})            # "b" ranks first, then "a"
A_FIRST = row({A: 2.0, B: 1.0, EOS: 0.5})  # unconstrained: the proposal is the argmax


def retry_checkpoint():
    # State left by rewind_checkpoint (job.py:955-966) after "b"+"ad" completed the banned "bad"
    return {"offset": 0, "explored_tokens": [B], "held_text": "", "held_tokens": Held(), "held_probs": Held(),
            "held_k_tokens": Held(), "held_k_probs": Held(), "held_logits": Held()}


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------

class MaskFixTests(unittest.TestCase):
    HEAD = None
    WORK = None

    @classmethod
    def setUpClass(cls):
        cls.HEAD = head_text(DOV)
        cls.WORK = work_text(DOV)

    def triple(self, logits_row, depth, **kw):
        serial = run_window(False, self.WORK, logits_row, depth, **kw)
        today = run_window(True, self.HEAD, logits_row, depth, **kw)
        fixed = run_window(True, self.WORK, logits_row, depth, **kw)
        self.assertFalse(serial["batched"])
        return serial, today, fixed

    def test_min_tokens_boundary_inside_d3_window_matches_serial(self):
        # new_tokens=5; the minimum is reached at window position `pending` (1..3) of a d3 window.
        for pending in (1, 2, 3):
            with self.subTest(pending=pending):
                serial, today, fixed = self.triple(EOS_FIRST, 3, new_tokens=5, min_new_tokens=5 + pending)
                self.assertEqual(serial["ids"], [A] * pending + [EOS])
                self.assertTrue(serial["eos"])
                # Served today: EOS stays suppressed for the whole window
                self.assertTrue(today["batched"])
                self.assertEqual(today["ids"], [A] * 4)
                self.assertFalse(today["eos"])
                self.assertNotEqual(today["ids"], serial["ids"])
                # Fixed: serial path, identical results incl. stream events, counters and RNG
                self.assertFalse(fixed["batched"])
                self.assertEqual(fixed, serial)

    def test_min_tokens_boundary_d5_d7(self):
        for depth in (5, 7):
            for pending in range(1, depth + 1):
                with self.subTest(depth=depth, pending=pending):
                    serial, today, fixed = self.triple(EOS_FIRST, depth, new_tokens=0, min_new_tokens=pending)
                    self.assertEqual(serial["ids"], [A] * pending + [EOS])
                    self.assertEqual(today["ids"], [A] * (depth + 1))
                    self.assertEqual(fixed, serial)

    def test_min_tokens_beyond_window_stays_batched_and_identical(self):
        # Static mask: EOS is suppressed at every window position on both paths.
        for depth in (3, 5, 7):
            for pending in (depth + 1, depth + 2, 100):
                with self.subTest(depth=depth, pending=pending):
                    serial, today, fixed = self.triple(EOS_FIRST, depth, new_tokens=5, min_new_tokens=5 + pending)
                    self.assertTrue(fixed["batched"])
                    self.assertEqual(fixed["widths"], [depth + 1])
                    self.assertEqual(fixed, today)
                    self.assertEqual({k: v for k, v in fixed.items() if k not in ("batched", "widths")},
                                     {k: v for k, v in serial.items() if k not in ("batched", "widths")})
                    self.assertEqual(fixed["ids"], [A] * (depth + 1))
                    self.assertFalse(fixed["eos"])

    def test_min_tokens_veto_uses_policy_maximum_depth(self):
        # A shallow window (d1) under a deeper policy ceiling (3) is conservatively serial; still identical.
        serial, today, fixed = self.triple(EOS_FIRST, 1, max_depth=3, new_tokens=5, min_new_tokens=8)
        self.assertFalse(fixed["batched"])
        self.assertEqual(fixed, serial)
        self.assertEqual(serial["ids"], [A, A])

    def test_banned_string_retry_clearing_matches_serial(self):
        for depth in (3, 5, 7):
            with self.subTest(depth=depth):
                serial, today, fixed = self.triple(B_FIRST, depth, banned_strings=["bad"],
                                                   checkpoint=retry_checkpoint())
                # Serial: "b" blocked at offset 0 -> "a"; "a" clears the checkpoint (job.py:996);
                # the next position samples "b" again (partial match, new checkpoint offset 1)
                self.assertEqual(serial["ids"], [A, B])
                self.assertEqual((serial["accepted"], serial["rejected"]), (1, depth - 1))
                self.assertEqual(serial["checkpoint"]["offset"], 1)
                # Served today: "b" stays blocked for the whole window
                self.assertTrue(today["batched"])
                self.assertEqual(today["ids"], [A] * (depth + 1))
                self.assertIsNone(today["checkpoint"])
                self.assertFalse(fixed["batched"])
                self.assertEqual(fixed, serial)

    def test_banned_strings_without_offset0_checkpoint_stay_batched(self):
        cases = dict(
            no_checkpoint=dict(banned_strings=["bad"]),
            holding_offset1=dict(banned_strings=["bad"], held_text="b",
                                 checkpoint=dict(retry_checkpoint(), offset=1)),
        )
        for name, kw in cases.items():
            for depth in (3, 7):
                with self.subTest(case=name, depth=depth):
                    serial, today, fixed = self.triple(A_FIRST, depth, **kw)
                    self.assertTrue(fixed["batched"])
                    self.assertEqual(fixed, today)
                    self.assertEqual({k: v for k, v in fixed.items() if k not in ("batched", "widths")},
                                     {k: v for k, v in serial.items() if k not in ("batched", "widths")})

    def test_unconstrained_request_takes_identical_batched_path(self):
        for depth in (1, 2, 3, 5, 7):
            for logits_row in (A_FIRST, EOS_FIRST, B_FIRST):
                with self.subTest(depth=depth, top=max(range(V), key=lambda t: logits_row[t])):
                    serial, today, fixed = self.triple(logits_row, depth)
                    self.assertTrue(fixed["batched"])
                    self.assertEqual(fixed["widths"], [depth + 1])
                    self.assertEqual(fixed, today)
                    self.assertEqual({k: v for k, v in fixed.items() if k not in ("batched", "widths")},
                                     {k: v for k, v in serial.items() if k not in ("batched", "widths")})

    def test_predicate_differs_from_head_only_on_the_two_vetoes(self):
        head, work = predicates(self.HEAD)["verification_batch_mode"], predicates(self.WORK)["verification_batch_mode"]
        checkpoints = (None, {"offset": 0, "explored_tokens": [B]}, {"offset": 1, "explored_tokens": [B]},
                       {"offset": 4, "explored_tokens": [B]})
        n = vetoed = 0
        for mode in ("greedy", "sampled", "other"):
            for depth in (1, 3, 7):
                for new_tokens in (-1, 0, 5):
                    for min_new in (0, 1, 3, 5, 6, 8, 9, 13, 100):
                        for cp in checkpoints:
                            for mask in (None, object()):
                                for filters in ([], [object()]):
                                    job = types.SimpleNamespace(
                                        sampler=types.SimpleNamespace(batch_verify_mode=mode, reqs_past_ids=False),
                                        new_tokens=new_tokens, forced_ids=None, filters=filters,
                                        device_logit_mask=mask, return_probs=False, return_top_tokens=0,
                                        sequences=[object()], checkpoint=cp, min_new_tokens=min_new,
                                        generator=types.SimpleNamespace(max_num_draft_tokens=depth))
                                    a, b = head(job), work(job)
                                    pending = min_new - new_tokens
                                    veto = bool(cp and cp["offset"] == 0) or 0 < pending <= depth
                                    n += 1
                                    if a is not None and veto:
                                        vetoed += 1
                                        self.assertIsNone(b)
                                    else:
                                        self.assertEqual(a, b)
        self.assertGreater(vetoed, 0)
        self.assertGreater(n, 1000)

    def test_only_the_predicate_differs_from_served(self):
        head_mod, work_mod = ast.parse(self.HEAD), ast.parse(self.WORK)
        strip = lambda m: [n for n in m.body if not (isinstance(n, ast.FunctionDef) and n.name == "verification_batch_mode")]
        self.assertEqual([ast.dump(n) for n in strip(head_mod)], [ast.dump(n) for n in strip(work_mod)])
        self.assertEqual(len(strip(head_mod)), len(head_mod.body) - 1)
        self.assertNotEqual(ast.dump(one_function(self.HEAD, "verification_batch_mode")),
                            ast.dump(one_function(self.WORK, "verification_batch_mode")))

    def test_round4_rows_never_take_the_upstream_batchable_readback(self):
        # A row sampled by round 4 (or decode overlap) must not be sampled again by 1.6.5's batchable pass
        branch = ast.Module(body=consumption_branch(), type_ignores=[])
        found = []
        for n in ast.walk(branch):
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "batchable" for t in n.targets):
                found.append(ast.unparse(n.value))
        self.assertEqual(len(found), 1)
        self.assertIn("id(job) not in round4_verify", found[0])
        self.assertIn("batched_verify_tokens is None", found[0])
        self.assertIn("not _NO_BATCH_READBACK", found[0])

    def test_generator_exposes_the_policy_maximum_as_max_num_draft_tokens(self):
        # The predicate bounds the min_new_tokens veto by generator.max_num_draft_tokens: it must alias the
        # capacity (policy maximum, hybrid window included), not the fallback depth
        init = [n for n in ast.walk(ast.parse(work_text(GEN))) if isinstance(n, ast.FunctionDef) and n.name == "__init__"][0]
        src = ast.unparse(init)
        self.assertIn("self.max_draft_tokens = max(self.max_num_draft_tokens, self.hybrid_draft_tokens)", src)
        self.assertIn("self.max_num_draft_tokens = self.max_draft_tokens", src)


if __name__ == "__main__":
    unittest.main()
