"""Scenario runner: drives the REAL exllamav3 code of one tree (served or patched) on CPU.

    python mtp_window_scenarios.py <package_parent_dir> <scenario>

prints one JSON document. The environment (EXL3_MTP_KV_WINDOW, EXL3_QSA_RAWK_RING, ...) is read by
the package at import time, so every run is its own process. Only the extension and triton are
stubbed (stubs.py); the Cache, CacheLayer_quant + QSA planes, Generator.__init__, the MTP drafting
round (Generator.iterate_draftmodel_mtp_gen), the verify round with the MTP accept prefill
(Generator.iterate_gen), Job.prefill, PageTable.defrag, BCAttn.__init__ and BCAttn.step are the
tree's own functions, patched or not. Models, jobs and the sampler are fakes that record what the
real code hands them.

Every scenario runs on both trees; with the flag unset the patched tree must print exactly what
the served tree prints (test_mtp_kv_window_cpu.py compares them).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
sys.path.insert(0, str(HERE))
import stubs  # noqa: E402


class Recorder:
    """ext stand-in that records which tensors an ext call receives."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            self.calls.append((name, args))
            return stubs.Stub(name + "()")
        return call


EXT = Recorder()
stubs.install(EXT)

PAGE = 256
H = 8          # fake width of the target state / MTP carry
VOCAB = 64


# ---- fake modules ---------------------------------------------------------------------------------

class FakeIndexer:
    """QSA indexer geometry; sparse_threshold is the real QSAIndexer method."""

    def __init__(self, head_dim = 128, compress_ratio = 4, block_topk = 32):
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.block_topk = block_topk

    def sparse_threshold(self):
        from exllamav3.modules.qsa_indexer import QSAIndexer
        return QSAIndexer.sparse_threshold(self)


class FakeAttn:
    """What CacheLayer_quant / QSAPlanes read; the real Attention.cache_layer_type picks the layer."""

    def __init__(self, layer_idx, num_kv_heads = 2, head_dim = 256, indexer = None):
        self.layer_idx = layer_idx
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.qsa_indexer = indexer
        self.cache_layers = []
        self.recurrent_layers = []
        self.caps = {"kv_cache": True}

    def cache_layer_type(self, default, kwargs):
        from exllamav3.modules.attn import Attention
        return Attention.cache_layer_type(self, default, kwargs)


class FakeModel:
    def __init__(self, attn_layers, caps):
        self.config = SimpleNamespace(layer_map = None, vocab_size = VOCAB)
        self.caps = dict(caps)
        self.cache_weakrefs = {}
        self.recurrent_state_cls = None
        self.loaded_tp = False
        self.attn = attn_layers
        self.draft_verifier_params = {}

    def get_cache_layers(self):
        return self.attn

    def get_recurrent_layers(self):
        return []

    def get_layer_instances(self, layer_idx):
        from exllamav3.model.model import Model
        return Model.get_layer_instances(self, layer_idx)


class FakeLMHead:
    def prepare_for_device(self, x, params):
        return x


class FakeMainModel(FakeModel):
    """Target model: records forwards, exports a per-(row, position) state, and (verify) returns
    logits whose argmax is a scripted token per row and position."""

    def __init__(self):
        super().__init__([FakeAttn(i, num_kv_heads = 1, head_dim = 32) for i in (3, 7)],
                         {"recurrent_states": True})
        self.modules = [FakeLMHead()]
        self.logit_layer_idx = 0
        self.output_device = torch.device("cpu")
        self.script = None
        self.forwards = []

    def forward(self, input_ids, params):
        rows, q = input_ids.shape
        self.forwards.append({"shape": [rows, q], "cache_seqlens": params["cache_seqlens"].tolist()})
        state = (100 * torch.arange(rows).view(rows, 1, 1) + torch.arange(q).view(1, q, 1)).float()
        params["export_states"] = [state.expand(rows, q, H).clone()]
        if self.script is None:
            return None
        logits = torch.zeros((rows, q, VOCAB))
        for r in range(rows):
            for i in range(q):
                logits[r, i, self.script[r][i]] = 1.0
        return logits


class FakeDraftModel(FakeModel):
    """MTP draft head: records every forward / prefill with copies of its block table and lengths."""

    def __init__(self, indexer = None):
        super().__init__([FakeAttn(0, indexer = indexer or FakeIndexer())],
                         {"mtp_draft": True, "attach_target": True, "default_draft_size": 4,
                          "supports_tp": False})
        self.calls = []
        self.cache_ref = None

    def attach_to(self, target):
        pass

    def _record(self, kind, input_ids, params):
        th = params.get("target_hidden")
        self.calls.append({
            "kind": kind,
            "ids": input_ids.tolist(),
            "block_table": params["block_table"].tolist(),
            "cache_seqlens": params["cache_seqlens"].tolist(),
            "cache_is_draft": params["cache"] is self.cache_ref,
            "target_hidden": None if th is None else th[..., 0].tolist(),
        })

    def forward(self, input_ids, params):
        self._record("forward", input_ids, params)
        return torch.full((input_ids.shape[0], 1, H), 5.0)

    def prefill(self, input_ids, params):
        self._record("prefill", input_ids, params)

    def sample_from_state(self, state, params):
        return torch.full((state.shape[0], 1), 11, dtype = torch.long)


# ---- caches ---------------------------------------------------------------------------------------

def describe_cache(c) -> dict:
    layers = []
    for key, layer in sorted(c.layers.items()):
        layers.append({
            "key": list(key),
            "type": type(layer).__name__,
            "max_num_tokens": layer.max_num_tokens,
            "qshape_k": list(layer.qshape_k), "qshape_v": list(layer.qshape_v),
            "qshape_s": list(layer.qshape_s),
            "raw_k_shape": list(layer.raw_k_shape) if hasattr(layer, "raw_k_shape") else None,
            "pooled_shape": list(layer.pooled_shape) if hasattr(layer, "pooled_shape") else None,
            "storage_size": int(layer.storage_size()),
            "scan_tokens": getattr(layer, "mtp_window_scan_tokens", None),
        })
    w = getattr(c, "mtp_window", None)
    return {
        "max_num_tokens": c.max_num_tokens,
        "num_slots": c.num_slots,
        "window": None if w is None else {
            "window_tokens": w.window_tokens, "slots": w.slots, "pages_per_slot": w.pages_per_slot,
            "scan_tokens": w.scan_tokens, "pool_tokens": w.pool_tokens,
        },
        "layers": layers,
    }


def make_caches(pool_tokens, draft_slots = 4, main_slots = 4, draft_indexer = None):
    from exllamav3.cache.cache import Cache
    from exllamav3.cache.fp16 import CacheLayer_fp16
    from exllamav3.cache.quant import CacheLayer_quant
    main = FakeMainModel()
    main_cache = Cache(main, max_num_tokens = pool_tokens, layer_type = CacheLayer_fp16,
                       max_batch_size = main_slots)
    draft = FakeDraftModel(indexer = draft_indexer)
    draft_cache = Cache(draft, max_num_tokens = pool_tokens, layer_type = CacheLayer_quant,
                        max_batch_size = draft_slots, k_bits = 8, v_bits = 8)
    draft.cache_ref = draft_cache
    for c in (main_cache, draft_cache):
        for layer in c.layers.values():
            layer.alloc(torch.device("cpu"))
        c.initialized = True
    return main, main_cache, draft, draft_cache


# ---- generator ------------------------------------------------------------------------------------

class CPUTierRecorder:
    last = None

    def __init__(self, caches, size):
        CPUTierRecorder.last = caches

    def attach(self, pagetable):
        pass


def build_generator(pool_pages = 64, draft_slots = 4, main_slots = 4, max_batch_size = 8,
                    num_draft_tokens = 3, cpu_cache = True, nvme = True, chunk = 512):
    import exllamav3.generator.generator as G
    import exllamav3.generator.disk_cache as D
    main, main_cache, draft, draft_cache = make_caches(pool_pages * PAGE, draft_slots, main_slots)
    names = {id(main_cache): "main", id(draft_cache): "draft", id(None): None}
    rec = {}
    G.CPUPageCache = CPUTierRecorder
    CPUTierRecorder.last = None
    G._NVME_TIER = "/nvme-test" if nvme else None

    def install(generator, cache, dcache, path):
        rec["nvme"] = [names[id(cache)], names[id(dcache)]]
        return None
    D.DiskPageCache.install = staticmethod(install)
    gen = G.Generator(
        main, main_cache, tokenizer = None, max_batch_size = max_batch_size, max_chunk_size = chunk,
        draft_model = draft, draft_cache = draft_cache, num_draft_tokens = num_draft_tokens,
        cpu_cache_size = (1 << 20) if cpu_cache else 0,
    )

    # The served pinned staging (not touched by the patch), without pinning: no accelerator here
    def staging(name, rows, width = None, dtype = torch.int32):
        key = (name, width)
        buf = gen.staging_buffers.get(key)
        if buf is None or buf.shape[0] < rows:
            shape = (max(rows, 32),) if width is None else (max(rows, 32), width)
            buf = gen.staging_buffers[key] = torch.zeros(shape, dtype = dtype)
        return buf[:rows]
    gen._staging = staging
    if CPUTierRecorder.last is not None:
        rec["cpu_tier"] = [names[id(c)] for c in CPUTierRecorder.last]
    return gen, rec, main, main_cache, draft, draft_cache


def window_state(gen) -> dict:
    w = getattr(gen, "draft_window", None)
    slots = getattr(gen, "draft_window_slots", None)
    return {
        "window": None if w is None else repr(w),
        "leases": None if slots is None else sorted(slots.leases.values()),
        "max_batch_size": gen.max_batch_size,
    }


class FakeState:
    def __init__(self):
        self.rewinds = []

    def rewind(self, n):
        self.rewinds.append(n)


class FakeSeq:
    """A decoding sequence: real Sequence ids/hashes, fake page objects for the rewind path."""

    def __init__(self, pages, kv_position, prompt):
        from exllamav3.generator.pagetable import Sequence
        ids = torch.tensor([prompt], dtype = torch.long)
        real = Sequence(ids, ids)
        real.prepare(False, 64)
        self.page_hashes = real.page_hashes
        self.sequence_ids = real.sequence_ids
        self.block_index_tensor = torch.tensor([pages], dtype = torch.int32)
        self.kv_position = kv_position
        self.allocated_pages = [SimpleNamespace(kv_position = PAGE) for _ in pages]


class FakeJob:
    """Decode-phase job with the Job methods the drafting and verify rounds call."""

    def __init__(self, name, pages, kv_position, prompt, script_eos = (), script_rq = ()):
        self.name = name
        self.sequences = [FakeSeq(pages, kv_position, prompt)]
        self.mtp_last_hidden = torch.full((1, 1, H), 1.0)
        self.time_first_token = 1.0
        self.embeddings = []
        self.new_tokens = 0
        self.filters = []
        self.filter_futures = []
        self.logit_masks = []
        self.checkpoint_rewound = False
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.recurrent_state = FakeState()
        self.script_eos = set(script_eos)
        self.script_rq = set(script_rq)
        self.samples = 0
        self.deallocated = 0
        self.draft_stats = []

    def is_prefill_done(self):
        return True

    def get_max_seq_len(self):
        return self.sequences[0].kv_position + 1

    def get_input_ids_list(self, draft_tokens = None, idx = 0, add_to_cache = False):
        last = torch.tensor([[7]], dtype = torch.long)
        if draft_tokens is None:
            return [last]
        return [torch.cat((last, draft_tokens[idx:idx + 1]), dim = -1)]

    def prepare_logit_mask(self):
        pass

    def prepare_sampling_past_ids(self):
        pass

    def receive_logits(self, logits):
        return logits.argmax(dim = -1).view(1, 1), None, None, None

    def receive_sample(self, logits, next_token, *args):
        self.samples += 1
        self.new_tokens += 1
        return self.samples in self.script_eos, next_token, self.samples in self.script_rq

    def is_checkpoint_boundary(self, interval = None):
        return False

    def maybe_stash_recurrent(self, *args, **kwargs):
        pass

    def deallocate_pages(self):
        self.deallocated += 1

    def prepare_for_requeue(self):
        return self


def prompt_tokens(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (n,), generator = g).tolist()


# ---- scenarios ------------------------------------------------------------------------------------

def s_alloc():
    """Cache geometry at the served pool (not allocated): the MTP layer and 12 main QSA layers at
    8,8 with the served QSA dims (2 x 256 K/V, indexer 128-dim, compress ratio 4, top-k 512)."""
    from exllamav3.cache.cache import Cache
    from exllamav3.cache.quant import CacheLayer_quant
    out = {}
    pool = int(os.environ.get("SCENARIO_POOL", "966656"))
    slots = int(os.environ.get("SCENARIO_SLOTS", "8"))
    draft_bits = int(os.environ.get("SCENARIO_DRAFT_BITS", "8"))     # draft_cache_mode Q8 / Q6 / Q4
    for name, caps, n_layers, bits in (("draft", {"mtp_draft": True}, 1, draft_bits), ("main", {}, 12, 8)):
        idx = FakeIndexer(head_dim = 128, compress_ratio = 4, block_topk = 512)
        model = FakeModel([FakeAttn(i, num_kv_heads = 2, head_dim = 256, indexer = idx)
                           for i in range(n_layers)], caps)
        c = Cache(model, max_num_tokens = pool, layer_type = CacheLayer_quant, max_batch_size = slots,
                  k_bits = bits, v_bits = bits)
        out[name] = describe_cache(c)
    return out


def s_alloc_small_window():
    """A window whose ring cycle does not exceed the QSA sparse threshold must be refused."""
    from exllamav3.cache.cache import Cache
    from exllamav3.cache.quant import CacheLayer_quant
    model = FakeModel([FakeAttn(0, indexer = FakeIndexer(block_topk = 512))], {"mtp_draft": True})
    try:
        c = Cache(model, max_num_tokens = 64 * PAGE, layer_type = CacheLayer_quant, max_batch_size = 4,
                  k_bits = 8, v_bits = 8)
        return {"error": None, "max_num_tokens": c.max_num_tokens}
    except ValueError as e:
        return {"error": str(e)}


def s_gen_init():
    """Real Generator.__init__: tier cache lists, the slot check after the recurrent batch clamp."""
    out = {}
    gen, rec, *_ = build_generator(max_batch_size = 8, main_slots = 4, draft_slots = 4)
    out["clamped"] = {"tiers": rec, "state": window_state(gen)}
    try:
        gen, rec, *_ = build_generator(max_batch_size = 8, main_slots = 4, draft_slots = 2)
        out["too_few_slots"] = {"error": None, "tiers": rec, "state": window_state(gen)}
    except ValueError as e:
        out["too_few_slots"] = {"error": str(e)}
    return out


def s_draft_round():
    """Real Generator.iterate_draftmodel_mtp_gen over two decoding jobs (depth 3)."""
    gen, rec, main, main_cache, draft, draft_cache = build_generator()
    a = FakeJob("a", list(range(10, 22)), 2900, prompt_tokens(2901, 1))
    b = FakeJob("b", [30, 31, 32], 700, prompt_tokens(701, 2))
    gen.active_jobs = [a, b]
    ids = gen.iterate_draftmodel_mtp_gen([])
    return {"draft_ids": ids.tolist(), "calls": draft.calls, "state": window_state(gen)}


def s_verify_round():
    """Real Generator.iterate_gen verifying depth-3 drafts: a accepts all three, b accepts none,
    c stops at its first token (EOS). The MTP accept prefill (target-state repair of the accepted
    positions) must address the draft cache through the jobs' window rows; c's slot is released."""
    gen, rec, main, main_cache, draft, draft_cache = build_generator()
    a = FakeJob("a", list(range(10, 22)), 2900, prompt_tokens(2901, 1))
    b = FakeJob("b", [30, 31, 32], 700, prompt_tokens(701, 2))
    c = FakeJob("c", [40, 41], 300, prompt_tokens(301, 3), script_eos = {1})
    gen.active_jobs = [a, b, c]
    gen.pending_jobs = [SimpleNamespace(name = "pending")]   # keep the queue from draining
    windowed = getattr(gen, "draft_window", None) is not None
    before = {j.name: gen.draft_window_slot(j) for j in (a, b, c)} if windowed else {}
    draft_tokens = torch.tensor([[21, 22, 23], [31, 32, 33], [41, 42, 43]], dtype = torch.long)
    main.script = [[21, 22, 23, 24], [9, 9, 9, 9], [41, 42, 43, 44]]
    gen.iterate_gen([], draft_tokens)
    return {
        "slots_before": before,
        "calls": draft.calls,
        "active": [j.name for j in gen.active_jobs],
        "deallocated": {j.name: j.deallocated for j in (a, b, c)},
        "accepted": {j.name: j.accepted_draft_tokens for j in (a, b, c)},
        "carry": {j.name: j.mtp_last_hidden[..., 0].tolist() for j in (a, b)},
        "state": window_state(gen),
    }


def s_verify_two_rounds():
    """Two real verify rounds with the batch order swapped in between: the staged accept-prefill
    rows must follow the jobs (round 1: a, b; round 2: b, a; a accepts everything both times)."""
    gen, rec, main, main_cache, draft, draft_cache = build_generator()
    a = FakeJob("a", list(range(10, 22)), 2900, prompt_tokens(2901, 1))
    b = FakeJob("b", [30, 31, 32], 700, prompt_tokens(701, 2))
    gen.pending_jobs = [SimpleNamespace(name = "pending")]
    out = []
    for order, script in (([a, b], [[21, 22, 23, 24], [9, 9, 9, 9]]),
                          ([b, a], [[9, 9, 9, 9], [21, 22, 23, 24]])):
        gen.active_jobs = list(order)
        draft.calls.clear()
        rows = {"a": [21, 22, 23], "b": [31, 32, 33]}
        draft_tokens = torch.tensor([rows[j.name] for j in order], dtype = torch.long)
        main.script = script
        gen.iterate_gen([], draft_tokens)
        out.append([c["block_table"] for c in draft.calls])
    slots = getattr(gen, "draft_window_slots", None)
    return {"rounds": out, "slot_a": None if slots is None else slots.lookup(id(a))}


def s_job_prefill():
    """Real Job.prefill over a 1,500-token prompt in 512-token chunks (the last page split off for
    the recurrent checkpoint): every draft prefill uses the job's one window row."""
    from exllamav3.generator.job import Job
    from exllamav3.generator.pagetable import Sequence
    gen, rec, main, main_cache, draft, draft_cache = build_generator(chunk = 512)
    ids = torch.tensor([prompt_tokens(1500, 4)], dtype = torch.long)
    seq = Sequence(ids, ids)
    seq.prepare(False, 64)
    seq.allocated_pages = [gen.pagetable.all_pages[i] for i in range(20, 27)]
    seq.build_block_index_tensor()
    job = Job.__new__(Job)
    job.generator = gen
    job.pagetable = gen.pagetable
    job.sequences = [seq]
    job.time_first_prefill = None
    job.recurrent_state = None
    job.embeddings = []
    job.alt_rope_freqs = None
    job.cached_pages = 0
    job.cached_tokens = 0
    job.serial_number = 0
    job.identifier = None
    job.mtp_last_hidden = None
    job.maybe_stash_recurrent = lambda *a, **k: None
    gen.active_jobs = [job]
    chunks = 0
    while not seq.prefill_complete and chunks < 10:
        job.prefill([])
        chunks += 1
    return {"chunks": chunks, "kv_position": seq.kv_position, "calls": draft.calls,
            "main_forwards": main.forwards, "state": window_state(gen)}


def s_defrag():
    """Real PageTable.defrag: whose tensors are rotated."""
    gen, rec, main, main_cache, draft, draft_cache = build_generator()
    names = {}
    for n, c in (("main", main_cache), ("draft", draft_cache)):
        for t in c.get_all_tensors():
            names[t.data_ptr()] = n
    pt = gen.pagetable
    pt.access_serial = pt.last_defrag_serial + pt.max_pages * 8 + 1
    for i, page in enumerate(pt.all_pages):      # reverse the access order: every page must move
        page.access_serial = pt.max_pages - i
    EXT.calls.clear()
    pt.defrag()
    rotated = [names.get(args[0].data_ptr(), "?") for name, args in EXT.calls if name == "cache_rotate"]
    return {"rotated": sorted(set(rotated)), "calls": len(rotated)}


def s_bc_step():
    """Real BCAttn.step: regime, rope position and t_total handed to the graph. qsa_scan_cap is
    what BCAttn.__init__ sets (0 on a served cache layer; the served tree never reads it)."""
    from exllamav3.modules.attention_fn.bc_attn import BCAttn
    out = {}
    for label, cap in (("unwindowed", 0), ("window", 1280)):
        for bsz, ctx in ((1, 100), (1, 1000), (1, 5000), (1, 100000), (2, 100000)):
            bca = BCAttn.__new__(BCAttn)
            bca.qsa = True
            bca.qsa_threshold = 131
            bca.qsa_scan_cap = cap
            bca.slot_widths = {}
            bca._configure = lambda *a: None
            bca.hidden_size = 4
            bca.o_dtype = torch.half
            seen = {}

            class BC:
                def run(self, bsz_, q_len, x, y, seqlens, bt, position, positions, position_ids,
                        inv_freq, regime, t_total):
                    seen.update(regime = regime, t_total = t_total, position = position)
            bca.bc = BC()
            x = torch.zeros((bsz, 1, 4), dtype = torch.half)
            host = torch.tensor([ctx, 50][:bsz], dtype = torch.int32)
            bca.step(x, host, torch.zeros((bsz, 1), dtype = torch.int32), 0, None, None, None,
                     causal = True, host_seqlens = host)
            out[f"{label}_b{bsz}_{ctx}"] = dict(seen)
    return out


def s_bc_init():
    """Real BCAttn.__init__ over the real draft cache layer (windowed or not): the scan clamp."""
    import exllamav3.modules.attention_fn.bc_attn as B
    import exllamav3.modules.attention_fn.triton_paged as TP
    main, main_cache, draft, draft_cache = make_caches(64 * PAGE, draft_slots = 4)
    layer = next(iter(draft_cache.layers.values()))
    lin = SimpleNamespace(inner = SimpleNamespace(bc = stubs.Stub("bc"), default_out_dtype = torch.half,
                                                  bias = None), quant_type = "exl3", in_features = 64)
    norm = SimpleNamespace(weight = SimpleNamespace(data = torch.ones(8)), rms_norm_eps = 1e-6)
    idx = draft.attn[0].qsa_indexer
    idx.index_qk_proj = lin
    idx.q_layernorm = norm
    idx.k_layernorm = norm
    idx.n_heads = 4
    module = SimpleNamespace(
        device = "cpu", head_dim = 256, num_q_heads = 4, num_kv_heads = 2, hidden_size = 64,
        sm_scale = 0.1, sliding_window = -1, logit_softcapping = 0.0, q_proj = lin, k_proj = lin,
        v_proj = lin, o_proj = lin, g_proj = None, multi_kv = None, multi_qg = None, rope = None,
        qsa_indexer = idx, q_norm_tensor = None, k_norm_tensor = None, norm_eps = 1e-6,
        norm_constant_bias = 0.0, v_norm = None,
    )
    B.g_tensor_cache = SimpleNamespace(get = lambda *a, **k: torch.zeros(1))
    TP._get_h32 = lambda dev: torch.zeros(1)
    bca = B.BCAttn(module, layer.qk, layer.qv, layer.sk, layer.sv, 8, 8, qsa_layer = layer)
    return {"qsa_scan_cap": getattr(bca, "qsa_scan_cap", None), "threshold": bca.qsa_threshold,
            "pool_tokens": draft_cache.max_num_tokens}


def s_requeue_affinity():
    """Slot reuse across jobs: a finishes (or is requeued) at 2,900 tokens and its slot keeps tags
    for its whole committed pages; an unrelated job d gets another slot, cleared; a's continuation
    (same prompt prefix, like a requeue or a follow-up turn) gets a's slot back with exactly the
    tagged pages intact and the rest cleared."""
    gen, rec, main, main_cache, draft, draft_cache = build_generator()
    if getattr(gen, "draft_window", None) is None:
        return {"window": None}
    P = gen.draft_window.pages_per_slot
    long_prompt = prompt_tokens(2901, 1)
    a = FakeJob("a", list(range(10, 22)), 2900, long_prompt)
    b = FakeJob("b", [30, 31, 32], 700, prompt_tokens(701, 2))
    gen.active_jobs = [a, b]
    sa, sb = gen.draft_window_slot(a), gen.draft_window_slot(b)
    tensors = draft_cache.get_all_tensors()
    for t in tensors:                      # mark every page with its physical index + 1
        for p in range(t.shape[0]):
            t[p].fill_(p + 1)
    gen.active_jobs = [b]
    gen.release_draft_window_slot(a, keep = True)
    d = FakeJob("d", [50], 100, prompt_tokens(101, 9))
    gen.active_jobs = [b, d]
    sd = gen.draft_window_slot(d)
    a2 = FakeJob("a2", list(range(10, 23)), 2950, long_prompt + prompt_tokens(50, 5))
    gen.active_jobs = [b, d, a2]
    sa2 = gen.draft_window_slot(a2)
    intact = {}
    for s, name in ((sa2, "a2"), (sd, "d"), (sb, "b")):
        intact[name] = [all(bool((t[s * P + r] == s * P + r + 1).all()) for t in tensors) for r in range(P)]
    zeroed = {name: [all(bool((t[s * P + r] == 0).all()) for t in tensors) for r in range(P)]
              for s, name in ((sa2, "a2"), (sd, "d"))}
    # A job whose prompt differs inside page 9 must not reuse pages 9 and 10 (chained hashes)
    gen.active_jobs = [b]
    gen.release_draft_window_slot(a2, keep = True)
    gen.release_draft_window_slot(d, keep = True)
    for t in tensors:
        for p in range(t.shape[0]):
            t[p].fill_(p + 1)
    forked = list(long_prompt)
    forked[9 * PAGE + 17] = (forked[9 * PAGE + 17] + 1) % VOCAB
    e = FakeJob("e", list(range(10, 23)), 2950, forked + prompt_tokens(50, 6))
    gen.active_jobs = [b, e]
    se = gen.draft_window_slot(e)
    intact["e"] = [all(bool((t[se * P + r] == se * P + r + 1).all()) for t in tensors) for r in range(P)]
    return {"P": P, "sa": sa, "sb": sb, "sd": sd, "sa2": sa2, "se": se, "intact": intact,
            "zeroed": zeroed}


def s_ring_contract():
    """Window bookkeeping through the real window_offsets rows: random interleaved schedules of
    prefill chunks (page-aligned and not), MTP rounds (draft steps write T..T+d, then the committed
    part is rewritten by the accept prefill; rejected positions keep draft garbage), requeue
    (the same positions rewritten by a replay). After every operation, every position of the exact
    window E(T) = sink page + pages max(1, c-(P-3))..c below T (c = T // 256) must read back its own
    committed write, from inside the job's own slot run. Two broken maps are run as controls."""
    import random
    from exllamav3.cache.mtp_window import window_offsets
    P = int(os.environ.get("SCENARIO_P", "5"))

    def good(slot, width):
        return (window_offsets(P, width) + slot * P).tolist()

    def bad_short_ring(slot, width):
        return [slot * P + (0 if q == 0 else 1 + (q - 1) % (P - 2)) for q in range(width)]

    def bad_no_sink(slot, width):
        return [slot * P + q % P for q in range(width)]

    def run(mapper, seed):
        rng = random.Random(seed)
        store = {}
        out = {"violations": 0, "checks": 0, "out_of_run": 0, "first": None}
        jobs = [{"slot": s, "T": 0, "commit": {}} for s in range(3)]
        width = 64

        def write(job, a, b, tag):
            row = mapper(job["slot"], width)
            for p in range(a, b):
                phys = row[p // PAGE]
                if not job["slot"] * P <= phys < (job["slot"] + 1) * P:
                    out["out_of_run"] += 1
                store[(phys, p % PAGE)] = (job["slot"], p, tag)

        def check(job):
            T = job["T"]
            c = T // PAGE
            row = mapper(job["slot"], width)
            pages = {0} | set(range(max(1, c - (P - 3)), c + 1))
            for L in sorted(pages):
                for p in range(L * PAGE, min((L + 1) * PAGE, T)):
                    out["checks"] += 1
                    want = (job["slot"], p, job["commit"][p])
                    got = store.get((row[L], p % PAGE))
                    if got != want:
                        out["violations"] += 1
                        if out["first"] is None:
                            out["first"] = {"slot": job["slot"], "T": T, "p": p, "got": got, "want": want}

        for step in range(900):
            job = rng.choice(jobs)
            T = job["T"]
            if T >= (width - 2) * PAGE:
                continue
            op = rng.random()
            if op < 0.12:
                n = rng.choice([PAGE, 2 * PAGE, 512 + 37, 100])            # prefill chunk
                for p in range(T, T + n):
                    job["commit"][p] = ("c", step)
                write(job, T, T + n, ("c", step))
                job["T"] = T + n
            elif op < 0.14 and T > PAGE:
                r = rng.randint(1, min(T, 600))                            # requeue / replay
                for p in range(T - r, T):
                    job["commit"][p] = ("c", step)
                write(job, T - r, T, ("c", step))
            else:
                d = rng.choice([1, 3, 3, 15])                              # MTP round, depth d
                write(job, T, T + d + 1, ("draft", step))
                acc = rng.randint(0, d)
                for p in range(T, T + acc + 1):
                    job["commit"][p] = ("c", step)
                write(job, T, T + acc + 1, ("c", step))
                job["T"] = T + acc + 1
            check(job)
        return out

    return {
        "P": P,
        "offsets_cycle": sorted(window_offsets(P, P).tolist()),
        "offsets_range": [int(window_offsets(P, 4096).min()), int(window_offsets(P, 4096).max())],
        "good": [run(good, s) for s in range(3)],
        "bad_short_ring": run(bad_short_ring, 0),
        "bad_no_sink": run(bad_no_sink, 0),
    }


SCENARIOS = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith("s_")}


if __name__ == "__main__":
    torch.manual_seed(0)
    result = SCENARIOS[sys.argv[2]]()
    print(json.dumps(result, sort_keys = True, default = str))
