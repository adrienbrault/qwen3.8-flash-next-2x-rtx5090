"""
CPU tests of the generator's MTP drafting loop with fakes: which IDs reach the embedding, with which hints,
on which device; default-flag behaviour identical to the pristine baseline.

The device chain requires ``temp_hidden.is_cuda``; a Tensor subclass reports True on this CPU-only machine
(torch.cat propagates the subclass), everything else is real generator code from the overlay / baseline.
"""
import types
import weakref

import pytest
import torch

from _harness import load_generator

H = 8
N_KEEP = 100


class FakeCuda(torch.Tensor):
    @property
    def is_cuda(self):
        return True


class FakeEmbed:
    def __init__(self):
        self.can_calls = []

    def can_embed_device_ids(self, device):
        self.can_calls.append(device)
        return True


class FakeDraft:
    """Records every forward's IDs and embedding hints; drafts deterministic in-set IDs."""
    caps = {"mtp_draft": True}

    def __init__(self, embed):
        self._embed = embed
        self.target_embed = weakref.ref(embed)
        self.calls = []

    def forward(self, ids, params):
        self.calls.append({
            "ids": ids.clone(),
            "ids_obj": ids,
            "pruned": params.get("embed_pruned"),
            "host_ids": params.get("embed_pruned_host_ids"),
        })
        return torch.zeros((ids.shape[0], 1, H))

    def sample_from_state(self, state, params):
        prev = self.calls[-1]["ids"]
        return ((prev * 7 + 3) % N_KEEP).to(torch.long)


class FakeJob:
    def __init__(self, tok):
        self.tok = tok
        self.sequences = [types.SimpleNamespace(block_index_tensor = torch.zeros((1, 4), dtype = torch.int32), kv_position = 5)]
        self.mtp_last_hidden = torch.zeros((1, 1, H)).as_subclass(FakeCuda)
        self.time_first_token = 1.0
        self.embeddings = []

    def is_prefill_done(self): return True
    def get_max_seq_len(self): return 16
    def get_input_ids_list(self): return [torch.tensor([[self.tok]])]


def make_gen(gen_mod, toks, pruned_embed = "unset", window = 3):
    g = object.__new__(gen_mod.Generator)
    embed = FakeEmbed()
    g.draft_model = FakeDraft(embed)
    g.model = types.SimpleNamespace(
        logit_layer_idx = 0,
        modules = [types.SimpleNamespace(prepare_for_device = lambda x, p: x, device = torch.device("cpu"))],
    )
    g.active_jobs = [FakeJob(t) for t in toks]
    g.max_num_draft_tokens = window
    g.num_draft_tokens = window
    g.num_draft_tokens_by_batch = None
    g.draft_calibrator = None
    g.draft_cache = None
    g.draft_input_ids_pinned = torch.empty((8, 1), dtype = torch.long)
    g.draft_ids_pinned = torch.empty((8, window), dtype = torch.long)
    if pruned_embed != "unset":
        g._pruned_embed = pruned_embed
    return g, embed


def expected_chain(toks, window):
    ids = torch.tensor(toks).view(-1, 1)
    out = []
    for _ in range(window):
        ids = (ids * 7 + 3) % N_KEEP
        out.append(ids)
    return torch.cat(out, dim = 1)


TOKS = [5, 150_000, 99, 248_319]          # two verified tokens outside the 100-row keep set


@pytest.mark.parametrize("root", ["baseline", "overlay"])
def test_device_chain_default_flags_unchanged(root):
    gen = load_generator(root, EXL3_MTP_DEVICE_DRAFT = 1)
    g, embed = make_gen(gen, TOKS)
    out = g.iterate_draftmodel_mtp_gen([])
    assert torch.equal(out[:len(TOKS)], expected_chain(TOKS, 3))
    assert embed.can_calls == [torch.device("cpu")]                     # temp_hidden.device
    assert all(c["pruned"] is None and c["host_ids"] is None for c in g.draft_model.calls)
    assert len(g.draft_model.calls) == 3


def test_overlay_default_flags_trace_equals_baseline():
    traces = []
    for root in ("baseline", "overlay"):
        for flags in ({}, {"EXL3_MTP_DEVICE_DRAFT": 1}):
            gen = load_generator(root, **flags)
            g, embed = make_gen(gen, TOKS)
            out = g.iterate_draftmodel_mtp_gen([]).clone()
            traces.append((root, tuple(flags), out, [c["ids"] for c in g.draft_model.calls], len(embed.can_calls)))
    b_off, b_on, o_off, o_on = traces
    for b, o in ((b_off, o_off), (b_on, o_on)):
        assert torch.equal(b[2], o[2]) and b[4] == o[4]
        assert all(torch.equal(x, y) for x, y in zip(b[3], o[3]))


def test_pruned_chain_hints():
    gen = load_generator("overlay", EXL3_MTP_DEVICE_DRAFT = 1, EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    keep = torch.arange(N_KEEP)
    g, embed = make_gen(gen, TOKS, pruned_embed = (keep, torch.device("cpu")))
    out = g.iterate_draftmodel_mtp_gen([])
    assert torch.equal(out[:len(TOKS)], expected_chain(TOKS, 3))        # same drafted IDs
    assert embed.can_calls == []                                          # full-mirror gate never consulted
    c0, c1, c2 = g.draft_model.calls
    # position 0: device IDs + their host copy (the verified tokens, may be out of set)
    assert c0["pruned"] is True
    assert c0["host_ids"] is not None and c0["host_ids"].device.type == "cpu"
    assert torch.equal(c0["host_ids"], torch.tensor(TOKS).view(-1, 1))
    assert torch.equal(c0["ids"], c0["host_ids"])
    # positions >= 1: argmax IDs, in-set by provenance, no host copy, no readback
    for c in (c1, c2):
        assert c["pruned"] is True and c["host_ids"] is None
        assert int(c["ids"].max()) < N_KEEP


def test_pruned_flag_but_unavailable_falls_back_to_host_chain():
    gen = load_generator("overlay", EXL3_MTP_DEVICE_DRAFT = 1, EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    g, embed = make_gen(gen, TOKS, pruned_embed = None)
    out = g.iterate_draftmodel_mtp_gen([])
    assert torch.equal(out[:len(TOKS)], expected_chain(TOKS, 3))
    assert embed.can_calls == []                                          # no full mirror in pruned mode
    assert all(c["pruned"] is None and c["host_ids"] is None for c in g.draft_model.calls)
    # host chain: every position's IDs are the host staging buffer
    assert all(c["ids_obj"].data_ptr() == g.draft_input_ids_pinned.data_ptr() for c in g.draft_model.calls)


def test_prepare_pruned_embed_gates():
    gen = load_generator("overlay", EXL3_MTP_DEVICE_DRAFT = 1, EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    g, embed = make_gen(gen, TOKS)
    g.mtp_draft = True
    keep = torch.arange(N_KEEP)
    built = []

    def prepare(device, keep_ids):
        built.append((device, keep_ids))
        return types.SimpleNamespace(n = keep_ids.numel(), num_rows = 248320, prefix = True, nbytes = lambda: 1 << 20)

    embed.prepare_pruned_mirror = prepare
    g.draft_model.draft_head_keep_ids = lambda: keep
    lm = g.model.modules[0]
    lm.device = "cuda:1"
    assert g._prepare_pruned_embed() == (keep, torch.device("cuda:1"))
    assert built == [(torch.device("cuda:1"), keep)]                     # mirror on the lm_head device
    g.draft_calibrator = object()
    assert g._prepare_pruned_embed() is None
    g.draft_calibrator = None
    g.draft_model.draft_head_keep_ids = lambda: None
    assert g._prepare_pruned_embed() is None
    g.draft_model.draft_head_keep_ids = lambda: keep
    lm.device = torch.device("cpu")
    assert g._prepare_pruned_embed() is None
    lm.device = "cuda:1"
    embed.prepare_pruned_mirror = lambda d, k: None
    assert g._prepare_pruned_embed() is None
    g.mtp_draft = False
    assert g._prepare_pruned_embed() is None


def test_prepare_pruned_embed_needs_device_draft_flag():
    gen = load_generator("overlay", EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    g, _ = make_gen(gen, TOKS)
    g.mtp_draft = True
    assert g._prepare_pruned_embed() is None
