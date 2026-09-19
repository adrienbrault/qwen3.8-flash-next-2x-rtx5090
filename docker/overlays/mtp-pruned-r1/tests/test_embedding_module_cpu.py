"""CPU tests of the Embedding module overlay: default-path identity with the baseline and the flag gates."""
import torch
from torch import nn

from _harness import load_embedding

V, D = 2048, 32


def make(mod, dtype = torch.half):
    g = torch.Generator().manual_seed(99)
    w = (torch.randn((V, D), generator = g) * 2).to(dtype)
    e = mod.Embedding(config = None, key = "model.language_model.embed_tokens", vocab_size = V, hidden_size = D)
    e.device = torch.device("cpu")
    e.embedding = nn.Embedding(V, D, device = "meta")
    e.embedding.weight = nn.Parameter(w)                # requires_grad=True, as Embedding.load builds it (embedding.py:62)
    return e, w


def run(e, ids, **params):
    with torch.inference_mode():
        return e.forward(ids, dict(params), out_dtype = torch.half)


def test_default_path_matches_baseline_host_ids():
    ids = torch.tensor([[3, V - 1, 1500, 0]])
    outs = []
    for root, flags in (("baseline", {}), ("overlay", {}), ("overlay", {"EXL3_EMBED_GPU": 1}),
                        ("overlay", {"EXL3_EMBED_GPU": 1, "EXL3_EMBED_GPU_PRUNED": 1})):
        mod = load_embedding(root, **flags)
        e, w = make(mod)
        outs.append(run(e, ids, embed_pruned = True, embed_pruned_host_ids = ids))
    for o in outs[1:]:
        assert torch.equal(o, outs[0])      # host IDs never reach the pruned branch


def test_flag_off_constants():
    mod = load_embedding("overlay")
    assert mod._EMBED_GPU_PRUNED is False and mod._EMBED_GPU is False
    e, _ = make(mod)
    assert e._pruned_mirror is None
    assert e.prepare_pruned_mirror("cuda:0", torch.arange(16)) is None


def test_can_embed_device_ids_unchanged_without_pruned():
    base = load_embedding("baseline", EXL3_EMBED_GPU = 1)
    eb, _ = make(base)
    over = load_embedding("overlay", EXL3_EMBED_GPU = 1)
    eo, _ = make(over)
    assert eb.can_embed_device_ids("cuda:1") == eo.can_embed_device_ids("cuda:1") == True
    small = load_embedding("overlay", EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_MAX_MB = 0)
    es, _ = make(small)
    assert es.can_embed_device_ids("cuda:1") is False


def test_pruned_mode_never_offers_full_mirror():
    mod = load_embedding("overlay", EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    e, _ = make(mod)
    assert e.can_embed_device_ids("cuda:1") is False


def test_prepare_pruned_mirror_gates():
    # needs both flags
    mod = load_embedding("overlay", EXL3_EMBED_GPU_PRUNED = 1)
    e, _ = make(mod)
    assert e.prepare_pruned_mirror("cuda:1", torch.arange(64)) is None
    mod = load_embedding("overlay", EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    e, _ = make(mod)
    # a non-CUDA target device is refused (this machine has no GPU, so the positive case is GPU-only)
    assert e.prepare_pruned_mirror("cpu", torch.arange(64)) is None
    assert e.prepare_pruned_mirror("cuda:1", None) is None
    # size cap applies to the pruned rows
    mod = load_embedding("overlay", EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1, EXL3_EMBED_GPU_MAX_MB = 0)
    e, _ = make(mod)
    assert e.prepare_pruned_mirror("cuda:1", torch.arange(64)) is None


def test_unload_clears_pruned_mirror():
    mod = load_embedding("overlay", EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    e, _ = make(mod)
    e._pruned_mirror = object()
    e.unload()
    assert e._pruned_mirror is None


def test_forward_output_is_cast_rows_not_raw_rows():
    """
    R522 regression (test-level): gpu_unit_pruned_embed.py compared Embedding.forward's output with raw table rows.
    With a bf16 table (embedding.py:55 loads with allow_bf16=True) forward returns fp16 rows, so that comparison
    fails on dtype alone. The GPU script now uses _refs.raw_rows for lookup-level and _refs.forward_reference
    for forward-level comparisons; this pins both contracts on the real module.
    """
    from _refs import raw_rows, forward_reference, bits_equal, describe
    mod = load_embedding("overlay", EXL3_EMBED_GPU = 1, EXL3_EMBED_GPU_PRUNED = 1)
    ids = torch.tensor([[V - 1], [5]])
    for dtype in (torch.bfloat16, torch.half):
        e, w = make(mod, dtype)
        out = run(e, ids)
        assert out.dtype == torch.half
        assert bits_equal(out, forward_reference(w, ids)), describe(out, forward_reference(w, ids))
        # the round-1 GPU check's comparison: only equal when the table is already fp16
        assert bits_equal(out, raw_rows(w, ids).view(2, 1, D)) == (dtype == torch.half)
