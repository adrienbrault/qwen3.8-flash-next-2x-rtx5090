"""CPU tests: remap table, pruned mirror contents, in-set device gather, out-of-set host fallback, bit-identity."""
import pytest
import torch
import torch.nn.functional as F

from _harness import load_pruned_helper

V, D = 4096, 64          # small stand-in for 248,320 x 2,560
N = 1024                 # pruned head width (prefix), stand-in for 65,536


@pytest.fixture(scope = "module")
def ep():
    return load_pruned_helper()


def table(dtype):
    g = torch.Generator().manual_seed(1234)
    w = torch.randn((V, D), generator = g, dtype = torch.float32) * 3.0
    # include exact extremes / subnormal-ish values so any accidental arithmetic would show
    w[7, :4] = torch.tensor([65504.0, -65504.0, 6e-8, -0.0])
    return w.to(dtype)


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_remap_prefix(ep):
    remap = ep.build_remap(torch.arange(N), V)
    assert remap.dtype == torch.int32 and remap.shape == (V,)
    assert torch.equal(remap[:N], torch.arange(N, dtype = torch.int32))
    assert bool((remap[N:] == -1).all())


def test_remap_general(ep):
    keep = torch.tensor([5, 3, 4000, 17])
    remap = ep.build_remap(keep, V)
    assert remap[5] == 0 and remap[3] == 1 and remap[4000] == 2 and remap[17] == 3
    assert int((remap >= 0).sum()) == 4
    assert remap[0] == -1 and remap[V - 1] == -1


@pytest.mark.parametrize("bad", [torch.tensor([1, 1]), torch.tensor([-1, 2]), torch.tensor([V]), torch.tensor([], dtype = torch.long)])
def test_remap_rejects(ep, bad):
    with pytest.raises(ValueError):
        ep.build_remap(bad, V)


def test_is_prefix(ep):
    assert ep.is_prefix(torch.arange(N))
    assert not ep.is_prefix(torch.arange(1, N + 1))
    assert not ep.is_prefix(torch.tensor([0, 2]))


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_mirror_rows_exact(ep, dtype):
    w = table(dtype)
    pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
    assert pm.prefix and pm.remap is None
    assert bits_equal(pm.mirror, w[:N])          # CPU copy aliases w; the real device copy is checked by gpu_unit_pruned_embed.py
    assert pm.nbytes() == N * D * w.element_size()


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_in_set_gather_bit_identical(ep, dtype):
    w = table(dtype)
    pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
    ids = torch.tensor([[0], [N - 1], [7], [N // 2]])            # (bsz, 1) like the draft chain
    ref = F.embedding(ids, w)                                      # the full-mirror path
    got = pm.lookup(w, ids, None)                                   # idx >= 1: in-set by provenance
    assert bits_equal(got, ref)
    got0 = pm.lookup(w, ids, ids.clone())                           # idx == 0, host copy says all in set
    assert bits_equal(got0, ref)


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_out_of_set_host_fallback_bit_identical(ep, dtype):
    w = table(dtype)
    pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
    before = dict(ep.stats)
    # verified tokens: boundary N-1 (in), N (first out), V-1 (last row), a mid out-of-set ID, row 7 (extremes)
    ids = torch.tensor([[N - 1], [N], [V - 1], [3000], [7]])
    ref = F.embedding(ids, w)
    got = pm.lookup(w, ids, ids.clone())
    assert bits_equal(got, ref)
    assert ep.stats["host_calls"] == before["host_calls"] + 1
    assert ep.stats["host_rows"] == before["host_rows"] + ids.numel()
    assert ep.stats["device_calls"] == before["device_calls"]


def test_out_of_set_never_touches_mirror(ep):
    """The host fallback must not read the mirror: poison it and check the result is still exact."""
    w = table(torch.half)
    pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
    # rebind (on CPU, w[:N].to("cpu") aliases the table, so an in-place fill would poison the reference too)
    pm.mirror = torch.full_like(pm.mirror, float("nan"))
    ids = torch.tensor([[1], [N + 5]])
    assert bits_equal(pm.lookup(w, ids, ids.clone()), F.embedding(ids, w))


def test_host_membership(ep):
    w = table(torch.half)
    pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
    assert pm.host_ids_in_set(torch.tensor([[0], [N - 1]]))
    assert not pm.host_ids_in_set(torch.tensor([[0], [N]]))
    assert not pm.host_ids_in_set(torch.tensor([[-1]]))
    assert pm.host_ids_in_set(torch.empty((0, 1), dtype = torch.long))


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_general_keep_set_remap_path(ep, dtype):
    """Non-prefix keep set (e.g. a hot-vocab head): device lookup goes through the remap table."""
    w = table(dtype)
    g = torch.Generator().manual_seed(7)
    keep = torch.randperm(V, generator = g)[:N]
    pm = ep.PrunedEmbeddingMirror(w, keep, "cpu")
    assert not pm.prefix and pm.remap is not None
    assert bits_equal(pm.mirror, w[keep])
    ids = keep[torch.tensor([0, 5, N - 1])].view(3, 1)
    assert bits_equal(pm.lookup(w, ids, None), F.embedding(ids, w))
    absent = torch.tensor([i for i in range(V) if i not in set(keep.tolist())][:2]).view(2, 1)
    mixed = torch.cat([ids[:1], absent])
    assert not pm.host_ids_in_set(mixed)
    assert bits_equal(pm.lookup(w, mixed, mixed.clone()), F.embedding(mixed, w))


def test_exhaustive_every_id(ep):
    """Every vocab ID, in-set via the mirror and out-of-set via the host path, equals the full-table row."""
    w = table(torch.half)
    pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
    ids = torch.arange(V).view(-1, 1)
    inset = ids[:N]
    assert bits_equal(pm.lookup(w, inset, None), F.embedding(inset, w))
    for chunk in ids.split(16):                     # batches of up to 16 rows (MAX_BSZN)
        assert bits_equal(pm.lookup(w, chunk, chunk.clone()), F.embedding(chunk, w))


def test_matches_keying(ep):
    w = table(torch.half)
    keep = torch.arange(N)
    pm = ep.PrunedEmbeddingMirror(w, keep, "cpu")
    assert pm.matches(w, keep, "cpu")
    assert not pm.matches(w, torch.arange(N), "cpu")        # different keep object -> rebuild
    assert not pm.matches(w.float(), keep, "cpu")           # dtype change -> rebuild


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
def test_staging_reuse_same_shape_distinct_ids(ep, dtype):
    """Two out-of-set lookups of the same shape back to back (the GPU path reuses one pinned buffer for them):
    each result must hold its own rows. On CPU there is no staging; the GPU script checks the async case."""
    w = table(dtype)
    pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
    a = torch.tensor([[N + 3], [5]])
    b = torch.tensor([[V - 1], [N]])
    ra = pm.lookup(w, a, a.clone())
    rb = pm.lookup(w, b, b.clone())
    assert bits_equal(ra, F.embedding(a, w)) and bits_equal(rb, F.embedding(b, w))


@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("grad_mode", ["grad", "no_grad", "inference"])
def test_lookup_with_requires_grad_parameter(ep, dtype, grad_mode):
    """
    R522 rerun regression: the served table is nn.Parameter(weight) with requires_grad=True (embedding.py:62).
    The staging gather uses index_select(out=buf), which raises for grad-requiring inputs outside
    no_grad/inference_mode. stage_rows (the exact op the CUDA path runs into the pinned buffer) and lookup must work
    in every grad mode and return rows that do not require grad, bit-identical to the table.
    """
    import contextlib
    from torch import nn
    w = nn.Parameter(table(dtype))
    assert w.requires_grad
    ctx = {"grad": contextlib.nullcontext(), "no_grad": torch.no_grad(), "inference": torch.inference_mode()}[grad_mode]
    ref = F.embedding(torch.tensor([[N + 3], [5]]), w.detach())
    with ctx:
        pm = ep.PrunedEmbeddingMirror(w, torch.arange(N), "cpu")
        flat = torch.tensor([N + 3, 5])
        buf = torch.empty((2, D), dtype = dtype)
        ep.stage_rows(w, flat, buf)
        oos = pm.lookup(w, flat.view(2, 1), flat.view(2, 1).clone())
        ins = pm.lookup(w, torch.tensor([[1], [2]]), None)
    assert bits_equal(buf.view(2, 1, D), ref)
    assert bits_equal(oos, ref) and not oos.requires_grad
    assert bits_equal(ins, F.embedding(torch.tensor([[1], [2]]), w.detach())) and not ins.requires_grad
