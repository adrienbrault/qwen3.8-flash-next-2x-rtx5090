"""rebase167 GDN recurrent-state carry (modules/gated_delta_net.py), CPU only, inside the built image (torch +
the integrated package; the extension is replaced by a recorder, nothing launches).

Covers: EXL3_GDN_STATE_BF16 flag parsing; bf16 pinning over the cache's recurrent_dtype (upstream default fp16);
unforced layers follow the cache's dtype (upstream behaviour); the two-row pool layout read by stash_views /
stash (row 2 * slot + parity), rank 5, stash() == stash_views(); the checkpoint dtype guard on unstash;
scan_geometry and the batched_scan_replay call carrying state_half / state_bf16; the prefill capture capability
(prefill_capture_supported, fail-closed _pm_capture_param); and, against the integrated cache/prefill_merge.py, a
real Capture of every slot / parity through stash_layout + Capture.capture + finish, restored by unstash.
"""
import importlib
import types

import pytest

torch = pytest.importorskip("torch")
gdn = pytest.importorskip("exllamav3.modules.gated_delta_net")


class _Rec:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, a, k))
            return ("job", name, a)
        return f


def _module(dtype = None, kda = False, mamba2 = False):
    m = types.SimpleNamespace(
        fdim_qkv = 64, conv_kernel_size = 4, num_v_heads = 2, num_k_heads = 1,
        k_head_dim = 32, v_head_dim = 32, kda = kda, mamba2 = mamba2,
    )
    if dtype is not None:
        m.recurrent_state_dtype = dtype
    return m


@pytest.mark.parametrize("value, expect", [("1", True), ("0", False), ("", False), ("true", False)])
def test_flag_parsing(monkeypatch, value, expect):
    monkeypatch.setenv("EXL3_GDN_STATE_BF16", value)
    mod = importlib.reload(gdn)
    try:
        assert mod._gdn_state_bf16 is expect
    finally:
        monkeypatch.delenv("EXL3_GDN_STATE_BF16")
        importlib.reload(gdn)


def test_bf16_pin_overrides_cache_dtype():
    l = gdn.GDNLayerState(_module(torch.bfloat16), 4, 3, 0, state_dtype = torch.float)
    assert l.recurrent_state.dtype == torch.bfloat16
    l.set_state_dtype(torch.half)          # Cache(recurrent_dtype = torch.half), upstream default
    assert l.recurrent_state.dtype == torch.bfloat16
    assert tuple(l.recurrent_state.shape) == (8, 1, 2, 32, 32)   # two rows per slot (3a8e2fb8)


@pytest.mark.parametrize("module_dtype", [None, torch.float])
def test_unforced_layer_follows_cache(module_dtype):
    l = gdn.GDNLayerState(_module(module_dtype), 4, 3, 0)
    assert l.recurrent_state.dtype == torch.float
    l.set_state_dtype(torch.half)
    assert l.recurrent_state.dtype == torch.half


def test_checkpoint_size_follows_dtype():
    a = gdn.GDNLayerState(_module(torch.bfloat16), 2, 3, 0)
    b = gdn.GDNLayerState(_module(None), 2, 3, 0)
    b.set_state_dtype(torch.half)
    c = gdn.GDNLayerState(_module(None), 2, 3, 0)
    assert a.get_checkpoint_size() == b.get_checkpoint_size() < c.get_checkpoint_size()


@pytest.mark.parametrize("parity", [0, 1])
def test_stash_views_reads_live_row(monkeypatch, parity):
    monkeypatch.setattr(gdn, "host_copy", lambda t: t.clone())
    l = gdn.GDNLayerState(_module(torch.bfloat16), 3, 3, 0)
    l.alloc("cpu")
    for r in range(l.recurrent_state.shape[0]):
        l.recurrent_state[r].fill_(float(r))
    l.conv_state[1].fill_(7.0)
    rs, cs = l.stash_views(1, parity)
    assert rs.shape == (1, 1, 2, 32, 32) and float(rs.float().mean()) == float(2 * 1 + parity)
    assert cs.shape == (64, 4) and float(cs.float().mean()) == 7.0
    s, c = l.stash(1, 0, parity)
    assert torch.equal(s, rs) and torch.equal(c, cs)
    # restore into the other parity row of slot 2
    l.unstash(2, (s, c), 0, 1 - parity)
    assert float(l.recurrent_state[2 * 2 + 1 - parity].float().mean()) == float(2 + parity)


def test_unstash_dtype_guard():
    l = gdn.GDNLayerState(_module(torch.bfloat16), 2, 3, 0)
    l.alloc("cpu")
    s = torch.zeros((1, 1, 2, 32, 32), dtype = torch.half)
    c = torch.zeros((64, 4), dtype = torch.bfloat16)
    with pytest.raises(TypeError):
        l.unstash(0, (s, c), 0, 0)


@pytest.mark.parametrize("dtype, half, bf16", [(torch.bfloat16, False, True), (torch.half, True, False), (torch.float, False, False)])
def test_replay_carries_state_kind(monkeypatch, dtype, half, bf16):
    rec = _Rec()
    monkeypatch.setattr(gdn, "ext", rec)
    l = gdn.GDNLayerState(_module(torch.bfloat16 if dtype == torch.bfloat16 else None), 2, 3, 0)
    if dtype != torch.bfloat16:
        l.set_state_dtype(dtype)
    l.alloc("cpu")
    assert l.scan_geometry() == (0, 1, 2, 32, 32, half, bf16)
    l.staged_views(1, 4)      # a 4-token speculative pass (MTP depth 3) recorded its scan inputs
    gdn._rewind_layers([l], 0, 3, 2, 0, (1, 4, 1), 0, 1)
    replay = [c for c in rec.calls if c[0] == "batched_scan_replay"]
    assert len(replay) == 1
    args = replay[0][1]
    assert args[2:] == (0, 1, 2, 32, 32, half, bf16)
    jobs = [c for c in rec.calls if c[0] == "ScanReplayJob"]
    assert len(jobs) == 1 and jobs[0][1][6] == 2          # prefix = last_history + 1 - num_tokens


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half, torch.float])
@pytest.mark.parametrize("parity", [0, 1])
def test_stash_copies_exactly_the_views(monkeypatch, dtype, parity):
    monkeypatch.setattr(gdn, "host_copy", lambda t: t.clone())
    l = gdn.GDNLayerState(_module(torch.bfloat16 if dtype == torch.bfloat16 else None), 3, 3, 0)
    if dtype != torch.bfloat16:
        l.set_state_dtype(dtype)
    l.alloc("cpu")
    l.recurrent_state.copy_(torch.randn(l.recurrent_state.shape).to(dtype))
    l.conv_state.copy_(torch.randn(l.conv_state.shape).to(l.conv_state.dtype))
    views = l.stash_views(2, parity)
    stashed = l.stash(2, 0, parity)
    assert tuple(views[0].shape) == (1, 1, 2, 32, 32)      # rank 5: one pool row, upstream's stash shape
    assert len(stashed) == len(views)
    for a, b in zip(stashed, views):
        assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.parametrize("two_row", [False, True, None])
def test_capability_and_fail_closed(monkeypatch, two_row):
    if two_row is None:      # a prefill_merge without the handshake (the served module)
        monkeypatch.delattr(gdn._pm, "TWO_ROW_POOL", raising = False)
    else:
        monkeypatch.setattr(gdn._pm, "TWO_ROW_POOL", two_row, raising = False)
    ok = bool(two_row)
    assert gdn.prefill_capture_supported() is ok
    cap = object()
    assert gdn._pm_capture_param({}, "_prefill_tail_capture") is None
    for key in ("_prefill_split", "_prefill_tail_capture"):
        if ok:
            assert gdn._pm_capture_param({key: cap}, key) is cap
        else:
            with pytest.raises(RuntimeError, match = "prefill_capture_supported"):
                gdn._pm_capture_param({key: cap}, key)


class _Pool:
    def __init__(self):
        self.released = 0

    def release(self, slab, events = None):
        self.released += 1


def _gdn_module_instance(dtype):
    m = object.__new__(gdn.GatedDeltaNet)
    for k, v in vars(_module(dtype)).items():
        object.__setattr__(m, k, v)
    return m


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
def test_integrated_capture_roundtrip(monkeypatch, dtype):
    """The integrated cache/prefill_merge.py (PY): its GDN layout comes from stash_views, its Capture of (slot,
    parity) equals stash(slot, parity) bit for bit and restores through unstash, for every slot and both parities"""
    pm = gdn._pm
    assert getattr(pm, "TWO_ROW_POOL", False) is True, "integrated prefill_merge must declare TWO_ROW_POOL"
    assert getattr(pm, "GDN_CAPTURE_QUERY", None) == "prefill_capture_supported"
    monkeypatch.setattr(pm, "_gdn_capability", None)
    assert pm.gdn_capture_supported() is True
    monkeypatch.setattr(gdn, "host_copy", lambda t: t.clone())
    l = gdn.GDNLayerState(_gdn_module_instance(torch.bfloat16 if dtype == torch.bfloat16 else None), 3, 3, 0)
    if dtype != torch.bfloat16:
        l.set_state_dtype(dtype)
    l.alloc("cpu")
    l.recurrent_state.copy_(torch.randn(l.recurrent_state.shape).to(dtype))
    l.conv_state.copy_(torch.randn(l.conv_state.shape).to(l.conv_state.dtype))
    key = "model.layers.0.linear_attn"
    cache = types.SimpleNamespace(get_all_recurrent_layers = lambda: {key: l})
    lay = pm.stash_layout(cache)
    assert lay is not None, "prefill_merge.stash_layout rejected a GDN layer"
    (k, layer, kind, parts), = lay[0]
    assert layer is l and kind == "gdn"
    assert [(tuple(sh), dt) for _, sh, dt, _ in parts] == [(tuple(v.shape), v.dtype) for v in l.stash_views(0, 0)]
    for slot in range(3):
        for parity in (0, 1):
            pool = _Pool()
            cap = pm.Capture(lay, slot, torch.zeros(lay[1], dtype = torch.uint8), pool, parity = parity)
            cap.capture(l)
            assert cap.complete()
            st = cap.finish(123, 456, asynchronous = False)
            pm.wait_stash(st)
            assert st["position"] == 123 and pool.released == 1
            want = l.stash(slot, 0, parity)
            got = st[k]
            assert len(got) == len(want)
            for a, b in zip(got, want):
                assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
            # restore into the other parity row of the next slot, then compare with the source row
            dst = (slot + 1) % 3
            l.unstash(dst, got, 0, 1 - parity)
            assert torch.equal(l.recurrent_state[2 * dst + 1 - parity], l.recurrent_state[2 * slot + parity])
            assert torch.equal(l.conv_state[dst, :, :4], l.conv_state[slot, :, :4])
            # fresh contents for the next (slot, parity)
            l.recurrent_state.copy_(torch.randn(l.recurrent_state.shape).to(dtype))
            l.conv_state.copy_(torch.randn(l.conv_state.shape).to(l.conv_state.dtype))
