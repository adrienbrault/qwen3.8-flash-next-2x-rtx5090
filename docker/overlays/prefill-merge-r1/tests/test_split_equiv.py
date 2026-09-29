#!/usr/bin/env python3
"""prefill-merge r1: the split inside a merged forward and the asynchronous stash, on CPU.

What these tests prove: the PLUMBING -- slicing, which served function each part goes through (and with which layout),
that the slot is copied after part 1 and before part 2, that the stash dict has GDNState.stash()'s layout and bytes,
that readers wait, that the NVMe tier's serializer sees the same bytes, and the Job-side guards. The kernels are CPU
references (tests/_harness.py: fp32 delta rule with the fla signature, the served torch conv reference, a fake PLE
stream pass). They do NOT prove that the Triton / CUDA kernels, or the upstream layers, give per-row results that do
not depend on the forward's row count; that is tests/gpu_stash_equiv.py on flan.
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import exllamav3.modules.gated_delta_net as gdn  # noqa: E402
import exllamav3.modules.ple as ple  # noqa: E402
import exllamav3.cache.prefill_merge as pm  # noqa: E402
import exllamav3.generator.disk_cache as dc  # noqa: E402
from exllamav3.cache.recurrent import RecurrentCache  # noqa: E402
from exllamav3.modules.gated_delta_net_fn.conv1d import causal_conv1d_update_function_torch  # noqa: E402
from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdr  # noqa: E402

NK, HV, DK, DV, K = 2, 4, 8, 8, 4
FD = 2 * NK * DK + HV * DV
CALLS = []


def ref_conv(mixed_qkv, conv_state, recurrent_slots, conv1d_weight, conv1d_bias = None, history = False,
             params = None, token_major = False):
    """Slotted CPU stand-in for causal_conv1d_update: the served torch reference on the slot (state = last K inputs
    in bf16, the weight's dtype), bf16 (bsz, seq, dim) output like the served kernels."""
    assert not history
    CALLS.append(("conv", tuple(mixed_qkv.shape), token_major, mixed_qkv.dtype))
    x = mixed_qkv.transpose(1, 2) if token_major else mixed_qkv
    s = int(recurrent_slots[0])
    y = causal_conv1d_update_function_torch(x, conv_state[s:s + 1], conv1d_weight, conv1d_bias)
    return y.transpose(1, 2).to(torch.bfloat16).contiguous()


gdn.causal_conv1d_update = ref_conv
_fla_calls = []


def _counted_chunk(*a, **k):
    _fla_calls.append(a[0].shape[1])
    return _harness.ref_chunk_gated_delta_rule(*a, **k)


sys.modules["exllamav3.vendor.fla"].chunk_gated_delta_rule = _counted_chunk


class HostPool:
    """Injected staging allocator: plain host memory (pin_memory needs CUDA)."""

    @staticmethod
    def alloc(n):
        return torch.empty(n, dtype = torch.uint8)


def gdn_module(state_dtype):
    m = gdn.GatedDeltaNet.__new__(gdn.GatedDeltaNet)
    m.fdim_qkv, m.conv_kernel_size, m.num_v_heads, m.k_head_dim, m.v_head_dim = FD, K, HV, DK, DV
    m.num_k_heads, m.k_dim, m.v_dim, m.kda = NK, NK * DK, HV * DV, False
    m.recurrent_state_dtype = state_dtype
    g = torch.Generator().manual_seed(7)
    m.conv1d_weight_flat = (torch.randn(FD, K, generator = g) * 0.5).to(torch.bfloat16)
    m.conv1d_bias = (torch.randn(FD, generator = g) * 0.1).to(torch.bfloat16)
    return m


def gdn_state(m, seed = 1):
    l = gdn.GDNLayerState(m, max_batch_size = 3, max_history = 2, cache_id = 0)
    l.alloc("cpu")
    g = torch.Generator().manual_seed(seed)
    l.recurrent_state.copy_(torch.randn(l.recurrent_state.shape, generator = g) * 0.3)
    l.conv_state.copy_(torch.randn(l.conv_state.shape, generator = g))
    return l


def gdn_inputs(T, seed = 3):
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(1, T, FD, generator = g).half()                    # token-major fp16 projection output
    beta = torch.rand(1, T, HV, generator = g).to(torch.bfloat16)
    gg = -torch.rand(1, T, HV, generator = g) * 0.2
    return qkv, beta, gg


def clone_state(l):
    c = gdn.GDNLayerState(l.module, l.max_batch_size, l.max_history, 0)
    c.alloc("cpu")
    c.recurrent_state.copy_(l.recurrent_state)
    c.conv_state.copy_(l.conv_state)
    return c


def served_forward_gdn(m, l, slot, qkv_rows, beta, gg):
    """What GatedDeltaNet.forward's torch path does for one separate forward of these rows (split projections,
    EXL3_GDN_CONV_TOKEN_MAJOR default on): token-major view when bsz * seqlen > 32, else channel-major bf16."""
    T = qkv_rows.shape[1]
    params = {"recurrent_slots": torch.tensor([slot])}
    if T > 32:
        x, tm = qkv_rows, True
    else:
        x, tm = qkv_rows.transpose(1, 2).to(torch.bfloat16).contiguous(), False
    x = gdn.causal_conv1d_update(mixed_qkv = x, conv_state = l.conv_state, recurrent_slots = params["recurrent_slots"],
                                 conv1d_weight = m.conv1d_weight_flat, conv1d_bias = m.conv1d_bias, history = False,
                                 params = params, token_major = tm)
    return gdn.gated_delta_rule_fn(x, beta, gg, l.recurrent_state, params["recurrent_slots"], False, True,
                                   NK, HV, NK * DK, HV * DV, DK, DV, params = params, channelwise_g = False)


def fake_cache(layers):
    c = types.SimpleNamespace()
    c.model = types.SimpleNamespace(loaded_tp = False)
    c.get_all_recurrent_layers = lambda: layers
    return c


def new_split(layers, slot, at, lpb):
    c = fake_cache(layers)
    lay = pm.stash_layout(c)
    assert lay is not None
    pool = pm._pool(c, lay, HostPool.alloc)
    return pm.Split(at, lpb, pm.Capture(lay, slot, pool.try_acquire(), pool)), c


def assert_same(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    assert torch.equal(a.view(torch.uint8) if a.dtype != torch.bool else a, b.view(torch.uint8) if b.dtype != torch.bool else b), what


def run_gdn_case(state_dtype, T, at, slot = 1):
    m = gdn_module(state_dtype)
    l0 = gdn_state(m)
    qkv, beta, gg = gdn_inputs(T)

    # reference: two separate forwards, stash taken between them the served way
    lr = clone_state(l0)
    o1 = served_forward_gdn(m, lr, slot, qkv[:, :at], beta[:, :at], gg[:, :at])
    stash_ref = tuple(t.clone() for t in lr.stash(slot))   # on CPU .cpu() is the live view: snapshot it
    o2 = served_forward_gdn(m, lr, slot, qkv[:, at:], beta[:, at:], gg[:, at:])
    out_ref = torch.cat((o1, o2), dim = 1)

    # the merged forward's split
    ls = clone_state(l0)
    split, _ = new_split({(5, 0): ls}, slot, at, 4096)
    params = {"recurrent_slots": torch.tensor([slot])}
    CALLS.clear()
    _fla_calls.clear()
    _harness.EXT.calls.clear()
    out = gdn.GatedDeltaNet._pm_split_conv_rule(m, split, ls, qkv, True, beta, gg, ls.conv_state, ls.recurrent_state,
                                                params["recurrent_slots"], params)
    assert split.cap.complete()
    st = split.cap.finish(4096, 123, asynchronous = False)

    assert_same(out, out_ref, "output")
    assert_same(ls.recurrent_state, lr.recurrent_state, "final recurrent slot")
    assert_same(ls.conv_state, lr.conv_state, "final conv slot")
    s_ref, c_ref = stash_ref
    s_got, c_got = st[(5, 0)]
    assert_same(s_got, s_ref, "stashed recurrent plane")
    assert_same(c_got, c_ref, "stashed conv window")
    assert s_got._base is None and c_got._base is None and s_got.is_contiguous() and c_got.is_contiguous()
    assert list(st.keys()) == ["position", "checkpoint_size", (5, 0)] and st["position"] == 4096
    # the tail used the served dispatch for its row count
    r = T - at
    tail_conv = CALLS[1]
    assert tail_conv[2] == (r > 32) and (tail_conv[3] == (torch.float16 if r > 32 else torch.bfloat16)), CALLS
    if r >= HV:
        assert _fla_calls == [at, r], _fla_calls
    else:
        assert _fla_calls == [at] and "cuda_recurrent_gated_delta_rule" in _harness.EXT.calls
    return m, l0, qkv, beta, gg, out


def test_gdn_split_bf16_state_chunk_tail():
    run_gdn_case(torch.bfloat16, T = 80, at = 40)          # tail 40 rows: token-major, chunk kernel


def test_gdn_split_bf16_state_short_tail():
    run_gdn_case(torch.bfloat16, T = 56, at = 40)          # tail 16 rows: channel-major bf16 conv, chunk kernel


def test_gdn_split_bf16_state_recurrent_tail():
    run_gdn_case(torch.bfloat16, T = 42, at = 40)          # tail 2 rows < num_v_heads: fused recurrent kernel


def test_gdn_split_fp32_state():
    run_gdn_case(torch.float, T = 80, at = 48)


def test_gdn_single_pass_differs():
    """Sensitivity: an unsplit single pass over the same rows is NOT what the split produces (the served leftover
    forward restarts from the bf16-rounded state and the bf16 conv window; a single pass carries both unrounded).
    If this ever held, the equivalence tests above would not be able to see a missing split."""
    m, l0, qkv, beta, gg, out = run_gdn_case(torch.bfloat16, T = 80, at = 40)
    lu = clone_state(l0)
    out_single = served_forward_gdn(m, lu, 1, qkv, beta, gg)
    assert not torch.equal(out_single, out)


# ------------------------------------------------------------------------------------------------ PLE

CTX, WIN, DIL, KP, HC, HD = 2, 6, 3, 3, 2, 4          # win = (kernel - 1) * dilation


class FakeNgram:
    context_len = CTX
    eos_token_id = 0

    def forward(self, history, params):
        # per row: a function of the row's (ctx + 1)-token window only
        h = history.float()
        w = torch.stack([h[:, i:i + h.shape[1] - CTX] for i in range(CTX + 1)], dim = -1)
        return torch.sin(w @ torch.tensor([0.3, -0.7, 1.1]))[..., None].repeat(1, 1, 5)


def fake_forward_streams(self, streams, token_history, params, conv_state = None, emb = None, fast = True):
    if emb is None:
        emb = self.ple_embedding.forward(token_history, params)
    bsz, seq, H, D = streams.shape
    vals = torch.tanh(streams.reshape(bsz, seq, H * D) * 0.5 + emb[..., :1]).half()
    if conv_state is None:
        conv_state = vals.new_zeros((bsz, H * D, WIN))
    xt = torch.cat((conv_state.to(vals.dtype), vals.transpose(1, 2)), dim = -1)
    w = torch.linspace(-1, 1, H * D * KP).view(H * D, 1, KP)
    y = F.conv1d(xt.float(), w, groups = H * D, dilation = DIL).transpose(1, 2).reshape(bsz, seq, H, D)
    delta = y + emb[..., :1, None]
    if fast:
        streams.add_(delta)
        return None, xt
    return delta, xt


def ple_module(fast):
    m = ple.PLELayer.__new__(ple.PLELayer)
    m.ple_embedding = FakeNgram()
    m.conv_state_len = WIN
    m.hc_mult, m.hidden_size = HC, HD
    m.forward_streams = types.MethodType(lambda self, *a, **k: fake_forward_streams(self, *a, fast = fast, **k), m)
    return m


def ple_state(m, seed = 5):
    l = ple.PLELayerState(m, max_batch_size = 3, max_history = 2, cache_id = 0)
    l.alloc("cpu")
    g = torch.Generator().manual_seed(seed)
    l.conv_state.copy_(torch.randn(l.conv_state.shape, generator = g).half())
    l.id_state.copy_(torch.randint(1, 50, l.id_state.shape, generator = g))
    return l


def clone_ple(l):
    c = ple.PLELayerState(l.module, l.max_batch_size, l.max_history, 0)
    c.alloc("cpu")
    c.conv_state.copy_(l.conv_state)
    c.id_state.copy_(l.id_state)
    return c


def served_forward_ple(m, l, s, x, ids):
    """PLELayer.forward's non-history branch for one separate forward of these rows (bsz 1)."""
    history = torch.cat((l.id_state[s, :CTX][None], ids), dim = 1)     # _state_history
    cs = torch.stack([l.conv_state[s, :, :WIN]])
    delta, stream = m.forward_streams(x, history, {}, conv_state = cs)
    l.conv_state[s, :, :WIN].copy_(stream[0, :, -WIN:])
    l.id_state[s, :CTX].copy_(history[0, -CTX:])
    return x if delta is None else x + delta


def run_ple_case(fast, seq = 23, at = 16, s = 2):
    m = ple_module(fast)
    l0 = ple_state(m)
    g = torch.Generator().manual_seed(9)
    x0 = torch.randn(1, seq, HC, HD, generator = g)
    ids = torch.randint(1, 50, (1, seq), generator = g)

    lr = clone_ple(l0)
    xa = x0.clone()
    ya = served_forward_ple(m, lr, s, xa[:, :at], ids[:, :at])
    stash_ref = tuple(t.clone() for t in lr.stash(s))      # on CPU .cpu() is the live view: snapshot it
    yb = served_forward_ple(m, lr, s, xa[:, at:], ids[:, at:])
    ref = xa if fast else torch.cat((ya, yb), dim = 1)

    ls = clone_ple(l0)
    split, _ = new_split({(-3, 0): ls}, s, at, 512)
    history = torch.cat((ls.id_state[s, :CTX][None], ids), dim = 1)
    xs = x0.clone()
    delta = ple.PLELayer._pm_split_streams(m, split, ls, s, xs, history, {})
    got = xs if delta is None else xs + delta
    st = split.cap.finish(512, 7, asynchronous = False)

    assert (delta is None) == fast
    assert torch.equal(got, ref), "PLE output"
    assert torch.equal(ls.conv_state, lr.conv_state) and torch.equal(ls.id_state, lr.id_state), "PLE final slot"
    c_got, id_got = st[(-3, 0)]
    assert_same(c_got, stash_ref[0], "PLE stashed conv")
    assert torch.equal(id_got, stash_ref[1]) and id_got.dtype == torch.long
    # the id context in the stash is a snapshot, not a view of the live slot
    ls.id_state[s, :CTX].fill_(99)
    assert not torch.equal(id_got, ls.id_state[s, :CTX])


def test_ple_split_fast_path():
    run_ple_case(fast = True)


def test_ple_split_reference_path():
    run_ple_case(fast = False)


# ------------------------------------------------------------------------------------------------ (d) async stash

def two_layer_cache(slot = 1):
    mg = gdn_module(torch.bfloat16)
    lg = gdn_state(mg, seed = 11)
    mp = ple_module(True)
    lp = ple_state(mp, seed = 12)
    layers = {(5, 0): lg, (-3, 0): lp}
    c = fake_cache(layers)
    state = types.SimpleNamespace(cache = c, slot = slot, position = 2816, checkpoint_size = 4242)
    return c, layers, state


def sync_stash(layers, state):
    d = {"position": state.position, "checkpoint_size": state.checkpoint_size}
    for k, l in layers.items():
        d[k] = tuple(t.clone() for t in l.stash(state.slot))
    return d


def payload(d):
    return b"".join(bytes(memoryview(p).cast("B")) if not isinstance(p, (bytes, bytearray)) else bytes(p)
                    for p in dc.serialize_stash(d))


def test_async_stash_bytes_and_readers():
    os.environ["EXL3_STASH_ASYNC"] = "1"
    try:
        c, layers, state = two_layer_cache()
        pm._pool(c, pm.stash_layout(c), HostPool.alloc)
        st = pm.async_stash(state)
        assert isinstance(st, pm.PendingStash)
        pm.wait_stash(st)
        ref = sync_stash(layers, state)
        assert list(st.keys()) == list(ref.keys())
        for k in layers:
            for a, b in zip(st[k], ref[k]):
                assert_same(a, b, f"async {k}")
                assert a._base is None and a.is_contiguous()
        assert dc.own_stash_tensors(st) == 0            # nothing for the NVMe tier to copy
        assert payload(st) == payload(ref)              # the tier writes the same bytes
    finally:
        del os.environ["EXL3_STASH_ASYNC"]


def test_async_stash_under_inference_mode():
    """flan R803 step 0: the generator runs under torch.inference_mode(), which is thread-local. Capture.finish()
    allocates the host destinations on the generator thread (inference tensors); the worker thread then filled them
    OUTSIDE inference mode -> "Inplace update to inference tensor outside InferenceMode is not allowed". The worker
    must run the fill in the finishing thread's mode, and the stash tensors must have the served sync stash's
    inference-tensor status (the served .cpu() under inference mode returns inference tensors)."""
    os.environ["EXL3_STASH_ASYNC"] = "1"
    try:
        for mode in (True, False):
            with torch.inference_mode(mode):
                c, layers, state = two_layer_cache()
                pm._pool(c, pm.stash_layout(c), HostPool.alloc)
                st = pm.async_stash(state)
                assert isinstance(st, pm.PendingStash)
                ref = {k: tuple(t.clone() for t in l.stash(state.slot)) for k, l in layers.items()}   # served .cpu() copy
            pm.wait_stash(st)                            # raised the worker's error before the fix
            for k in layers:
                for a, b in zip(st[k], ref[k]):
                    assert_same(a, b, f"async {k} (inference_mode {mode})")
                    assert a.is_inference() == b.is_inference() == mode, (k, mode, a.is_inference(), b.is_inference())
            # the merged-forward snapshot finishes on the worker too under EXL3_STASH_ASYNC
            with torch.inference_mode(mode):
                split, _ = new_split({(5, 0): layers[(5, 0)]}, state.slot, 16, 512)
                split.capture(layers[(5, 0)])
                st2 = split.cap.finish(512, 1, asynchronous = True)
            pm.wait_stash(st2)
            assert all(t.is_inference() == mode for t in st2[(5, 0)])
        # a slab first allocated under inference mode is reused outside it (and the reverse): never an inference tensor
        c, layers, state = two_layer_cache()
        with torch.inference_mode():
            pm._pool(c, pm.stash_layout(c), HostPool.alloc)
            pm.wait_stash(pm.async_stash(state))
        pm.drain_worker()
        pool = c._pm_pool
        slab = pool.try_acquire()
        assert slab is not None and not slab.is_inference()
        pool.release(slab)
        pm.wait_stash(pm.async_stash(state))            # outside inference mode, same slab
    finally:
        del os.environ["EXL3_STASH_ASYNC"]


def test_snapshot_async_counter():
    # flan R803 step 0: a MERGE_ASYNC turn whose only stash is the snapshot takes no GDNState.stash(); the gate needs
    # the snapshot's own async finish counted (async_snapshots), and only with the knob on
    _, layers, state = two_layer_cache()
    for knob in (False, True):
        if knob:
            os.environ["EXL3_STASH_ASYNC"] = "1"
        try:
            split, _ = new_split(layers, state.slot, 16, 512)
            for l in layers.values():
                split.capture(l)
            n0 = pm.metrics["async_snapshots"]
            st = pm._SnapshotState(split, 5).stash()
            pm.wait_stash(st)
            assert isinstance(st, pm.PendingStash) and st["position"] == 512
            assert pm.metrics["async_snapshots"] == n0 + (1 if knob else 0), (knob, pm.metrics)
        finally:
            os.environ.pop("EXL3_STASH_ASYNC", None)


def test_async_stash_off_and_exhausted():
    c, layers, state = two_layer_cache()
    assert pm.async_stash(state) is None                # knob off: served path
    os.environ["EXL3_STASH_ASYNC"] = "1"
    try:
        c2, _, state2 = two_layer_cache()
        pool = pm.StagingPool(pm.stash_layout(c2)[1], 1, HostPool.alloc)
        c2._pm_pool = pool
        held = pool.try_acquire()
        n0 = pm.metrics["async_no_slab"]
        assert pm.async_stash(state2) is None and pm.metrics["async_no_slab"] == n0 + 1
        pool.release(held)
        st = pm.async_stash(state2)
        assert st is not None
        pm.drain_worker()
        assert pool.try_acquire() is not None          # the worker returned the slab
    finally:
        del os.environ["EXL3_STASH_ASYNC"]


def test_wait_stash_blocks_and_raises():
    st = pm.PendingStash(position = 1)
    t0 = time.monotonic()
    threading.Timer(0.2, st._pm_ready.set).start()
    pm.wait_stash(st)
    assert time.monotonic() - t0 >= 0.19
    pm.wait_stash({"position": 1})                      # plain dicts: no-op
    bad = pm.PendingStash()
    bad._pm_error = "boom"
    bad._pm_ready.set()
    try:
        pm.wait_stash(bad)
        raise AssertionError("no raise")
    except RuntimeError:
        pass


def test_gdnstate_hooks():
    """GDNState.stash takes the async path under the knob; GDNState.unstash waits for a pending stash."""
    os.environ["EXL3_STASH_ASYNC"] = "1"
    try:
        c, layers, _ = two_layer_cache()
        pm._pool(c, pm.stash_layout(c), HostPool.alloc)
        gs = gdn.GDNState.__new__(gdn.GDNState)
        gs.cache, gs.slot, gs.position, gs.checkpoint_size = c, 1, 2816, 4242
        st = gs.stash()
        assert isinstance(st, pm.PendingStash)
        pm.drain_worker()
        # restore into slot 2 through unstash, with the ready flag held back for 0.2 s
        st2 = pm.PendingStash(st)
        gs2 = gdn.GDNState.__new__(gdn.GDNState)
        gs2.cache, gs2.slot, gs2.position = c, 2, 2816
        threading.Timer(0.2, st2._pm_ready.set).start()
        t0 = time.monotonic()
        gs2.unstash(st2)
        assert time.monotonic() - t0 >= 0.19
        lg = layers[(5, 0)]
        assert torch.equal(lg.recurrent_state[2, :1], lg.recurrent_state[1, :1])
    finally:
        del os.environ["EXL3_STASH_ASYNC"]


# ------------------------------------------------------------------------------------------------ Job side

class Page:
    def __init__(self, i, kv):
        self.phash = bytes([i]) * 16
        self.kv_position = kv


def fake_job(layers, prefill_start = 0, n_pages = 4):
    c = fake_cache(layers)
    rc = RecurrentCache(types.SimpleNamespace(loaded_tp = False))
    seq = types.SimpleNamespace(allocated_pages = [Page(i, 256) for i in range(n_pages)])
    rs = types.SimpleNamespace(cache = c, slot = 1, position = prefill_start, checkpoint_size = 99, exported = False)
    gen = types.SimpleNamespace(recurrent_cache = rc, model = types.SimpleNamespace(loaded_tp = False))
    return types.SimpleNamespace(generator = gen, sequences = [seq], recurrent_state = rs, embeddings = None,
                                 last_recurrent_checkpoint_pos = None), c


def test_try_split_and_stash_split():
    _, layers, _ = two_layer_cache()
    job, c = fake_job(layers)
    pm._pool(c, pm.stash_layout(c), HostPool.alloc)
    assert pm.try_split(job, 0, 700, 700) is None                  # knob off
    os.environ["EXL3_PREFILL_MERGE"] = "1"
    try:
        assert pm.try_split(job, 0, 512, 700) is None              # the forward does not reach the prompt end
        sp = pm.try_split(job, 0, 700, 700)
        assert sp is not None and sp.at == 512 and sp.last_page_b == 512
        for l in layers.values():                                  # what the layers do inside the forward
            sp.capture(l)
        pm.stash_split(job, sp)
        key = job.sequences[0].allocated_pages[1].phash
        assert key in job.generator.recurrent_cache and job.last_recurrent_checkpoint_pos == 512
        got = job.generator.recurrent_cache[key]
        assert got["position"] == 512 and got["checkpoint_size"] == 99
        # same position again: skipped (maybe_stash_recurrent's guard), slab returned
        sp2 = pm.try_split(job, 0, 700, 700)
        for l in layers.values():
            sp2.capture(l)
        n = pm.metrics["snapshot_skipped"]
        pm.stash_split(job, sp2)
        assert pm.metrics["snapshot_skipped"] == n + 1 and sp2.cap.slab is None
        # a job whose state is not at the forward start (replay) never merges
        job.recurrent_state.position = 256
        assert pm.try_split(job, 0, 700, 700) is None
        job.recurrent_state.position = 0
        # incomplete capture: no stash, merge disables itself for the process
        job.last_recurrent_checkpoint_pos = None
        job.generator.recurrent_cache.clear()
        sp3 = pm.try_split(job, 0, 700, 700)
        sp3.capture(layers[(5, 0)])
        pm.stash_split(job, sp3)
        assert len(job.generator.recurrent_cache) == 0 and not pm.merge_enabled()
    finally:
        pm._disabled_reason = None
        del os.environ["EXL3_PREFILL_MERGE"]


if __name__ == "__main__":
    _harness.run_tests(globals())
