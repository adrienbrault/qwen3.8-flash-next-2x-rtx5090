#!/usr/bin/env python3
"""prefill-merge r1: the forward-level wiring of the split (review R803 S8), on CPU.

test_split_equiv.py calls the split helpers directly; these tests drive GatedDeltaNet.forward and PLELayer.forward,
so they fail when the gate in forward() is broken (wrong bsz / save_history / `at` condition, branch removed) or when
the no-split path stops being the served code:
  - GDN: with params["_prefill_split"] (bsz 1, recurrent states, 0 < at < seqlen) forward() calls _pm_split_conv_rule
    once and neither served kernel entry directly; without it (and for bsz 2, save_history, at 0 / at >= seqlen)
    forward() makes exactly the calls the BASE file's forward() makes (the served file, loaded from base/), same
    function, same keyword names, same tensors.
  - PLE: with the split, forward() runs two parts, captures between them, and leaves the output and slot state of two
    separate served forwards; without it, the patched forward() equals the base forward() on output and slot state.
The kernels are CPU stand-ins; what is under test is the dispatch, not the numerics.
"""
import importlib.util
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: E402

import torch  # noqa: E402

import exllamav3.modules.gated_delta_net as gdn  # noqa: E402
import exllamav3.modules.ple as ple  # noqa: E402
import exllamav3.cache.prefill_merge as pm  # noqa: E402
import test_split_equiv as TS  # noqa: E402  (module builders + the HostPool / fake PLE streams)

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base", "modules")


def load_base(name):
    """The served (unpatched) module file as a sibling module of the patched one (relative imports resolve)."""
    spec = importlib.util.spec_from_file_location(f"exllamav3.modules._pm_base_{name}", os.path.join(BASE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


base_gdn = load_base("gated_delta_net")
base_ple = load_base("ple")

# ------------------------------------------------------------------------------------------------ GDN

REC = []


def rec_conv(**kw):
    REC.append(("conv", kw))
    return torch.zeros(1, 1, 1)


def rec_rule(**kw):
    REC.append(("rule", kw))
    bsz, seqlen = kw["beta"].shape[:2]
    return torch.zeros(bsz, seqlen, TS.HV, TS.DV)


def rec_split(self, *a):
    REC.append(("split", a))
    bsz, seqlen = a[4].shape[:2]        # beta
    return torch.zeros(bsz, seqlen, TS.HV, TS.DV)


for mod in (gdn, base_gdn):
    mod.causal_conv1d_update = rec_conv
    mod.gated_delta_rule_fn = rec_rule
gdn.GatedDeltaNet._pm_split_conv_rule = rec_split


def fused_op_2(b, a, dt_bias, a_log, beta, g, beta_scale):
    """CPU stand-in for ext.gated_delta_net_fused_op_2. forward() allocates beta and g with torch.empty and the ext
    fills them; the harness's generic FakeExt stub returns without writing, which left both tensors holding whatever
    the allocator handed out (torch 2.9 in the image: different bytes per forward, so base vs patched 'beta' differed
    -- a test artifact, R803b build 2026-09-29). Deterministic here, so the calls compare by value."""
    beta.copy_(torch.sigmoid(b.float() * beta_scale).to(beta.dtype))
    g.copy_(-(torch.nn.functional.softplus(a.float() + dt_bias.float()) * torch.exp(a_log.float())))


# forward() reads `ext` from its module; both the patched and the base module share the harness's FakeExt object
assert gdn.ext is _harness.EXT and base_gdn.ext is _harness.EXT
_harness.EXT.gated_delta_net_fused_op_2 = fused_op_2


class Proj:
    def __init__(self, dim, seed):
        self.dim, self.seed = dim, seed

    def forward(self, x, params, out_dtype = None):
        g = torch.Generator().manual_seed(self.seed)
        w = torch.randn(x.shape[-1], self.dim, generator = g)
        return (x.float() @ w).to(out_dtype or torch.half)


class Passthrough:
    def forward(self, x, params, gate = None):
        return x


def gdn_forward_module(cls):
    m = cls.__new__(cls)
    m.fdim_qkv, m.conv_kernel_size, m.num_v_heads, m.k_head_dim, m.v_head_dim = TS.FD, TS.K, TS.HV, TS.DK, TS.DV
    m.num_k_heads, m.k_dim, m.v_dim, m.kda = TS.NK, TS.NK * TS.DK, TS.HV * TS.DV, False
    m.recurrent_state_dtype = torch.bfloat16
    m.conv1d_weight = torch.zeros(TS.FD, 1, TS.K, dtype = torch.bfloat16)
    m.conv1d_weight_flat = torch.zeros(TS.FD, TS.K, dtype = torch.bfloat16)
    m.conv1d_bias = None
    m.out_dtype, m.tp_reduce, m.device, m.layer_idx = None, False, "cpu", 5
    m.bc_split, m.qkvz_proj, m.ba_proj, m.multi_qkvz = False, None, None, None
    m.qkv_proj, m.z_proj = Proj(TS.FD, 1), Proj(TS.HV * TS.DV, 2)
    m.b_proj, m.a_proj = Proj(TS.HV, 3), Proj(TS.HV, 4)
    m.dt_bias, m.a_log, m.beta_scale = torch.zeros(TS.HV), torch.zeros(TS.HV), 1.0
    m.norm, m.o_proj = Passthrough(), Passthrough()
    return m


def gdn_params(m, bsz, split = None, history = False):
    rsl = TS.gdn_state(TS.gdn_module(torch.bfloat16))
    rs = types.SimpleNamespace(exported = False, cache = types.SimpleNamespace(get_recurrent_layer = lambda inst: rsl))
    p = {"recurrent_states": [rs], "recurrent_slots": torch.arange(bsz)}
    if history:
        p["recurrent_history"] = True
    if split is not None:
        p["_prefill_split"] = split
    return p, rsl


_POISON = [0]


def poisoned_empty(*a, **k):
    """torch.empty that never hands out reusable bytes: every allocation is filled with a different value, so a tensor
    forward() reads without having written differs between two forwards on any allocator (the image's torch 2.9 on
    Linux exposed it where the Mac's did not)."""
    t = _real_empty(*a, **k)
    _POISON[0] += 1
    if t.is_floating_point():
        t.fill_(float(_POISON[0]) * 1e3 + 0.5)
    elif t.dtype != torch.bool:
        t.fill_(_POISON[0])
    return t


_real_empty = torch.empty


def run_gdn(cls, bsz, seqlen, split = None, history = False):
    m = gdn_forward_module(cls)
    p, rsl = gdn_params(m, bsz, split, history)
    x = torch.randn(bsz, seqlen, 16, generator = torch.Generator().manual_seed(5)).half()
    REC.clear()
    torch.empty = poisoned_empty
    try:
        m.forward(x, p)
    finally:
        torch.empty = _real_empty
    return list(REC), rsl


def same_calls(a, b):
    assert [n for n, _ in a] == [n for n, _ in b], ([n for n, _ in a], [n for n, _ in b])
    for (n, ka), (_, kb) in zip(a, b):
        assert sorted(ka) == sorted(kb), (n, sorted(ka), sorted(kb))
        for k in ka:
            va, vb = ka[k], kb[k]
            if isinstance(va, torch.Tensor):
                assert isinstance(vb, torch.Tensor) and va.shape == vb.shape and va.dtype == vb.dtype, (n, k)
                if k not in ("conv_state", "recurrent_state", "mixed_qkv") or n == "conv":
                    assert torch.equal(va, vb), (n, k)
            elif k != "params":
                assert va == vb, (n, k, va, vb)


def split_for(at):
    return pm.Split(at, 256, types.SimpleNamespace(capture = lambda rsl: None))


def test_gdn_forward_takes_split():
    calls, rsl = run_gdn(gdn.GatedDeltaNet, 1, 48, split_for(40))
    assert [n for n, _ in calls] == ["split"], calls
    a = calls[0][1]
    assert a[0].at == 40 and a[1] is rsl and a[2].shape == (1, 48, TS.FD) and a[3] is True   # token-major, 48 > 32 rows
    assert a[6] is rsl.conv_state and a[7] is rsl.recurrent_state and a[8].tolist() == [0]


def test_gdn_forward_served_without_split():
    for bsz, seqlen in ((1, 48), (1, 12), (2, 20)):
        got, _ = run_gdn(gdn.GatedDeltaNet, bsz, seqlen)
        ref, _ = run_gdn(base_gdn.GatedDeltaNet, bsz, seqlen)
        assert [n for n, _ in got] == ["conv", "rule"], got
        same_calls(got, ref)


def test_gdn_forward_gate_conditions():
    # bsz 2, save_history, a split at 0 or at/after the end: never split, always the base calls
    for bsz, seqlen, at, hist in ((2, 48, 40, False), (1, 48, 40, True), (1, 48, 0, False), (1, 48, 48, False)):
        got, _ = run_gdn(gdn.GatedDeltaNet, bsz, seqlen, split_for(at), hist)
        ref, _ = run_gdn(base_gdn.GatedDeltaNet, bsz, seqlen, None, hist)
        assert "split" not in [n for n, _ in got], (bsz, at, hist, got)
        same_calls(got, ref)


def test_gdn_override_on_forward():
    # review N1: the decorator belongs to forward (as in the base file), not to the new helper
    import inspect
    src = inspect.getsource(gdn.GatedDeltaNet)
    assert "@override\n    def forward(" in src and "@override\n    def _pm_split_conv_rule(" not in src


# ------------------------------------------------------------------------------------------------ PLE

def ple_forward_module(cls, fast):
    m = cls.__new__(cls)
    m.ple_embedding = TS.FakeNgram()
    m.conv_state_len = TS.WIN
    m.hc_mult, m.hidden_size = TS.HC, TS.HD
    m.stub, m.tp_owner, m.mm_token_id, m.layer_idx = False, None, None, -3
    m.forward_streams = types.MethodType(lambda self, *a, **k: TS.fake_forward_streams(self, *a, fast = fast, **k), m)
    return m


def ple_run(cls, fast, rsl, x, ids, split = None, slot = 2):
    m = ple_forward_module(cls, fast)
    rs = types.SimpleNamespace(exported = False, cache = types.SimpleNamespace(get_recurrent_layer = lambda inst: rsl))
    p = {"recurrent_states": [rs], "recurrent_slots": torch.tensor([slot]), "input_ids": ids}
    if split is not None:
        p["_prefill_split"] = split
    return m.forward(x, p)


def ple_case(fast, seq = 23, at = 16, s = 2):
    m0 = TS.ple_module(fast)
    l0 = TS.ple_state(m0)
    g = torch.Generator().manual_seed(9)
    x0 = torch.randn(1, seq, TS.HC, TS.HD, generator = g)
    ids = torch.randint(1, 50, (1, seq), generator = g)
    return l0, x0, ids


def test_ple_forward_split_equals_two_forwards():
    for fast in (True, False):
        l0, x0, ids = ple_case(fast)
        at = 16
        lr = TS.clone_ple(l0)
        ya = ple_run(base_ple.PLELayer, fast, lr, x0[:, :at].clone(), ids[:, :at])
        ref_stash = tuple(t.clone() for t in lr.stash(2))
        yb = ple_run(base_ple.PLELayer, fast, lr, x0[:, at:].clone(), ids[:, at:])
        ls = TS.clone_ple(l0)
        split, _ = TS.new_split({(-3, 0): ls}, 2, at, 512)
        y = ple_run(ple.PLELayer, fast, ls, x0.clone(), ids, split)
        assert split.cap.complete(), "forward() did not capture at the split"
        assert torch.equal(y, torch.cat((ya, yb), dim = 1)), fast
        assert torch.equal(ls.conv_state, lr.conv_state) and torch.equal(ls.id_state, lr.id_state)
        st = split.cap.finish(512, 7, asynchronous = False)
        assert torch.equal(st[(-3, 0)][0], ref_stash[0]) and torch.equal(st[(-3, 0)][1], ref_stash[1])


def test_ple_forward_served_without_split():
    for fast in (True, False):
        l0, x0, ids = ple_case(fast)
        la, lb = TS.clone_ple(l0), TS.clone_ple(l0)
        ya = ple_run(ple.PLELayer, fast, la, x0.clone(), ids)
        yb = ple_run(base_ple.PLELayer, fast, lb, x0.clone(), ids)
        assert torch.equal(ya, yb) and torch.equal(la.conv_state, lb.conv_state) and torch.equal(la.id_state, lb.id_state)
        # a split at 0 or at the end is ignored: the base result again
        for at in (0, x0.shape[1]):
            lc = TS.clone_ple(l0)
            called = []
            sp = pm.Split(at, 512, types.SimpleNamespace(capture = lambda rsl: called.append(1)))
            yc = ple_run(ple.PLELayer, fast, lc, x0.clone(), ids, sp)
            assert not called and torch.equal(yc, yb) and torch.equal(lc.conv_state, lb.conv_state)


def test_capture_unknown_layer_left_incomplete():
    # review N8: a layer state the cached layout does not know leaves the split incomplete instead of raising inside
    # the forward; stash_split then skips the stash and disables merge (test_split_equiv covers that part)
    l0 = TS.ple_state(TS.ple_module(True))
    split, _ = TS.new_split({(-3, 0): l0}, 2, 16, 512)
    split.capture(TS.clone_ple(l0))
    assert not split.cap.complete() and not split.cap.done
    split.cap.abandon()


if __name__ == "__main__":
    _harness.run_tests(globals())
