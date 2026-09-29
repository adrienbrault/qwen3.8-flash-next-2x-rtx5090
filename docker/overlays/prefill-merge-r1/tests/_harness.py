"""CPU harness for the prefill-merge r1 tests (same idea as rebase-dev r3's tests/_runner.py).

Imports the exllamav3 package WITHOUT the CUDA extension and without Triton: `exllamav3.ext` and `triton` are stubbed
before the first import, and `exllamav3.vendor.fla` is replaced by a module whose chunk kernels are the fp32 torch
reference below. Which package is imported:
  PM_TREE=<dir>  the directory that CONTAINS the exllamav3 package (tests/run_offline.sh builds base + fix.patch there)
  unset          whatever `import exllamav3` finds (the image's patched site-packages in the Docker build)
Nothing is written into the tree (sys.dont_write_bytecode).
"""
import os
import sys
import types

sys.dont_write_bytecode = True

_tree = os.environ.get("PM_TREE")
if _tree:
    sys.path.insert(0, os.path.abspath(_tree))

import torch  # noqa: E402


class FakeExt:
    """Stands in for exllamav3_ext: records calls; the few entry points the tests exercise are CPU references."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self.calls.append(name)
            return None
        return _call

    # the fused recurrent delta rule (gated_delta_rule.py fused branch): state read from the slot (widened to fp32),
    # the sequence processed in fp32, the final state stored once (rounded to the slot dtype), bf16 output
    def cuda_recurrent_gated_delta_rule(self, mixed_qkv, g, beta, recurrent_state, core_attn_out,
                                        num_k_heads, num_v_heads, k_head_dim, v_head_dim, recurrent_slots, history):
        self.calls.append("cuda_recurrent_gated_delta_rule")
        assert not history
        bsz, seqlen, _ = mixed_qkv.shape
        k_dim, v_dim = num_k_heads * k_head_dim, num_v_heads * v_head_dim
        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
        q = q.reshape(bsz, seqlen, num_k_heads, k_head_dim)
        k = k.reshape(bsz, seqlen, num_k_heads, k_head_dim)
        v = v.reshape(bsz, seqlen, num_v_heads, v_head_dim)
        for i, s in enumerate(recurrent_slots.tolist()):
            st = recurrent_state[s, 0].unsqueeze(0).float()
            o, st = ref_chunk_gated_delta_rule(q[i:i + 1], k[i:i + 1], v[i:i + 1], g = g[i:i + 1], beta = beta[i:i + 1],
                                               initial_state = st, output_final_state = True,
                                               use_qk_l2norm_in_kernel = True)
            recurrent_state[s, 0].copy_(st[0])
            core_attn_out[i].copy_(o[0])


def _l2norm(x, eps = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + eps)


def ref_chunk_gated_delta_rule(q, k, v, g = None, beta = None, initial_state = None, output_final_state = False,
                               use_qk_l2norm_in_kernel = False):
    """fp32 sequential delta rule with the fla chunk kernel's call signature (grouped k heads -> v heads). Unlike
    gated_delta_rule.torch_recurrent_gated_delta_rule it keeps the initial state in fp32 (the served chunk kernel does),
    so a test would see a split that carried an unrounded state."""
    b, s, hk, dk = k.shape
    hv, dv = v.shape[2], v.shape[3]
    rep = hv // hk
    q, k, v, g, beta = [t.float() for t in (q, k, v, g, beta)]
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    q = q.repeat_interleave(rep, dim = 2) * dk ** -0.5
    k = k.repeat_interleave(rep, dim = 2)
    st = torch.zeros(b, hv, dk, dv) if initial_state is None else initial_state.float().clone()
    out = torch.empty(b, s, hv, dv)
    for i in range(s):
        st = st * g[:, i].exp()[..., None, None]
        kv = (st * k[:, i][..., None]).sum(dim = -2)
        delta = (v[:, i] - kv) * beta[:, i][..., None]
        st = st + k[:, i][..., None] * delta[..., None, :]
        out[:, i] = (st * q[:, i][..., None]).sum(dim = -2)
    return out.to(torch.bfloat16), (st if output_final_state else None)


EXT = FakeExt()
_ext_pkg = types.ModuleType("exllamav3.ext")
_ext_pkg.exllamav3_ext = EXT
_ext_pkg.is_precompiled_extension_available = lambda: False
_ext_pkg.exllamav3_ext_obj = EXT
sys.modules["exllamav3.ext"] = _ext_pkg


def _triton_stub():
    stub = types.ModuleType("triton")

    class _Fn:
        def __init__(self, fn, **meta):
            self.fn = fn

        def __getitem__(self, grid):
            return lambda *a, **k: None

        def __call__(self, *args, **kwargs):
            return None

    def jit(fn = None, **meta):
        if fn is not None:
            return _Fn(fn)
        return lambda f: _Fn(f)

    stub.jit = jit
    stub.autotune = lambda *a, **k: (lambda f: f)
    stub.heuristics = lambda *a, **k: (lambda f: f)
    stub.Config = lambda *a, **k: None
    stub.cdiv = lambda a, b: -(-a // b)
    stub.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    stub.runtime = types.ModuleType("triton.runtime")
    stub.runtime.driver = types.SimpleNamespace(active = types.SimpleNamespace(
        get_current_target = lambda: types.SimpleNamespace(arch = "cpu", backend = "cpu", warp_size = 32)))
    lang = types.ModuleType("triton.language")
    for n in ("float32", "float16", "bfloat16", "int32", "int64", "int1", "int8", "uint8"):
        setattr(lang, n, n)
    lang.constexpr = int
    core = types.ModuleType("triton.language.core")
    core.constexpr = int
    core.dtype = "f32"
    lang.core = core
    stub.language = lang
    sys.modules["triton"] = stub
    sys.modules["triton.runtime"] = stub.runtime
    sys.modules["triton.language"] = lang
    sys.modules["triton.language.core"] = core


_triton_stub()

_fla = types.ModuleType("exllamav3.vendor.fla")
_fla.chunk_gated_delta_rule = ref_chunk_gated_delta_rule
_fla.chunk_kda = None
sys.modules["exllamav3.vendor.fla"] = _fla

for _k in list(os.environ):
    if _k.startswith("EXL3_"):
        del os.environ[_k]


def run_tests(module_globals):
    """Run every test_* function of a module; exit 1 on any failure."""
    import traceback
    names = sorted(n for n in module_globals if n.startswith("test_") and callable(module_globals[n]))
    failed = 0
    for n in names:
        try:
            module_globals[n]()
            print(f"  ok   {n}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {n}")
            traceback.print_exc()
    print(f"{len(names) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
