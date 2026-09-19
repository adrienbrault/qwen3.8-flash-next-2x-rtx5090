"""CPU dry run of tp_bound_gpu.py against the REAL exllamav3 in src/ (see dryrun_env.py for the stubs).

The probe's own code (Probe.run_gdn / run_attn / run_moe / run_mtp / run_hc / run_head, the half builders in
tpb_common.py, the r521 coop call) runs unmodified on the meta device over a synthetic checkpoint (mini_ckpt.py) whose
tensors have the served per-tensor shapes. Every call into the compiled extension goes through ext_sigs.StubExt, which
rejects names the image does not bind and arities the bindings would reject. Differences from the box, all explicit:

  * device = meta (shapes only; no numbers): sum_check is replaced by a stub, the Gpu class by FakeGpu.
  * BlockSparseMLP.load builds its decode path only when device.type == "cuda" (block_sparse_mlp.py:778); on meta the
    same three calls it would make (cpu_post_load, load_local, load_routing) are made by a wrapper.
  * BC_LinearEXL3.run_alloc returns a tensor shaped as linear.cpp:56-64 allocates it; needs_configure returns True so
    the configure_slot paths are exercised.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DAILY_ENV = {"EXL3_HOST_GAP_REWIND": "1", "EXL3_HC_MIX_V2": "1", "EXL3_HC_MIX_V2_MIN_R": "1",
             "EXL3_LS_PREFILL_PIPELINE": "1", "EXL3_MOE_COOP_V2": "1", "EXL3_SHARED_EXPERT_OVERLAP": "1",
             "EXL3_DRAFT_PINNED_STAGING": "1", "EXL3_BATCH_VERIFY": "1", "EXL3_MTP_HEAD_N": "65536",
             "EXL3_MOE_PREFILL_E3": "1"}


class FakeGpu:
    name = "dryrun"
    l2_bytes = 96 << 20
    clock_khz = 2_400_000

    def __init__(self, required):
        self.required = required
        self.profiled = 0

    def sync(self):
        pass

    def sleep_ms(self, ms):
        pass

    def events(self):
        class E:
            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return 1.0
        return E(), E()

    def profile_kernels(self, fn, n):
        for i in range(n):
            fn(i)
        self.profiled += 1
        return {v: {"us": 1.0, "launches": 1.0} for v in self.required}


def setup():
    for k, v in DAILY_ENV.items():
        os.environ[k] = v
    import dryrun_env
    ext = dryrun_env.install()
    import torch
    ext.returns["BC_LinearEXL3.run_alloc"] = lambda x, n, fp32: torch.empty(
        tuple(x.shape[:-1]) + (n,), device=x.device, dtype=torch.float if fp32 else torch.half)
    for k in ("BC_GatedDeltaNetSplit.needs_configure", "BC_Attention.needs_configure"):
        ext.returns[k] = lambda *a, **kw: True
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    if not getattr(BlockSparseMLP, "_dryrun_wrapped", False):
        orig = BlockSparseMLP.load

        def load(self, device, **kw):
            orig(self, device, **kw)
            if torch.device(device).type == "meta":
                self.cpu_post_load()
                self.load_local(**kw)
                self.load_routing(**kw)
        BlockSparseMLP.load = load
        BlockSparseMLP._dryrun_wrapped = True
    return ext


def run(ckpt_dir: str, extra_args=None, log=lambda m: None, spawn=None):
    """Build the synthetic checkpoint and run the whole probe on meta. Returns (res, ext, recv_log). `spawn` stands in
    for the fresh-process respawn (tp_bound_gpu.spawn_child)."""
    ext = setup()
    import torch
    import mini_ckpt
    import tp_bound_gpu as G
    import tpb_common as C
    mini_ckpt.build(ckpt_dir)
    recv_log = []
    orig_recv = C.StubConsumer.recv

    def recv(self, imp, cuda=False, slice_dim=None, first=None, last=None):
        t = imp.get("tensor")
        recv_log.append({"shape": tuple(t.shape) if t is not None else None,
                         "dtype": str(t.dtype) if t is not None else None,
                         "slice_dim": slice_dim, "first": first, "last": last})
        return orig_recv(self, imp, cuda, slice_dim, first, last)
    C.StubConsumer.recv = recv
    orig_sum = C.sum_check
    C.sum_check = lambda full, parts, **k: {"ok": True, "rel_maxabs": 0.0, "cos": 1.0, "dryrun": True}
    try:
        a = G.parse_args(["--model", ckpt_dir, "--device", "meta", "--gdn-layers", "0,1", "--attn-layers", "3,7",
                          "--moe-candidates", "0,1,2", "--iters", "2", "--rounds", "2", "--warmup", "1",
                          "--profile-calls", "1", "--routing-sets", "2"] + list(extra_args or []))
        xl = G.XL()
        res = G.run_probe(a, torch, xl, FakeGpu(G.REQUIRED_KERNEL.values()), G.served_kernel_names(),
                          log=log, config_kwargs={"load_method": "python"}, spawn=spawn)
    finally:
        C.StubConsumer.recv = orig_recv
        C.sum_check = orig_sum
    return res, ext, recv_log
