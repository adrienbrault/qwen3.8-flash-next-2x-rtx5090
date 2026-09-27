#!/usr/bin/env python3
"""rebase-dev r3 landing check, run inside the built image (python3, CPU only, no GPU needed).

  landing_r3.py --tree r3        tabbyapi:rebase-dev-r3  (served stack ported onto upstream dev 5783a93, v1.5.2)
  landing_r3.py --tree vanilla   tabbyapi:dev-vanilla-r3 (upstream dev 5783a93, unmodified)

Base image for both: tabbyapi:stack-r3-rows32-tokcount-loopthink4 (served stack-r3-rows32 + tokcount-r1 in site-packages,
loop-think r4 in /app). Last line "rebase-dev-r3 landed: ..." / "dev-vanilla-r3 landed: ..." and exit 0, or an
AssertionError.

1. The base image bakes EXL3_* keys into its ENV. The checks are about defaults, so the script re-executes itself once
   with every EXL3_* key stripped, before anything imports exllamav3 (the r2 / stack-r3 / rows32 landing pattern).
2. Where things come from: exllamav3 and the precompiled exllamav3_ext both resolve inside site-packages (purelib), no
   exllamav3_ext build is left in TORCH_EXTENSIONS_DIR, the version is upstream's 1.5.2. torch is imported BEFORE the
   extension (it links libc10.so, R700).
3. Upstream dev entry points present in both trees: tiled HC mix, deterministic routing GEMM, fractional-K quant,
   DFlash2, and r3's new g_get_smem_max + attention_fn/smem.py + bc_attn.BCKernelTooLarge. Whether torch binds
   shared_memory_per_block_optin is reported (a WARNING in the last line if not: smem.py would then assume 96 KiB,
   below sm_120's 99 KiB opt-in).
4. r3 only: the served image's own landing checks, unchanged, against the ported tree (/opt/rows32-r4/landing_rows32.py
   --include "mf3 dg2", which runs /opt/stack-r3/tools/landing.py first); the full served env (the live launcher's 41
   EXTRA_ENV keys) imports every flag-reading module with the real extension and resolves to the served values; the
   tokcount-r1 line is present, and the base's own /opt/tokcount-r1/landing_tokcount.py passes on the ported job.py.
5. vanilla only: none of the served markers exist, and the requeue line is upstream's.
6. Both: the base's /opt/loopthink-r4/landing_loopthink.py passes against the new exllamav3 (it imports /app's patched
   chat_completion, which imports exllamav3.generator.loop_detect.LoopDetector), and the loop-think engine contract
   holds (LoopDetector(W, P), feed_many, _total; Job max_rq_tokens / stop_on_loop, detector rebuilt per requeue;
   AsyncJob.constrain_output_now).
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import sysconfig

# R741 run 1: `python3 /opt/<tree>/landing_*.py` puts /opt/<tree> first on sys.path, and /opt/<tree> holds the COPYed
# source tree (exllamav3/), so `import exllamav3` found the copy instead of site-packages and the origin assert fired.
# Drop the script's own directory (and a cwd equal to it) before anything imports exllamav3.
_HERE = os.path.dirname(os.path.realpath(__file__))
sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.getcwd()) != _HERE]

REEXEC = "REBASE_DEV_R3_LANDING_REEXEC"
VERSION = "1.5.2"

SERVED_ENV = (
    "EXL3_HOST_GAP_REWIND=1 EXL3_HC_MIX_V2=1 EXL3_HC_MIX_V2_MIN_R=1 EXL3_LS_PREFILL_PIPELINE=1 EXL3_MOE_COOP_V2=1 "
    "EXL3_SHARED_EXPERT_OVERLAP=1 EXL3_DRAFT_PINNED_STAGING=1 EXL3_BATCH_VERIFY=1 EXL3_MTP_HEAD_N=65536 "
    "EXL3_MOE_PREFILL_E3=1 EXL3_HC_MIX_V2_INT8=1 EXL3_MTP_DEVICE_DRAFT=1 EXL3_EMBED_GPU=1 EXL3_EMBED_GPU_PRUNED=1 "
    "EXL3_MOE_PREFILL_E3_DET=1 EXL3_GDN_BA_WARP1=1 EXL3_HC_APPLY_WARP1=1 EXL3_GR_STATE_REGRID=1 EXL3_GR_STATE_IN_UP=1 "
    "EXL3_QSA_RAWK_RING=1 EXL3_GDN_STATE_BF16=1 EXL3_NGRAM_PREFETCH2=1 EXL3_HC_MIX_V3=2 "
    "EXL3_HC_MIX_V3_DOTS_B=1:1,4:2,32:4 EXL3_HC_MIX_V3_UP_B=1:1,8:4,32:8 EXL3_MOE_COOP_V3=3 "
    "EXL3_HC_MIX_V3_DOTS_J=1:4,32:8 EXL3_HC_MIX_V3_DOTS_PF=1:1,32:0 EXL3_HC_MIX_V3_PDL=0 "
    "EXL3_HC_MIX_V3_UP_Q=1:4,8:2,32:4 EXL3_DENSE_V2=1 EXL3_LC_GDN_RR=1 EXL3_LC_QSA_COMBINE_STAGES=1 "
    "EXL3_LC_QSA_DIV16=1 EXL3_LC_QSA_FORK=1 EXL3_LC_QSA_SPLIT_STAGES=2 EXL3_MOE_COOP_V3_MAP=2-4:2,17-32:2 "
    "EXL3_SHARED_EXPERT_EARLY=1 EXL3_DENSE_ROWS32=1 EXL3_MOE_COOP_ROWS32=1 EXL3_SHARED_EXPERT_ROWS32=1"
)
FIXED = '"rq_new_tokens": self.rq_new_tokens + self.new_tokens,'
UNFIXED = '"rq_new_tokens": self.new_tokens,'


def child(code, env_add):
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
    env.update(env_add)
    r = subprocess.run([sys.executable, "-c", code], env = env, capture_output = True, text = True)
    assert r.returncode == 0, f"child check failed with {sorted(env_add)}:\n{(r.stdout + r.stderr)[-2000:]}"
    return r.stdout.strip()


def run_base_landing(path, what):
    assert os.path.isfile(path), f"{path} missing: the base must be tabbyapi:stack-r3-rows32-tokcount-loopthink4 ({what})"
    r = subprocess.run([sys.executable, path], capture_output = True, text = True, cwd = "/")
    out = (r.stdout + r.stderr).strip()
    assert r.returncode == 0, f"{what} landing failed against the new exllamav3:\n{out[-2000:]}"
    return out.splitlines()[-1][:300] if out else ""


def loopthink_contract():
    import inspect
    from exllamav3.generator.loop_detect import LoopDetector
    from exllamav3.generator.job import Job
    from exllamav3 import AsyncJob
    ps = inspect.signature(LoopDetector.__init__).parameters
    assert list(ps)[1:3] == ["window_size", "max_period"], list(ps)
    d = LoopDetector(8, 4)
    assert d._total == 0 and not d.feed_many([1, 2, 3]) and d.feed_many([1, 2, 3] * 3) and d._total == 12
    jp = inspect.signature(Job.__init__).parameters
    assert "max_rq_tokens" in jp and "stop_on_loop" in jp
    assert "self.loop_detector = LoopDetector(window_size, window_size // min_reps)" in inspect.getsource(Job.__init__)
    assert "stop_on_loop = self.stop_on_loop," in inspect.getsource(Job.prepare_for_requeue)
    assert callable(getattr(Job, "constrain_output_now", None)) and callable(getattr(AsyncJob, "constrain_output_now", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required = True, choices = ("r3", "vanilla"))
    ap.add_argument("--rows32-landing", default = "/opt/rows32-r4/landing_rows32.py")
    ap.add_argument("--tokcount-landing", default = "/opt/tokcount-r1/landing_tokcount.py")
    ap.add_argument("--loopthink-landing", default = "/opt/loopthink-r4/landing_loopthink.py")
    a = ap.parse_args()
    leaked = sorted(k for k in os.environ if k.startswith("EXL3_"))
    if leaked:
        assert os.environ.get(REEXEC) != "1", f"EXL3_* keys survived the re-exec: {leaked}"
        print(f"landing: stripping image EXL3_* keys and re-executing: {leaked}", flush = True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
        env[REEXEC] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

    site = os.path.realpath(sysconfig.get_paths()["purelib"])
    import torch  # noqa: F401  (before the extension)
    so = importlib.util.find_spec("exllamav3_ext").origin
    assert os.path.realpath(os.path.dirname(so)) == site, f"exllamav3_ext from {so}, expected the precompiled build in {site}"
    import exllamav3_ext as e
    import exllamav3
    from exllamav3.version import __version__ as ver
    import exllamav3.ext as xext
    assert xext.exllamav3_ext is e, "exllamav3.ext did not take the precompiled extension"
    pkg = os.path.realpath(os.path.dirname(exllamav3.__file__))
    assert pkg == os.path.join(site, "exllamav3"), f"exllamav3 imported from {pkg}"
    assert ver == VERSION, f"version {ver}, expected upstream dev's {VERSION}"
    ted = os.environ.get("TORCH_EXTENSIONS_DIR") or os.path.expanduser("~/.cache/torch_extensions")
    stale = [os.path.join(r, d) for r, ds, _ in os.walk(ted) for d in ds if d.startswith("exllamav3_ext")] if os.path.isdir(ted) else []
    assert not stale, f"stale JIT builds of exllamav3_ext in {ted}: {stale}"
    print(f"ext: {so}; exllamav3 {ver} from {pkg}")

    # upstream dev entry points (both trees), r3's smem budget included
    for name in ("gr_mix_tiled", "gr_mix_tiled_slices", "routing_gemm_det", "det_quant_weight", "dflash2_topk",
                 "g_get_smem_max", "exl3_gemm_shape_compat"):
        assert hasattr(e, name), f"upstream dev symbol {name} missing"
    # torch >= 2.7 binds it (upstream PR #325 commit 1873964). Without it smem.py assumes 96 KiB (sm_120 grants 99 KiB),
    # and a BC kernel between 96 and 99 KiB would decline to eager: slower, not wrong. Reported, not fatal; the box can
    # prove no decline with EXL3_TRITON_SMEM_DEBUG=1 (grep "declining to eager").
    optin = hasattr(torch._C._CudaDeviceProperties, "shared_memory_per_block_optin")
    optin_note = "optin attr present" if optin else "WARNING: torch lacks shared_memory_per_block_optin, smem.py uses 96 KiB"
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}): {optin_note}")
    import exllamav3.modules.hyperconnections as hc
    import exllamav3.modules.block_sparse_mlp as bsm
    import exllamav3.modules.attention_fn.smem as smem
    import exllamav3.modules.attention_fn.bc_attn as ba
    assert hc._gr_mix_tiled_enable is True, "EXL3_GR_MIX_TILED must default on (upstream)"
    assert bsm._moe_shared_coop is True, "EXL3_MOE_SHARED_COOP must default on (upstream)"
    assert smem._env_limit == 0, "EXL3_TRITON_SMEM_LIMIT is set: Triton tiles would be sized below the device limit"
    assert issubclass(ba.BCKernelTooLarge, RuntimeError)
    job_src = open(os.path.join(pkg, "generator", "job.py")).read()
    loopthink_contract()
    lt = run_base_landing(a.loopthink_landing, "loop-think r4")
    print("loop-think r4 landing (on this exllamav3):", lt)

    if a.tree == "vanilla":
        for name in ("hc_mix_v3_revision", "latchain_revision", "dense_v2_revision", "moe_rows32_revision",
                     "moe_coop_v3_revision", "gr_mix_v2_int8_v3", "exl3_moe_coop_ev", "exl3_moe_prefill_e3"):
            assert not hasattr(e, name), f"served symbol {name} present in the vanilla build"
        assert UNFIXED in job_src and FIXED not in job_src, "vanilla job.py must carry upstream's requeue line"
        import exllamav3.generator.generator as gen
        import inspect
        assert "num_draft_tokens_by_batch" not in inspect.signature(gen.Generator.__init__).parameters, \
            "vanilla Generator unexpectedly takes num_draft_tokens_by_batch"
        print(f"dev-vanilla-r3 landed: upstream exllamav3 {ver} (dev 5783a93) with its own precompiled extension; no "
              f"served markers; tiled HC prefill and sh_coop default on; requeue line upstream (count bug present); "
              f"loop-think r4 contract holds; {optin_note}; Generator has no num_draft_tokens_by_batch -> the launcher must run "
              f"DRAFT_POLICY= (empty) on this image")
        return

    # r3: the served image's own landing, unchanged, against the ported tree
    assert os.path.isfile(a.rows32_landing), f"{a.rows32_landing} missing (base must derive from tabbyapi:stack-r3-rows32)"
    r = subprocess.run([sys.executable, a.rows32_landing, "--include", "mf3 dg2"], capture_output = True, text = True)
    out = (r.stdout + r.stderr).strip()
    assert r.returncode == 0, f"served landing (rows32 r4 + stack-r3) failed on the ported tree:\n{out[-2500:]}"
    print("served landing:", out.splitlines()[-1][:300])

    # full served env: every flag-reading module imports with the real extension and resolves as served
    env = dict(kv.split("=", 1) for kv in SERVED_ENV.split())
    assert len(env) == 41, len(env)
    code = (
        "import torch, exllamav3_ext as e\n"
        "import exllamav3.modules.hyperconnections as hc, exllamav3.modules.block_sparse_mlp as bsm\n"
        "import exllamav3.modules.mlp as mlp, exllamav3.modules.attention_fn.bc_attn as ba\n"
        "import exllamav3.modules.gated_delta_net as gdn, exllamav3.modules.ngram_embedding as ng\n"
        "import exllamav3.generator.generator as gen, exllamav3.modules.embedding as emb\n"
        "import exllamav3.architecture.qwen4_exp_mtp as qm, exllamav3.cache.qsa as qsa\n"
        "import exllamav3.generator.disk_cache, exllamav3.generator.prefill_pipeline\n"
        "G = hc.GatedResidual\n"
        "assert G.MIX_V2 and G.MIX_V2_INT8 and G.STATE_REGRID and G.STATE_IN_UP and G.MIX_V3_LEVEL == 2\n"
        "assert bsm.SHARED_EXPERT_OVERLAP and bsm.MOE_PREFILL_E3 and bsm.MOE_PREFILL_E3_DET and bsm.MAX_BSZN == 32 and mlp.MAX_BSZN == 32\n"
        "assert ba._LC_QSA_SPLIT_STAGES == 2 and ba._LC_QSA_COMBINE_STAGES == 1 and ba._LC_QSA_DIV16\n"
        "assert gdn._host_gap_rewind and gdn._gdn_state_bf16 and ng.PREFETCH2 and qsa._rawk_ring\n"
        "assert gen._DRAFT_PINNED_STAGING and gen._BATCH_VERIFY and gen._MTP_DEVICE_DRAFT and gen._EMBED_GPU_PRUNED\n"
        "assert emb._EMBED_GPU and qm._MTP_HEAD_N == 65536\n"
        "assert e.exl3_moe_coop_rows_cap() == 32, e.exl3_moe_coop_rows_cap()\n"
        "assert tuple(e.dense_v2_modes()) == (1, 1), e.dense_v2_modes()\n"
        "import inspect\n"
        "assert 'num_draft_tokens_by_batch' in inspect.signature(gen.Generator.__init__).parameters\n"
        "print('SERVED-ENV OK')\n"
    )
    got = child(code, env)
    assert got.endswith("SERVED-ENV OK"), got
    assert FIXED in job_src and UNFIXED not in job_src, "tokcount-r1 line missing from the ported job.py"
    tc = run_base_landing(a.tokcount_landing, "tokcount r1")
    print("tokcount r1 landing (on the ported job.py):", tc)
    print(f"rebase-dev-r3 landed: exllamav3 {ver} (served stack-r3-rows32 on dev 5783a93), precompiled extension in site; "
          f"served landing (rows32 r4 + stack-r3) PASS on the ported tree; the 41-key served env imports and resolves "
          f"(V3 level 2, rows cap 32, dense (1, 1), E3-DET, LC 2/1/div16); tokcount-r1 present (base landing PASS); "
          f"loop-think r4 landing and engine contract PASS; upstream tiled / det-routing / fractional-K / DFlash2 / "
          f"smem-budget entry points present; {optin_note}")


if __name__ == "__main__":
    main()
