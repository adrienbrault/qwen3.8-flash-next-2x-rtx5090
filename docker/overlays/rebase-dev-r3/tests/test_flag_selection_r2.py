"""rebase-dev r2 flag tests (CPU only, extension stubbed): the served keys added since r1
(stack-r1 .. stack-r3-rows32) plus the r2 merge decisions.

Every test runs in a fresh subprocess (flags are read at import). The stubbed extension records
calls; constants the modules check at import (moe_rows32_revision, ...) are set on the stub.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VENV_PY = os.environ.get("PY", sys.executable)
PORTED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The live launcher's EXTRA_ENV at R717c (flan/launch-flashnext-tabby.sh, 41 keys): the served env.
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
SERVED = dict(kv.split("=", 1) for kv in SERVED_ENV.split())
assert len(SERVED) == 41, len(SERVED)

RESULTS = []

# Constants the modules compare at import (the real extension exports them). RecordingExt only
# synthesises callables for missing attributes, so plain setattr works.
STUB_CONSTS = (
    "ext = _runner.get_ext()\n"
    "ext.moe_rows32_revision = 2\n"
    "ext.hc_mix_v2_revision = 2\n"
    "ext.dense_v2_revision = 1\n"
)


def run_snippet(snippet, env = None):
    e = {k: v for k, v in os.environ.items() if not k.startswith("EXL3_")}
    if env:
        e.update(env)
    code = (
        "import sys, os; sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {PORTED!r})\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        "import _runner\n"
        + STUB_CONSTS
        + snippet
    )
    r = subprocess.run([VENV_PY, "-c", code], capture_output = True, text = True, env = e, timeout = 300)
    if r.returncode != 0:
        raise AssertionError(f"snippet failed: {r.stdout[-400:]}\n{r.stderr[-1200:]}")
    return r.stdout


def check(name, snippet, env = None):
    try:
        out = run_snippet(snippet, env = env)
        RESULTS.append((name, True))
        print(f"PASS {name}")
        if out.strip():
            print(f"     {out.strip()[:400]}")
    except AssertionError as err:
        RESULTS.append((name, False))
        print(f"FAIL {name}\n     {err}")


# --- 1. the whole served env imports cleanly (import-time asserts, table parsing) ----------------

check(
    "served env (41 keys): every flag-reading module imports; values parse as served",
    "import exllamav3.modules.hyperconnections as hc\n"
    "import exllamav3.modules.block_sparse_mlp as bsm\n"
    "import exllamav3.modules.mlp as mlp\n"
    "import exllamav3.modules.attention_fn.bc_attn as ba\n"
    "import exllamav3.modules.gated_delta_net as gdn\n"
    "import exllamav3.modules.ngram_embedding as ng\n"
    "import exllamav3.generator.generator as gen\n"
    "import exllamav3.modules.embedding as emb\n"
    "import exllamav3.architecture.qwen4_exp_mtp as qm\n"
    "import exllamav3.cache.qsa as qsa\n"
    "G = hc.GatedResidual\n"
    "assert G.MIX_V2 and G.MIX_V2_MIN_R == 1 and G.MIX_V2_INT8 and G.STATE_REGRID and G.STATE_IN_UP\n"
    "assert G.MIX_V3 and G.MIX_V3_LEVEL == 2 and G.MIX_V3_PDL is False\n"
    "assert G.MIX_V3_DOTS_B[1] == 1 and G.MIX_V3_DOTS_B[4] == 2 and G.MIX_V3_DOTS_B[5] == 4 and G.MIX_V3_DOTS_B[32] == 4\n"
    "assert G.MIX_V3_UP_B[1] == 1 and G.MIX_V3_UP_B[8] == 4 and G.MIX_V3_UP_B[9] == 8\n"
    "assert G.MIX_V3_DOTS_J[1] == 4 and G.MIX_V3_DOTS_J[2] == 8 and G.MIX_V3_DOTS_PF[1] == 1 and G.MIX_V3_DOTS_PF[2] == 0\n"
    "assert G.MIX_V3_UP_Q[1] == 4 and G.MIX_V3_UP_Q[8] == 2 and G.MIX_V3_UP_Q[9] == 4\n"
    "assert bsm.SHARED_EXPERT_OVERLAP and bsm.MOE_COOP_ROWS32 and bsm.MAX_BSZN == 32 and mlp.MAX_BSZN == 32\n"
    "assert bsm.MOE_PREFILL_E3 and bsm.MOE_PREFILL_E3_DET\n"
    "assert ba._LC_QSA_SPLIT_STAGES == 2 and ba._LC_QSA_COMBINE_STAGES == 1 and ba._LC_QSA_DIV16\n"
    "assert gdn._host_gap_rewind and gdn._gdn_state_bf16 and ng.PREFETCH2 and gen._NGRAM_PREFETCH2\n"
    "assert gen._DRAFT_PINNED_STAGING and gen._BATCH_VERIFY and gen._MTP_DEVICE_DRAFT and gen._EMBED_GPU_PRUNED\n"
    "assert emb._EMBED_GPU and emb._EMBED_GPU_PRUNED and qm._MTP_HEAD_N == 65536 and qsa._rawk_ring\n"
    "print('served env parses: V3 level 2 tables, rows32 capacity 32, E3-DET, LC stages 2/1 + div16')",
    env = SERVED,
)

# Where each served key is read (quoted literal), identical to the served tree stack-r3-rows32
# (audited 2026-09-26 against the served image's /opt/venv exllamav3 tree). C++-read keys are
# checked here by source only; their behaviour is covered by the box A/B.
READERS = {
    "HOST_GAP_REWIND": ["modules/gated_delta_net.py"],
    "HC_MIX_V2": ["modules/hyperconnections.py"],
    "HC_MIX_V2_MIN_R": ["modules/hyperconnections.py"],
    "LS_PREFILL_PIPELINE": ["generator/generator.py"],
    "MOE_COOP_V2": ["exllamav3_ext/quant/exl3_moe_coop.cu"],
    "SHARED_EXPERT_OVERLAP": ["exllamav3_ext/libtorch/blocksparse_mlp.cpp", "modules/block_sparse_mlp.py"],
    "DRAFT_PINNED_STAGING": ["generator/generator.py"],
    "BATCH_VERIFY": ["generator/generator.py"],
    "MTP_HEAD_N": ["architecture/qwen4_exp_mtp.py"],
    "MOE_PREFILL_E3": ["modules/block_sparse_mlp.py"],
    "HC_MIX_V2_INT8": ["modules/hyperconnections.py"],
    "MTP_DEVICE_DRAFT": ["generator/generator.py"],
    "EMBED_GPU": ["modules/embedding.py"],
    "EMBED_GPU_PRUNED": ["generator/generator.py", "modules/embedding.py"],
    "MOE_PREFILL_E3_DET": ["modules/block_sparse_mlp.py"],
    "GDN_BA_WARP1": ["exllamav3_ext/gdn.cu"],
    "HC_APPLY_WARP1": ["exllamav3_ext/hc_mix.cu"],
    "GR_STATE_REGRID": ["modules/hyperconnections.py"],
    "GR_STATE_IN_UP": ["modules/hyperconnections.py"],
    "QSA_RAWK_RING": ["cache/qsa.py"],
    "GDN_STATE_BF16": ["modules/gated_delta_net.py"],
    "NGRAM_PREFETCH2": ["generator/generator.py", "modules/ngram_embedding.py"],
    "HC_MIX_V3": ["modules/hyperconnections.py"],
    "HC_MIX_V3_DOTS_B": ["modules/hyperconnections.py"],
    "HC_MIX_V3_UP_B": ["modules/hyperconnections.py"],
    "MOE_COOP_V3": ["exllamav3_ext/quant/exl3_moe_coop.cu"],
    "HC_MIX_V3_DOTS_J": ["modules/hyperconnections.py"],
    "HC_MIX_V3_DOTS_PF": ["modules/hyperconnections.py"],
    "HC_MIX_V3_PDL": ["modules/hyperconnections.py"],
    "HC_MIX_V3_UP_Q": ["modules/hyperconnections.py"],
    "DENSE_V2": ["exllamav3_ext/quant/exl3_dense_v2.cu"],
    "LC_GDN_RR": ["exllamav3_ext/gdn.cu"],
    "LC_QSA_COMBINE_STAGES": ["modules/attention_fn/bc_attn.py"],
    "LC_QSA_DIV16": ["modules/attention_fn/bc_attn.py"],
    "LC_QSA_FORK": ["exllamav3_ext/libtorch/attention.cpp", "exllamav3_ext/quant/exl3_dense_v2.cu"],
    "LC_QSA_SPLIT_STAGES": ["modules/attention_fn/bc_attn.py"],
    "MOE_COOP_V3_MAP": ["exllamav3_ext/quant/exl3_moe_coop.cu"],
    "SHARED_EXPERT_EARLY": ["exllamav3_ext/libtorch/blocksparse_mlp.cpp"],
    "DENSE_ROWS32": ["exllamav3_ext/quant/exl3_dense_v2.cu"],
    "MOE_COOP_ROWS32": ["exllamav3_ext/quant/exl3_moe_coop.cu", "modules/block_sparse_mlp.py"],
    "SHARED_EXPERT_ROWS32": ["modules/mlp.py"],
}
assert sorted("EXL3_" + k for k in READERS) == sorted(SERVED), "READERS must list exactly the 41 served keys"


def _readers_check():
    pkg = os.path.join(PORTED, "exllamav3")
    found = {k: [] for k in READERS}
    for root, _, files in os.walk(pkg):
        for f in files:
            if not f.endswith((".py", ".cu", ".cpp", ".cuh", ".h")):
                continue
            path = os.path.join(root, f)
            src = open(path, errors = "replace").read()
            rel = os.path.relpath(path, pkg)
            for k in READERS:
                if f'"EXL3_{k}"' in src:
                    found[k].append(rel)
    bad = {k: (sorted(v), READERS[k]) for k, v in found.items() if sorted(v) != sorted(READERS[k])}
    name = "all 41 served keys have their served reader files (py + C++), none dropped by the merge"
    if bad:
        RESULTS.append((name, False))
        print(f"FAIL {name}\n     {bad}")
    else:
        RESULTS.append((name, True))
        print(f"PASS {name}")


_readers_check()

# --- 2. the served decode mixer under the served env: V3 int8 state-in-up at 1..32 rows ----------

MIXER = (
    "import torch, exllamav3.modules.hyperconnections as hc\n"
    "gr = object.__new__(hc.GatedResidual)\n"
    "gr.hc_mult, gr.hidden_size, gr.rms_eps = 4, 128, 1e-5\n"
    "gr.use_combine = True; gr.rank = 64\n"
    "gr.fn_q = torch.zeros(68, 512, dtype=torch.int8); gr.fn_s = torch.ones(68)\n"
    "gr.upx_q = torch.zeros(4, 32, 64, 4, dtype=torch.int8); gr.upx_s = torch.ones(4, 128)\n"
    "gr.fn_h = None; gr.upx_h = None\n"
    "gr.w_h = torch.zeros(512, dtype=torch.half)\n"
    "gr.tiled = True; gr.proj_i8 = torch.zeros(2, 128, 512, dtype=torch.int8); gr.proj_sb = torch.zeros(128)\n"
    "gr.up_i8 = torch.zeros(2, 512, 64, dtype=torch.int8); gr.up_sb = torch.zeros(512); gr.proj_m = 68\n"
    "ext.__dict__['gr_mix_tiled_slices'] = lambda R, D, Mpad: 2\n"
)

check(
    "served env: rows 1, 8, 24, 32 take gr_mix_v2_int8_v3 (V3 level 2, state in up); 33 rows take upstream tiled",
    MIXER
    + "for R in (1, 8, 24, 32):\n"
    "    ext.reset(); gr._mix(torch.zeros(1, R, 4, 128))\n"
    "    names = [c[0] for c in ext.calls]\n"
    "    assert names == ['gr_mix_v2_int8_v3'], (R, names)\n"
    "    args = ext.calls[0][1]\n"
    "    assert args[11] == 15, ('mode', R, args[11])          # 15 | (16 if PDL): PDL off\n"
    "    assert args[12] == hc.GatedResidual.MIX_V3_DOTS_B[R] and args[13] == hc.GatedResidual.MIX_V3_UP_B[R]\n"
    "ext.reset(); gr._mix(torch.zeros(1, 33, 4, 128))\n"
    "names = [c[0] for c in ext.calls]\n"
    "assert 'gr_mix_tiled' in names and not any(n.startswith('gr_mix_v2') for n in names), names\n"
    "print('decode mixer = served V3 kernels (hc_mix_v3.cu, served bytes); prefill = upstream tiled')",
    env = SERVED,
)

check(
    "served env + EXL3_GR_MIX_TILED=0 parses as off (flip arm PT0: prefill mixer back to cuBLAS)",
    "import exllamav3.modules.hyperconnections as hc\n"
    "assert hc._gr_mix_tiled_enable is False\n",
    env = {**SERVED, "EXL3_GR_MIX_TILED": "0"},
)

check(
    "EXL3_GR_MIX_TILED default on (upstream); EXL3_HC_MIX_V3 / GR_STATE_IN_UP default off",
    "import exllamav3.modules.hyperconnections as hc\n"
    "assert hc._gr_mix_tiled_enable is True\n"
    "G = hc.GatedResidual\n"
    "assert G.MIX_V3 is False and G.MIX_V3_LEVEL == 0 and G.STATE_IN_UP is False and G.MIX_V2 is False\n",
)

check(
    "GR_STATE_IN_UP=1 without V3 takes gr_mix_v2_int8_statein (mixstate r1)",
    MIXER
    + "ext.reset(); gr._mix(torch.zeros(1, 8, 4, 128))\n"
    "names = [c[0] for c in ext.calls]\n"
    "assert names == ['gr_mix_v2_int8_statein'], names\n",
    env = {"EXL3_HC_MIX_V2": "1", "EXL3_HC_MIX_V2_MIN_R": "1", "EXL3_HC_MIX_V2_INT8": "1", "EXL3_GR_STATE_IN_UP": "1"},
)

# --- 3. rows32 and the shared-expert keys ------------------------------------------------------

check(
    "rows32 flags default off: fused decode capacity 16 (served pre-rows32), strict 0/1 parsing",
    "import exllamav3.modules.block_sparse_mlp as bsm, exllamav3.modules.mlp as mlp\n"
    "assert bsm.MOE_COOP_ROWS32 is False and bsm.MAX_BSZN == 16 and mlp.MAX_BSZN == 16\n"
    "os.environ['EXL3_X'] = 'true'\n"
    "try:\n"
    "    mlp._rows32_flag('EXL3_X')\n"
    "    raise SystemExit('non-literal accepted')\n"
    "except ValueError:\n"
    "    pass\n",
)

check(
    "EXL3_MOE_COOP_ROWS32=1 without EXL3_SHARED_EXPERT_ROWS32 refuses to import (served guard kept)",
    "try:\n"
    "    import exllamav3.modules.block_sparse_mlp as bsm\n"
    "    raise SystemExit('imported')\n"
    "except RuntimeError as e:\n"
    "    assert 'SHARED_EXPERT_ROWS32' in str(e), e\n",
    env = {"EXL3_MOE_COOP_ROWS32": "1"},
)

check(
    "flag-flip arm F1 must drop SHARED_EXPERT_EARLY with SHARED_EXPERT_OVERLAP (C++ ctor TORCH_CHECK); sh_coop takes over",
    "import exllamav3.modules.block_sparse_mlp as bsm\n"
    "assert bsm.SHARED_EXPERT_OVERLAP is False and bsm._moe_shared_coop is True\n"
    "src = open(os.path.join(os.path.dirname(bsm.__file__), '..', 'exllamav3_ext', 'libtorch', 'blocksparse_mlp.cpp')).read()\n"
    "assert 'EXL3_SHARED_EXPERT_EARLY / EXL3_SHARED_EXPERT_PRIO require EXL3_SHARED_EXPERT_OVERLAP=1' in src\n",
    env = {k: v for k, v in SERVED.items() if k not in ("EXL3_SHARED_EXPERT_OVERLAP", "EXL3_SHARED_EXPERT_EARLY")},
)

# --- 4. keys that are off in the served env stay off ------------------------------------------------

check(
    "MTP KV window unset = off (served since R728); NGRAM_PREFETCH2 default off",
    "import exllamav3.cache.mtp_window as mw, exllamav3.modules.ngram_embedding as ng\n"
    "assert mw._parse_window(os.environ.get('EXL3_MTP_KV_WINDOW')) == 0\n"
    "assert ng.PREFETCH2 is False\n",
)

# --- 5. r2 merge decisions ----------------------------------------------------------------------------

check(
    "tokcount-r1 fix present (requeue carries the cumulative count), flag-free",
    "import inspect, exllamav3.generator.job as job\n"
    "src = inspect.getsource(job)\n"
    "assert '\"rq_new_tokens\": self.rq_new_tokens + self.new_tokens,' in src\n"
    "assert '\"rq_new_tokens\": self.new_tokens,' not in src\n",
)

check(
    "fractional-K bridge: served coop / densegemm paths gated on integer bitrates in the C++ sources",
    "ext_dir = os.path.join(os.path.dirname(__import__('exllamav3').__file__), 'exllamav3_ext')\n"
    "coop = open(os.path.join(ext_dir, 'quant', 'exl3_moe_coop.cu')).read()\n"
    "assert 'const BitsK bk_gu = bits_from_K(K_gu), bk_d = bits_from_K(K_d);' in coop\n"
    "assert 'if (v2 && !bk_gu.half && !bk_d.half' in coop\n"
    "gemm = open(os.path.join(ext_dir, 'quant', 'exl3_gemm.cu')).read()\n"
    "assert gemm.count('!half_k && ') == 2, gemm.count('!half_k && ')\n"
    "gemv = open(os.path.join(ext_dir, 'quant', 'exl3_gemv.cu')).read()\n"
    "assert 'if (!half_k && !smem && dense_v2_gemv_on())' in gemv\n"
    "v2k = open(os.path.join(ext_dir, 'quant', 'exl3_gemm_v2_kernel.cuh')).read()\n"
    "assert v2k.count('static_assert(!half_k') == 2 and '_v2_kernel<_bits, false, _c_fp32, _cb,' in v2k\n",
)

check(
    "upstream's float-K ABI reaches the served coop launcher (float K_gu / K_d in every declaration)",
    "ext_dir = os.path.join(os.path.dirname(__import__('exllamav3').__file__), 'exllamav3_ext')\n"
    "h = open(os.path.join(ext_dir, 'quant', 'exl3_moe_coop.cuh')).read()\n"
    "assert 'int K_gu' not in h and 'int Kg, int Ku' not in h, 'int-K declaration left'\n"
    "assert h.count('float Kg, float Ku, float Kd,') == 3\n",
)

check(
    "version = upstream 1.5.2 (dev 5783a93; rebase-dev r3 runs this r2 suite, only this check changed)",
    "import exllamav3\n"
    "from exllamav3.version import __version__ as v\n"
    "assert v == '1.5.2', v\n",
)

passed = sum(1 for _, ok in RESULTS if ok)
print(f"\n== {passed}/{len(RESULTS)} r2 flag tests passed ==")
sys.exit(0 if passed == len(RESULTS) else 1)
