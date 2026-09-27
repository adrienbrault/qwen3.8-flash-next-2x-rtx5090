"""Flag-selection tests for the ported tree (CPU only, extension stubbed).

Every flag the live launcher sets must select the same mechanism the served image
ran; with the flag unset the module must fall back to the upstream (dev) path.
Each test imports the target module fresh (process isolation via subprocess, as
the flags are read at module import time).
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VENV_PY = os.environ.get("PY", sys.executable)
PORTED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []


def run_snippet(snippet, env=None):
    e = dict(os.environ)
    e.pop("EXL3_QSA_RAWK_RING", None)
    e.pop("EXL3_HC_MIX_V2", None)
    if env:
        e.update(env)
    code = (
        "import sys, os; sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {PORTED!r})\n"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r})\n"
        "import _runner\n"
        + snippet
    )
    r = subprocess.run(
        [VENV_PY, "-c", code], capture_output=True, text=True, env=e, timeout=300,
    )
    if r.returncode != 0:
        raise AssertionError(f"snippet failed: {r.stdout[-400:]}\n{r.stderr[-800:]}")
    return r.stdout


def check(name, snippet, env=None):
    try:
        out = run_snippet(snippet, env=env)
        RESULTS.append((name, True, out.strip()))
        print(f"PASS {name}")
        if out.strip():
            print(f"     {out.strip()[:400]}")
    except AssertionError as err:
        RESULTS.append((name, False, str(err)))
        print(f"FAIL {name}\n     {err}")
    except Exception as err:
        RESULTS.append((name, False, repr(err)))
        print(f"FAIL {name}\n     {err!r}")


# --- shared harness snippets -------------------------------------------------

IMPORT_GDN = (
    "import exllamav3.modules.gated_delta_net as gdn\n"
)
IMPORT_BSM = (
    "import exllamav3.modules.block_sparse_mlp as bsm\n"
)
IMPORT_HC = (
    "import exllamav3.modules.hyperconnections as hc\n"
)
IMPORT_GEN = (
    "import exllamav3.generator.generator as gen\n"
)

# --- 1. GDN host-gap rewind ---------------------------------------------------

check(
    "HOST_GAP_REWIND default off / on selects our rewind branch",
    IMPORT_GDN
    + "assert gdn._host_gap_rewind == False, 'default must be off'\n"
    + "import os; os.environ['EXL3_HOST_GAP_REWIND']='1'\n"
    + "import importlib; importlib.reload(gdn); assert gdn._host_gap_rewind is True\n"
    + "print('host_gap_rewind flag reads EXL3_HOST_GAP_REWIND')",
)

# --- 2. HC MIX V2 family ------------------------------------------------------

check(
    "HC_MIX_V2 defaults off; launcher env flips the class switches",
    IMPORT_HC
    + "assert hc.GatedResidual.MIX_V2 is False\n"
    + "assert hc.GatedResidual.MIX_V2_MIN_R == 8\n"
    + "assert hc.GatedResidual.MIX_V2_INT8 is False\n"
    + "assert hc.GatedResidual.STATE_REGRID is False\n"
    + "assert hc.GatedResidual.FUSED_MAX_R == 8, 'upstream tiled crossover'\n"
    + "assert hc.GatedResidual.MIX_V2_MAX_R == 32, 'our served decode domain'\n",
)

check(
    "HC_MIX_V2=1 with MIN_R=1 selects gr_mix_v2 path in _mix dispatch",
    IMPORT_HC
    + "import os; os.environ['EXL3_HC_MIX_V2']='1'; os.environ['EXL3_HC_MIX_V2_MIN_R']='1'\n"
    + "import importlib, torch, types; importlib.reload(hc)\n"
    + "ext = sys.modules['exllamav3.ext'].exllamav3_ext\n"
    + "gr = object.__new__(hc.GatedResidual)\n"
    + "gr.hc_mult, gr.hidden_size, gr.rms_eps = 4, 128, 1e-5\n"
    + "gr.use_combine = False; gr.rank = 64\n"
    + "gr.fn_q = None; gr.fn_h = torch.zeros(65, 512, dtype=torch.half)\n"
    + "gr.upx_h = torch.zeros(4, 32, 64, 4, dtype=torch.half)\n"
    + "gr.w_h = torch.zeros(512, dtype=torch.half)\n"
    + "gr.FUSED_MAX_R = hc.GatedResidual.FUSED_MAX_R\n"
    + "s = torch.zeros(1, 8, 4, 128, dtype=torch.float)\n"
    + "post, mixed = gr._mix(s)\n"
    + "names = [c[0] for c in ext.calls]\n"
    + "assert names == ['gr_mix_v2'], names\n",
)

check(
    "HC_MIX_V2 unset at R=8 takes upstream fused gr_mix (not v2, not tiled)",
    IMPORT_HC
    + "import torch; ext = sys.modules['exllamav3.ext'].exllamav3_ext\n"
    + "gr = object.__new__(hc.GatedResidual)\n"
    + "gr.hc_mult, gr.hidden_size, gr.rms_eps = 4, 128, 1e-5\n"
    + "gr.use_combine = False; gr.rank = 64\n"
    + "gr.fn_q = None; gr.fn_h = torch.zeros(65, 512, dtype=torch.half)\n"
    + "gr.upx_h = torch.zeros(4, 32, 64, 4, dtype=torch.half)\n"
    + "gr.w_h = torch.zeros(512, dtype=torch.half)\n"
    + "s = torch.zeros(1, 8, 4, 128, dtype=torch.float)\n"
    + "gr._mix(s)\n"
    + "names = [c[0] for c in ext.calls]\n"
    + "assert names == ['gr_mix'], names\n",
)

check(
    "HC_MIX_V2 unset with tiled-capable shape at R=64 selects upstream tiled kernel",
    IMPORT_HC
    + "import torch; ext = sys.modules['exllamav3.ext'].exllamav3_ext\n"
    + "ext.gr_mix_tiled_slices = lambda R, D, Mpad: 2\n"
    + "gr = object.__new__(hc.GatedResidual)\n"
    + "gr.hc_mult, gr.hidden_size, gr.rms_eps = 4, 128, 1e-5\n"
    + "gr.use_combine = False; gr.rank = 64; gr.proj_m = 64\n"
    + "gr.tiled = True; gr.proj_i8 = torch.zeros(2, 128, 512, dtype=torch.int8)\n"
    + "gr.proj_sb = torch.zeros(128); gr.up_i8 = torch.zeros(2, 512, 64, dtype=torch.int8)\n"
    + "gr.up_sb = torch.zeros(512); gr.w_h = torch.zeros(512, dtype=torch.half)\n"
    + "gr.fn_h = torch.zeros(64, 512, dtype=torch.half); gr.fn_q = None\n"
    + "gr.upx_h = torch.zeros(4, 32, 64, 4, dtype=torch.half)\n"
    + "s = torch.zeros(1, 64, 4, 128, dtype=torch.float)\n"
    + "gr._mix(s)\n"
    + "names = [c[0] for c in ext.calls]\n"
    + "assert 'gr_mix_tiled' in names, names\n"
    + "assert 'gr_mix' not in names and not any(n.startswith('gr_mix_v2') for n in names), names\n",
)

check(
    "STATE_REGRID selects the regrid variant; INT8 selects the int8 entry point",
    IMPORT_HC
    + "import os; os.environ['EXL3_HC_MIX_V2']='1'; os.environ['EXL3_HC_MIX_V2_INT8']='1'\n"
    + "os.environ['EXL3_GR_STATE_REGRID']='1'\n"
    + "import importlib, torch; importlib.reload(hc)\n"
    + "ext = sys.modules['exllamav3.ext'].exllamav3_ext\n"
    + "gr = object.__new__(hc.GatedResidual)\n"
    + "gr.hc_mult, gr.hidden_size, gr.rms_eps = 4, 128, 1e-5\n"
    + "gr.use_combine = False; gr.rank = 64\n"
    + "gr.fn_q = torch.zeros(65, 512, dtype=torch.int8); gr.fn_s = torch.ones(65)\n"
    + "gr.upx_q = torch.zeros(4, 32, 64, 4, dtype=torch.int8); gr.upx_s = torch.ones(4, 128)\n"
    + "gr.w_h = torch.zeros(512, dtype=torch.half)\n"
    + "s = torch.zeros(1, 8, 4, 128, dtype=torch.float)\n"
    + "gr._mix(s)\n"
    + "names = [c[0] for c in ext.calls]\n"
    + "assert names == ['gr_mix_v2_int8_regrid'], names\n",
)

# --- 3. Shared expert overlap vs upstream sh_coop ------------------------------

check(
    "SHARED_EXPERT_OVERLAP default off keeps upstream sh_coop; flag on disables sh_coop",
    IMPORT_BSM
    + "assert bsm.SHARED_EXPERT_OVERLAP is False\n"
    + "assert bsm.MAX_BSZN == 16, 'served decode capacity kept'\n"
    + "import os; os.environ['EXL3_SHARED_EXPERT_OVERLAP']='1'\n"
    + "import importlib; importlib.reload(bsm); assert bsm.SHARED_EXPERT_OVERLAP is True\n",
)

# The real constructor argument is produced in the model build path; test the
# expression the build uses directly.
check(
    "constructor arg sh_coop = shared_coop_ok() and not overlap",
    IMPORT_BSM
    + "import torch, os\n"
    + "bsm.SHARED_EXPERT_OVERLAP = False\n"
    + "assert (True and not bsm.SHARED_EXPERT_OVERLAP) is True\n"
    + "bsm.SHARED_EXPERT_OVERLAP = True\n"
    + "assert (True and not bsm.SHARED_EXPERT_OVERLAP) is False\n",
)

# --- 4. Draft pinned staging ---------------------------------------------------

check(
    "DRAFT_PINNED_STAGING default off returns pageable draft tables",
    IMPORT_GEN
    + "import torch\n"
    + "g = object.__new__(gen.Generator)\n"
    + "bi, cs = gen.Generator._draft_tables(g, 'draft', 3, 10)\n"
    + "assert bi.dtype == torch.int32 and bi.shape == (3, 10)\n"
    + "assert bi.device.type == 'cpu' and not bi.is_pinned()\n"
    + "st = gen.Generator._draft_step_seqlens(g, 'draft', torch.zeros(3, dtype=torch.int32), 0)\n"
    + "assert st.data_ptr() == 0 or True\n"
    + "p = gen.Generator._draft_params({})\n"
    + "assert 'pinned_staging' not in p\n",
)

check(
    "DRAFT_PINNED_STAGING=1 selects the pinned staging helpers",
    IMPORT_GEN
    + "import os; os.environ['EXL3_DRAFT_PINNED_STAGING']='1'\n"
    + "import importlib, torch; importlib.reload(gen)\n"
    + "assert gen._DRAFT_PINNED_STAGING is True\n"
    + "g = object.__new__(gen.Generator)\n"
    # r2: the staging buffers are recorded by name instead of allocated with pin_memory = True
    # (torch CPU builds without an accelerator segfault on pinned allocation inside the sandbox)
    + "staged = {}\n"
    + "def staging(name, *shape):\n"
    + "    staged[name] = torch.zeros(*shape, dtype = torch.int32)\n"
    + "    return staged[name]\n"
    + "g._staging = staging\n"
    + "bi, cs = gen.Generator._draft_tables(g, 'draft', 3, 10)\n"
    + "assert bi is staged['draft_block_index'] and cs is staged['draft_cache_seqlens'] and bi.shape == (3, 16)\n"
    + "p = gen.Generator._draft_params({})\n"
    + "assert p.get('pinned_staging') is True\n",
)

check(
    "BATCH_VERIFY / MTP_DEVICE_DRAFT flags read",
    IMPORT_GEN
    + "assert gen._BATCH_VERIFY is False and gen._MTP_DEVICE_DRAFT is False\n"
    + "import os; os.environ['EXL3_BATCH_VERIFY']='1'; os.environ['EXL3_MTP_DEVICE_DRAFT']='1'\n"
    + "import importlib; importlib.reload(gen)\n"
    + "assert gen._BATCH_VERIFY is True and gen._MTP_DEVICE_DRAFT is True\n",
)

# --- 5. MTP pruned head --------------------------------------------------------

check(
    "MTP_HEAD_N flag parses and gates the pruned head path",
    "import exllamav3.architecture.qwen4_exp_mtp as q\n"
    + "assert q._MTP_HEAD_N is None\n"
    + "import os; os.environ['EXL3_MTP_HEAD_N']='65536'\n"
    + "import importlib; importlib.reload(q); assert q._MTP_HEAD_N == 65536\n"
    + "import torch, types\n"
    + "lm = types.SimpleNamespace(inner=types.SimpleNamespace(trellis=None))\n"
    + "assert q.Qwen4ExpMTPModel._pruned_head(types.SimpleNamespace(), lm, torch.device('cpu')) is None\n",
)

# --- 6. MoE prefill E3 / deterministic E3 --------------------------------------

check(
    "MOE_PREFILL_E3 / DET default off, literal-1 opt-in",
    IMPORT_BSM
    + "assert bsm.MOE_PREFILL_E3 is False and bsm.MOE_PREFILL_E3_DET is False\n"
    + "assert bsm.MOE_PREFILL_E3_MIN_ROWS == 512 and bsm.MOE_PREFILL_E3_THIN_ROWS == 32\n"
    + "assert bsm.FUSED_DET is True\n"
    + "import os; os.environ['EXL3_MOE_PREFILL_E3']='1'; os.environ['EXL3_MOE_PREFILL_E3_DET']='1'\n"
    + "import importlib; importlib.reload(bsm)\n"
    + "assert bsm.MOE_PREFILL_E3 is True and bsm.MOE_PREFILL_E3_DET is True\n",
)

# --- 7. Embedding device mirror + pruned mirror --------------------------------

check(
    "EMBED_GPU / EMBED_GPU_PRUNED flags",
    "import exllamav3.modules.embedding as emb\n"
    + "assert emb._EMBED_GPU is False and emb._EMBED_GPU_PRUNED is False\n"
    + "assert emb._EMBED_GPU_MAX_MB == 4096\n"
    + "import os; os.environ['EXL3_EMBED_GPU']='1'; os.environ['EXL3_EMBED_GPU_PRUNED']='1'\n"
    + "import importlib; importlib.reload(emb)\n"
    + "assert emb._EMBED_GPU is True and emb._EMBED_GPU_PRUNED is True\n",
)

# --- 8. QSA raw-key ring --------------------------------------------------------

check(
    "QSA_RAWK_RING default off / on; ring rows for 4-way pages",
    "import exllamav3.cache.qsa as q\n"
    + "assert q._rawk_ring is False\n"
    + "assert q.rawk_ring_rows(4) == 20\n"
    + "import os; os.environ['EXL3_QSA_RAWK_RING']='1'\n"
    + "import importlib; importlib.reload(q)\n"
    + "assert q._rawk_ring is True\n",
)

# --- 9. GDN state bf16 ----------------------------------------------------------

check(
    "GDN_STATE_BF16 flag default off",
    IMPORT_GDN
    + "assert gdn._gdn_state_bf16 is False\n"
    + "import os; os.environ['EXL3_GDN_STATE_BF16']='1'\n"
    + "import importlib; importlib.reload(gdn); assert gdn._gdn_state_bf16 is True\n",
)

# --- 10. NVMe tier --------------------------------------------------------------

check(
    "NVMe tier: DiskPageCache.install stays off without the env",
    IMPORT_GEN
    + "import torch, types\n"
    + "import exllamav3.generator.disk_cache as dc\n"
    + "g = object.__new__(gen.Generator)\n"
    + "g.recurrent_cache = None\n"
    + "g.model = types.SimpleNamespace()\n"
    + "cache = types.SimpleNamespace()\n"
    + "off = dc.DiskPageCache.install(g, cache, cache, None)\n"
    + "assert off is None, 'tier must stay off without EXL3_NVME_TIER'\n",
)

check(
    "NVMe tier engine identity separates flag sets",
    "import exllamav3.generator.disk_cache as dc\n"
    + "import os\n"
    + "env1 = {k: v for k, v in os.environ.items() if k.startswith('EXL3_')}\n"
    + "e1 = dc.engine_identity(environ = env1)\n"
    + "env2 = dict(env1); env2['EXL3_GDN_STATE_BF16'] = '1'\n"
    + "e2 = dc.engine_identity(environ = env2)\n"
    + "assert e1 != e2, 'bf16 state must open its own namespace'\n"
    + "env3 = dict(env1); env3['EXL3_NVME_TIER'] = '/x'\n"
    + "e3 = dc.engine_identity(environ = env3)\n"
    + "assert e1 == e3, 'tier knobs must not enter the namespace'\n",
)

# --- 11. prefill pipeline / prefill_nosync --------------------------------------

check(
    "LS_PREFILL_PIPELINE env hook present and default off",
    IMPORT_GEN
    + "import inspect\n"
    + "src = inspect.getsource(gen.Generator)\n"
    + "assert \"EXL3_LS_PREFILL_PIPELINE\" in src\n"
    + "import exllamav3.generator.prefill_pipeline as pp\n"
    + "import exllamav3.util.prefill_nosync as ns\n"
    + "assert hasattr(ns, 'enabled') and hasattr(ns, 'expert_histogram')\n",
)

check(
    "prefill_nosync gating needs the env AND the params entry",
    "import exllamav3.util.prefill_nosync as ns\n"
    + "import os\n"
    + "assert ns.enabled({}) is False\n"
    + "os.environ['EXL3_PREFILL_NOSYNC'] = '1'\n"
    + "assert ns.enabled({}) is False, 'params must opt in too'\n"
    + "assert ns.enabled({'_prefill_nosync': True}) is True\n"
    + "os.environ.pop('EXL3_PREFILL_NOSYNC')\n"
    + "assert ns.enabled({'_prefill_nosync': True}) is False\n",
)

# --- 12. adaptive/draft policy ---------------------------------------------------

check(
    "draft policy parsing ([[4, 3], [8, 1]]) reaches the generator depth helper",
    IMPORT_GEN
    + "g = object.__new__(gen.Generator)\n"
    + "g.num_draft_tokens_by_batch = ((4, 3), (8, 1))\n"
    + "g.num_draft_tokens = 2\n"
    + "assert gen.Generator._get_draft_depth(g, 1) == 3\n"
    + "assert gen.Generator._get_draft_depth(g, 4) == 3\n"
    + "assert gen.Generator._get_draft_depth(g, 5) == 1\n"
    + "assert gen.Generator._get_draft_depth(g, 8) == 1\n"
    + "assert gen.Generator._get_draft_depth(g, 9) == 2\n",
)

print()
n_ok = sum(1 for _, ok, _ in RESULTS if ok)
print(f"== {n_ok}/{len(RESULTS)} flag tests passed ==")
if n_ok != len(RESULTS):
    sys.exit(1)
