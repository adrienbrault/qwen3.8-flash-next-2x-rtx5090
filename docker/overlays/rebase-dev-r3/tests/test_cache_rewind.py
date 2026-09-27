"""Cache / pagetable / disk-cache logic tests (CPU only).

Covers the served overlays' pure-Python data structures: the QSA raw-key ring
plane shapes and storage accounting, the GDN rewind job descriptors (flag on
and off, fp32 and bf16 states), and the NVMe tier namespace contract.
"""
import os
import sys
import types

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _runner  # noqa: E402  (stubs exllamav3.ext and triton)

import torch  # noqa: E402

import exllamav3.cache.qsa as qsa  # noqa: E402
import exllamav3.modules.gated_delta_net as gdn  # noqa: E402
import exllamav3.generator.disk_cache as dc  # noqa: E402

from exllamav3.constants import PAGE_SIZE  # noqa: E402

ext = sys.modules["exllamav3.ext"].exllamav3_ext

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception as e:
        import traceback
        RESULTS.append((name, False, traceback.format_exc()))
        print(f"FAIL {name}\n     {e!r}")


# --- 1. QSA raw-key ring -------------------------------------------------------

def qsa_ring_rows():
    # 20 ring rows per page for the served compress ratio 4 (ring + 4 spare rows)
    assert qsa.rawk_ring_rows(4) == 20
    assert qsa._rawk_ring is False


def qsa_ring_layout():
    # Ring plane shape vs the served full plane, storage delta per layer
    ring = qsa.CacheLayer_qsa_quant.__new__(qsa.CacheLayer_qsa_quant)
    ring.index_head_dim = 128
    ring.compress_ratio = 4
    num_pages = 3
    ring.raw_ring_rows = qsa.rawk_ring_rows(4)
    ring.raw_k_shape = (num_pages, ring.raw_ring_rows, ring.index_head_dim)
    ring.pooled_shape = (num_pages, 8, 128)
    ring.raw_k = torch.zeros(ring.raw_k_shape, dtype=torch.half)
    ring.pooled = torch.zeros(ring.pooled_shape, dtype=torch.half)
    served_shape = (num_pages, PAGE_SIZE, 128)
    ring_served = torch.zeros(served_shape, dtype=torch.half)
    diff = (ring_served.numel() - ring.raw_k.numel()) * 2
    # 236 raw rows dropped per page: (256 - 20) * 128 * 2 B = 60,416 B/page
    assert diff == num_pages * (PAGE_SIZE - 20) * 128 * 2


def qsa_ring_copy():
    # Ring layout, block-aligned copy: the raw ring rows are final pooled keys'
    # property (no partial copies), so only the pooled rows move; a partial-page
    # copy must raise (the generator never does this on QSA models)
    ring = qsa.CacheLayer_qsa_quant.__new__(qsa.CacheLayer_qsa_quant)
    ring.index_head_dim = 128
    ring.compress_ratio = 4
    ring.raw_ring_rows = qsa.rawk_ring_rows(4)
    ring.raw_k_shape = (3, ring.raw_ring_rows, 128)
    ring.pooled_shape = (3, 8, 128)
    ring.raw_k = torch.zeros(ring.raw_k_shape, dtype=torch.half)
    ring.pooled = torch.zeros(ring.pooled_shape, dtype=torch.half)
    dst = qsa.CacheLayer_qsa_quant.__new__(qsa.CacheLayer_qsa_quant)
    dst.raw_ring_rows = ring.raw_ring_rows
    dst.compress_ratio = 4
    dst.raw_k = torch.zeros_like(ring.raw_k)
    dst.pooled = torch.zeros_like(ring.pooled)
    # the quant layer's page copy (super().copy_page) moves the packed K/V planes
    for t in ("qk", "qv", "sk", "sv"):
        setattr(dst, t, torch.zeros((3, 64, 128), dtype=torch.half))
        setattr(ring, t, torch.zeros((3, 64, 128), dtype=torch.half))
    ring.qshape_k = None
    ring.qshape_v = None
    dst.qshape_k = ring.qshape_k
    dst.qshape_v = ring.qshape_v
    ring.pooled[0, :2] = 1.0
    ring.qk[0, :2] = 2.0
    dst.copy_page(ring, 0, 2, 8)  # block aligned: 8 tokens = 2 groups
    assert torch.equal(dst.pooled[2, :2], ring.pooled[0, :2])
    assert dst.pooled[2, 2:].abs().sum() == 0
    assert torch.equal(dst.qk[2, :2], ring.qk[0, :2])
    # misaligned copy must raise (control: proves the comparison is live)
    try:
        dst.copy_page(ring, 0, 1, 3)
        raise AssertionError("misaligned ring copy must raise")
    except RuntimeError:
        pass


check("qsa ring plane layout storage delta", qsa_ring_layout)
check("qsa ring copy_page semantics", qsa_ring_copy)


# --- 2. GDN rewind job descriptors ---------------------------------------------

class _Job:
    def __init__(self, *args):
        self.args = args


class _ModuleStub:
    conv_kernel_size = 4


def _state(conv=True):
    st = gdn.GDNLayerState.__new__(gdn.GDNLayerState)
    st.module = _ModuleStub()
    return st


def rewind_conv_default():
    # ext.ConvRewindJob / StateRewindJob must be constructible for the test
    ext.ConvRewindJob = _ModuleStub() if False else _Job
    ext.StateRewindJob = _Job
    st = _state()
    st.conv_state = torch.zeros(2, 5, 16)  # [slots, channels, padded history]
    st.recurrent_state = torch.zeros(2, 4, 3, 2, 2)
    st.module.conv_kernel_size = 4
    # last_history 0 -> None (kernel must not run)
    assert st.rewind_conv_job(0, 0, 3) is None
    # num_tokens = 3, history length 12: window [12-3-4, 12-3) copied to [0, 4)
    job = st.rewind_conv_job(0, 12, 3)
    src, dst, dim, cdim, stride = job.args
    cs = st.conv_state
    es = cs.element_size()
    base = cs.data_ptr() + 0 * cs.stride(0) * es
    expect_src = base + (16 - 3 - 4) * cs.stride(2) * es
    assert src == expect_src and dst == base
    assert dim == 5 and cdim == 4 and stride == cs.stride(1)


def rewind_conv_flag_on():
    os.environ["EXL3_HOST_GAP_REWIND"] = "1"
    import importlib
    importlib.reload(gdn)
    st = _state()
    st.conv_state = torch.zeros(2, 5, 16)
    job = st.rewind_conv_job(0, 12, 3)
    # flag-on branch: identical numbers to the default arithmetic for valid indices
    cs = st.conv_state
    es = cs.element_size()
    base = cs.data_ptr()
    expect_src = base + (16 - 3 - 4) * cs.stride(2) * es
    assert job.args[0] == expect_src and job.args[1] == base
    os.environ.pop("EXL3_HOST_GAP_REWIND")
    importlib.reload(gdn)


def rewind_state_words_fp32():
    st = _state()
    st.recurrent_state = torch.zeros(2, 8, 3, 2, 2)  # fp32: 24 elements per token state
    job = st.rewind_state_job(0, 7, 3)
    src, dst, n = job.args
    rs = st.recurrent_state
    base = rs.data_ptr()
    assert src == base + (7 + 1 - 3) * rs.stride(1) * rs.element_size()
    assert dst == base
    # fp32: words == elements
    assert n == rs.stride(1) == 12


def rewind_state_words_bf16():
    st = _state()
    st.recurrent_state = torch.zeros(2, 8, 3, 2, 2, dtype=torch.bfloat16)
    job = st.rewind_state_job(0, 7, 3)
    src, dst, n = job.args
    rs = st.recurrent_state
    # bf16: the kernel copies float4 vectors, so the descriptor must carry 4-byte
    # words (numel * 2 / 4), not elements -- upstream's element count would
    # double-copy past the end of a bf16 plane
    assert n == rs.stride(1) * rs.element_size() // 4 == rs.stride(1) // 2


def _state_words(t, numel):
    return numel * t.element_size() // 4


def rewind_state_flag_on():
    os.environ["EXL3_HOST_GAP_REWIND"] = "1"
    import importlib
    importlib.reload(gdn)
    st = _state()
    st.recurrent_state = torch.zeros(2, 8, 3, 2, 2)
    job = st.rewind_state_job(0, 7, 3)
    rs = st.recurrent_state
    base = rs.data_ptr()
    assert job.args[0] == base + (7 + 1 - 3) * rs.stride(1) * rs.element_size()
    assert job.args[1] == base
    assert job.args[2] == _state_words(rs, rs.numel() // rs.shape[0] // rs.shape[1])
    # zero-advance rewind returns no job
    assert st.rewind_state_job(0, 7, 0) is None
    os.environ.pop("EXL3_HOST_GAP_REWIND")
    importlib.reload(gdn)


def _state_words(t, numel):
    return numel * t.element_size() // 4


check("gdn rewind_conv_job default (upstream) arithmetic", rewind_conv_default)
check("gdn rewind_conv_job flag-on branch equals default arithmetic", rewind_conv_flag_on)
check("gdn rewind_state_job fp32 word count", rewind_state_words_fp32)
check("gdn rewind bf16 state descriptor uses 4-byte words", rewind_state_words_bf16)
check("gdn rewind_state_job flag-on branch", rewind_state_flag_on)


# --- 3. NVMe tier namespace contract ---------------------------------------------

def nvme_identity_core():
    base = {k: "1" for k in (
        "EXL3_HOST_GAP_REWIND", "EXL3_HC_MIX_V2", "EXL3_HC_MIX_V2_MIN_R",
        "EXL3_LS_PREFILL_PIPELINE", "EXL3_MOE_COOP_V2", "EXL3_SHARED_EXPERT_OVERLAP",
        "EXL3_DRAFT_PINNED_STAGING", "EXL3_BATCH_VERIFY", "EXL3_MTP_HEAD_N",
        "EXL3_MOE_PREFILL_E3", "EXL3_HC_MIX_V2_INT8", "EXL3_MTP_DEVICE_DRAFT",
        "EXL3_EMBED_GPU", "EXL3_EMBED_GPU_PRUNED", "EXL3_MOE_PREFILL_E3_DET",
        "EXL3_GDN_BA_WARP1", "EXL3_HC_APPLY_WARP1", "EXL3_GR_STATE_REGRID",
        "EXL3_QSA_RAWK_RING", "EXL3_GDN_STATE_BF16",
    )}
    env1 = dict(base)
    e1 = dc.engine_identity(environ=env1)
    # every served flag enters the namespace
    env2 = dict(base)
    env2.pop("EXL3_QSA_RAWK_RING")
    assert dc.engine_identity(environ=env2) != e1
    # the tier knobs are excluded
    env3 = dict(base)
    env3["EXL3_NVME_TIER"] = "/somewhere"
    env3["EXL3_NVME_TIER_GB"] = "16"
    assert dc.engine_identity(environ=env3) == e1


check("nvme tier identity: flags in, tier knobs out", nvme_identity_core)

print()
n_ok = sum(1 for _, ok, _ in RESULTS if ok)
print(f"== {n_ok}/{len(RESULTS)} cache/rewind tests passed ==")
if n_ok != len(RESULTS):
    sys.exit(1)
