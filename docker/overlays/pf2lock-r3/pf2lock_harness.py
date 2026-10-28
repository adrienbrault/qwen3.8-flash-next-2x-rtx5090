"""
pf2lock test harness shared by cpu_stress.py and pf2lock_tests.py (CPU only, no GPU, no network).

load_module(path)    the n-gram module under test: the installed exllamav3.modules.ngram_embedding (path None), or
                     the file at `path` loaded AS exllamav3.modules.ngram_embedding. Inside the image the real
                     exllamav3 package resolves its relative imports; where exllamav3 is not installed (the Mac),
                     minimal stand-ins are registered for the five names the module imports (Module, Config,
                     DiskTensorHandle, exllamav3_ext, ngram_codec's ROW_DIM/words_per_row) and STUBBED is True.
install_shims(ne)    test-side shims, never a module edit: pageable allocations instead of pinned ones (pin_memory
                     needs a CUDA context) and a Python stand-in for ext.ngram_hash_cpu (ngram_size 2,
                     heads_per_ngram 1: the row id is the position's token id; unique sorted ids + inverse map, as
                     the native hash returns). The r2 module allocates through ne._pinned_empty; the served and r1
                     modules allocate inside _PinSet.grow, which is then replaced by an equivalent pageable grow.
make_module(ne)      a fresh NGramEmbedding on one fp16 RAM shard whose row r holds the value r in every column, so
                     any wrong row is visible; expected(history) is the reference output.
"""
import importlib
import importlib.util
import os
import sys
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch  # noqa: E402

ROWS = 1000
STUBBED = False
STORE = None
_ALLOC_HOOK = None      # tests: called with (shape, dtype) before every pageable allocation (may raise)


def _install_stubs():
    global STUBBED

    def mod(name, **kw):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(kw)
        sys.modules[name] = m
        return m

    class Module:
        def __init__(self, config, key, qmap):
            self.config, self.key, self.qmap = config, key, qmap
            self.device = None
            self.caps = {}

    class Config:
        pass

    class DiskTensorHandle:
        pass

    row_dim = 160

    def words_per_row(K):
        return 1 + row_dim * K // 16

    def unavailable(*_a, **_k):
        raise NotImplementedError("stubbed exllamav3: codec not available")

    mod("exllamav3")
    mod("exllamav3.model")
    mod("exllamav3.model.config", Config = Config)
    mod("exllamav3.loader")
    mod("exllamav3.loader.safetensors", DiskTensorHandle = DiskTensorHandle)
    mod("exllamav3.ext", exllamav3_ext = types.SimpleNamespace())
    mod("exllamav3.modules", Module = Module)
    mod("exllamav3.modules.quant")
    mod("exllamav3.modules.quant.exl3_lib")
    mod("exllamav3.modules.quant.exl3_lib.ngram_codec", ROW_DIM = row_dim, words_per_row = words_per_row,
        mul1_codebook = unavailable, dequant_rows = unavailable)
    STUBBED = True


def load_module(path = None):
    if path is None:
        return importlib.import_module("exllamav3.modules.ngram_embedding")
    try:
        importlib.import_module("exllamav3.modules")
    except ImportError:
        _install_stubs()
    name = "exllamav3.modules.ngram_embedding"
    spec = importlib.util.spec_from_file_location(name, path)
    ne = importlib.util.module_from_spec(spec)
    sys.modules[name] = ne
    spec.loader.exec_module(ne)
    return ne


def _pageable(shape, dtype):
    if _ALLOC_HOOK is not None:
        _ALLOC_HOOK(shape, dtype)
    return torch.empty(shape, dtype = dtype)


def install_shims(ne, grow_sleep_us: int = 0):
    if hasattr(ne, "_pinned_empty"):
        def alloc(shape, dtype):
            t = _pageable(shape, dtype)
            if grow_sleep_us:
                time.sleep(grow_sleep_us / 1e6)
            return t
        ne._pinned_empty = alloc
    else:
        def grow(self, n, row_words, row_dtype):
            if self.uids is None or self.uids.numel() < n or self.packed.shape[1] != row_words \
                    or self.packed.dtype != row_dtype:
                self.uids = _pageable(n, torch.int64)
                self.inverse = _pageable(n, torch.int64)
                self.heads = _pageable(n, torch.int32)
                self.packed = _pageable((n, row_words), row_dtype)
            if grow_sleep_us:
                time.sleep(grow_sleep_us / 1e6)
        ne._PinSet.grow = grow

    real_ext = ne.ext

    def ngram_hash_cpu(history, out_len, layer_multipliers, head_offsets, head_vocab_sizes,
                       heads_per_ngram, eos_token_id, uids, inverse, heads):
        ids = history[:, history.shape[1] - out_len:].reshape(-1)
        u, inv = torch.unique(ids, sorted = True, return_inverse = True)
        U = u.numel()
        uids[:U] = u
        inverse[:ids.numel()] = inv
        heads[:U] = 0
        return U

    ne.ext = types.SimpleNamespace(ngram_hash_cpu = ngram_hash_cpu, _real = real_ext)


def make_module(ne):
    global STORE
    if STORE is None:
        STORE = torch.arange(ROWS, dtype = torch.float16).view(ROWS, 1).expand(ROWS, ne.ROW_DIM).contiguous()
    m = ne.NGramEmbedding(None, "stress.ple", ngram_size = 2, heads_per_ngram = 1, ple_embed_dim = ne.ROW_DIM,
                          eos_token_id = ROWS + 7)
    m.mode = "fp16_ram"
    m.tables = [STORE]
    m.handles = None
    m.rows_per_shard = ROWS
    m.num_rows = ROWS
    m._row_dtype = torch.float16
    m.device = torch.device("cpu")
    return m


def expected(history):
    ids = history[:, 1:]
    return STORE[ids.reshape(-1)].view(history.shape[0], history.shape[1] - 1, STORE.shape[1])


def h(*ids):
    return torch.tensor([list(ids)], dtype = torch.int64)


def shutdown(m):
    """Stop a module's worker without blocking on a set a failed iteration may have stranded."""
    ex = getattr(m, "_executor", None)
    m._executor = None
    if ex is not None:
        ex.shutdown(wait = False, cancel_futures = True)
