"""Generator-shaped call schedules for the QSA raw-key ring (shared by the CPU interpreter tests and the GPU harness).

Every schedule drives two copies of the indexer planes with the same staged raw keys:

- ``ref``: the served layout (full ``raw_k`` plane, ``PAGE_SIZE`` rows per page) updated by the served kernels in the
  served order (``_mla_plane_update_kernel`` then ``_qsa_pool_update_kernel``; the eager path and the BC graph run
  the same pair).
- ``ring``: the ring layout (``ring`` rows per page) updated by the patch's kernels, either as the BC graph runs them
  (ring append, then ``_qsa_pool_update_ring_bc_kernel``) or as the eager path does (``qsa_ring_plane_update``: ring
  pool first, then ring append).

After every call the two pooled planes must be ``torch.equal``. ``K`` supplies the kernels (real Triton on a GPU, or
the CPU interpreter of the tests) and ``K.ctx()`` the device context for launches.
"""
from __future__ import annotations

import random

import torch


PAGE_SIZE = 256
P = 4
D = 128


class Planes:
    def __init__(self, num_pages: int, rows: int, device, d: int = D):
        self.raw = torch.zeros((num_pages, rows, d), dtype = torch.half, device = device)
        self.pooled = torch.zeros((num_pages, PAGE_SIZE // P, d), dtype = torch.half, device = device)


class World:
    """Two plane sets (served / ring) over one physical page pool, plus per-sequence block tables."""

    def __init__(self, K, device, num_pages: int, ring: int, seed: int, rope_r: int = 64, d: int = D,
                 attn_factor: float = 1.0, eps: float = 1e-6):
        self.K = K
        self.dev = device
        self.ring = ring
        self.d = d
        self.rope_r = rope_r
        self.attn_factor = attn_factor
        self.eps = eps
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.ref = Planes(num_pages, PAGE_SIZE, device, d)
        self.rg = Planes(num_pages, ring, device, d)
        self.k_norm_w = (torch.randn(d, generator = self.gen) * 0.1).half().to(device)
        self.inv_freq = (1.0 / (10000.0 ** (torch.arange(0, rope_r, 2, dtype = torch.float32) / rope_r))).to(device)
        self.free = list(range(num_pages))
        self.rng.shuffle(self.free)
        self.tables = {}
        self.calls = 0
        self.mismatch = None
        self.log = []

    def new_seq(self, name: str, pages: int, share_from: tuple | None = None):
        """Allocate a sequence; share_from = (other_name, n_pages) aliases the other's first pages (prefix reuse)."""
        table = []
        if share_from is not None:
            other, n = share_from
            table += self.tables[other][:n]
        while len(table) < pages:
            table.append(self.free.pop())
        self.tables[name] = table

    def _bt(self, names):
        width = max(len(self.tables[n]) for n in names)
        rows = [self.tables[n] + [0] * (width - len(self.tables[n])) for n in names]
        return torch.tensor(rows, dtype = torch.int32, device = self.dev)

    def call(self, names, pos0s, length: int, path: str):
        """One cached forward of `length` tokens for each named sequence at its pos0 (bsz = len(names))."""
        assert path in ("bc", "eager")
        bsz = len(names)
        kraw = (torch.randn((bsz, length, self.d), generator = self.gen) * 2.0).half().to(self.dev)
        bt = self._bt(names)
        seqlens = torch.tensor(pos0s, dtype = torch.int32, device = self.dev)
        npr = bt.shape[1]
        K = self.K
        with K.ctx(self.dev):
            # served: the same kernel pair in the eager path and the BC graph
            K.mla[(bsz * length,)](
                kraw, self.ref.raw, bt, seqlens, npr, length,
                page_size = PAGE_SIZE, D = self.d, DST_D = 0, DST_OFF = 0,
            )
            K.pool[(bsz, length // P + 1)](
                self.ref.raw.view(-1, self.d), self.ref.pooled.view(-1, self.d), self.k_norm_w,
                self.inv_freq, bt, seqlens, npr, length,
                page_size = PAGE_SIZE, P = P, D = self.d, ROPE_R = self.rope_r,
                attn_factor = float(self.attn_factor), eps = float(self.eps), MAXPOOLS = 1,
            )
            if path == "bc":
                K.ring_append[(bsz * length,)](
                    kraw, self.rg.raw, bt, seqlens, npr, length,
                    page_size = PAGE_SIZE, D = self.d, RING = self.ring,
                )
                K.ring_bc_pool[(bsz, length // P + 1)](
                    self.rg.raw.view(-1, self.d), self.rg.pooled.view(-1, self.d), self.k_norm_w,
                    self.inv_freq, bt, seqlens, npr, length,
                    page_size = PAGE_SIZE, P = P, D = self.d, ROPE_R = self.rope_r,
                    attn_factor = float(self.attn_factor), eps = float(self.eps),
                    MAXPOOLS = length // P + 1, RING = self.ring,
                )
            else:
                K.ring_plane_update(
                    kraw, self.rg.raw, self.rg.pooled, self.k_norm_w, self.inv_freq,
                    self.attn_factor, self.eps, bt, seqlens, P,
                )
        self.calls += 1
        self.log.append((tuple(names), tuple(pos0s), length, path))
        if self.mismatch is None and not torch.equal(self.ref.pooled, self.rg.pooled):
            self.mismatch = self.calls - 1

    def result(self, name: str) -> dict:
        return {
            "name": name,
            "calls": self.calls,
            "pooled_equal": self.mismatch is None,
            "first_mismatch_call": self.mismatch,
            "first_mismatch": None if self.mismatch is None else self.log[self.mismatch],
        }


def _path(length: int, bsz: int = 1) -> str:
    # decode_flash_attn: the graph takes bsz <= 8 and q_len <= 16, everything else runs eager
    return "bc" if (bsz <= 8 and length <= 16) else "eager"


def prefill(w: World, name: str, start: int, end: int, chunk: int):
    """Job.prefill: chunk ends are page-aligned except the last (end = len - 1)."""
    pos = start
    while pos < end:
        stop = min((pos + chunk) // PAGE_SIZE * PAGE_SIZE, end)
        if stop <= pos:
            stop = end
        w.call([name], [pos], stop - pos, _path(stop - pos))
        pos = stop
    return pos


def mtp_rounds(w: World, target: str, draft: str | None, kv: int, rounds: int, window: int,
               eager_verify: bool = False) -> int:
    """MTP decode: draft steps (1 token each at kv + idx) into the draft planes, one verify window of
    window + 1 tokens at kv into the target planes, accept a in [1, window + 1], the draft accept
    prefill rewrites kv + 1 .. kv + a - 1, next round at kv + a."""
    for _ in range(rounds):
        if draft is not None:
            for idx in range(window):
                w.call([draft], [kv + idx], 1, "bc")
        length = window + 1
        w.call([target], [kv], length, "eager" if eager_verify else _path(length))
        a = w.rng.randint(1, length)
        if draft is not None and a > 1:
            w.call([draft], [kv + 1], a - 1, _path(a - 1))
        kv += a
    return kv


def rewind_replay(w: World, name: str, kv: int, offset: int, chunk: int) -> int:
    """Job.rewind_checkpoint on a recurrent model: restore the page-aligned stash at or before the target and
    replay prefill up to the target."""
    target = kv - offset
    replay_from = target // PAGE_SIZE * PAGE_SIZE
    prefill(w, name, replay_from, target, chunk)
    return target


# ---- schedules -----------------------------------------------------------------------------------------------------

def s_prefill_decode(K, dev, ring, seed = 1, chunk = 512, length = 1203, steps = 40):
    w = World(K, dev, 16, ring, seed)
    w.new_seq("a", 8)
    kv = prefill(w, "a", 0, length, chunk)
    for _ in range(steps):
        w.call(["a"], [kv], 1, "bc")
        kv += 1
    # the same continuation through the eager path (EXL3_BC_ATTN=0 / declined graph)
    for _ in range(9):
        w.call(["a"], [kv], 1, "eager")
        kv += 1
    return w.result("prefill_chunks_then_decode")


def s_mtp(K, dev, ring, seed = 2, rounds = 60, window = 3):
    w = World(K, dev, 24, ring, seed)
    w.new_seq("t", 8)
    w.new_seq("d", 8)
    kv = prefill(w, "t", 0, 301, 256)
    prefill(w, "d", 0, 301, 256)
    mtp_rounds(w, "t", "d", kv, rounds, window)
    return w.result(f"mtp_verify_rewind_depth{window}")


def s_mtp_max_graph(K, dev, ring, seed = 3, rounds = 40):
    # the largest graph call: q_len 16 (MAX_QLEN), random acceptance down to 1, crossing page boundaries
    w = World(K, dev, 24, ring, seed)
    w.new_seq("t", 10)
    w.new_seq("d", 10)
    kv = prefill(w, "t", 0, 239, 512)
    prefill(w, "d", 0, 239, 512)
    kv = mtp_rounds(w, "t", "d", kv, rounds, 15)
    # an abandoned window re-run at the same start (zero advance)
    w.call(["t"], [kv], 16, "bc")
    w.call(["t"], [kv], 16, "bc")
    w.call(["t"], [kv + 1], 1, "bc")
    return w.result("mtp_verify_q16_graph")


def s_eager_verify_limit(K, dev, ring, seed = 4, rounds = 40):
    # eager verify windows at the guard limit (ring - P + 1 tokens), rewound by random acceptance
    w = World(K, dev, 24, ring, seed)
    w.new_seq("t", 10)
    kv = prefill(w, "t", 0, 402, 512)
    mtp_rounds(w, "t", None, kv, rounds, ring - P, eager_verify = True)
    return w.result(f"eager_verify_q{ring - P + 1}")


def s_replay(K, dev, ring, seed = 5):
    w = World(K, dev, 24, ring, seed)
    w.new_seq("a", 10)
    kv = prefill(w, "a", 0, 777, 512)
    kv = mtp_rounds(w, "a", None, kv, 30, 3)
    # stop-string rewind across a page boundary, replayed from the stash page; then decode again
    kv = rewind_replay(w, "a", kv, 300, 512)
    kv = mtp_rounds(w, "a", None, kv, 20, 3)
    # a rewind whose replay tail is a short (graph-sized) chunk
    kv = kv // PAGE_SIZE * PAGE_SIZE + 7 + PAGE_SIZE
    w.call(["a"], [kv - PAGE_SIZE - 7], PAGE_SIZE + 7, "eager")
    kv = rewind_replay(w, "a", kv, 2, 512)
    for _ in range(12):
        w.call(["a"], [kv], 1, "bc")
        kv += 1
    return w.result("rewind_replay")


def s_batched(K, dev, ring, seed = 6):
    # bsz > 1: eager batches with different pos0 per row, graph batches, per-row acceptance
    w = World(K, dev, 32, ring, seed)
    names = ["a", "b", "c"]
    kvs = []
    for i, n in enumerate(names):
        w.new_seq(n, 8)
        kvs.append(prefill(w, n, 0, 257 + 41 * i + i, 512))
    w.call(names, kvs, 5, "eager")
    kvs = [k + 5 for k in kvs]
    w.call(names, kvs, 20, "eager")
    kvs = [k + 20 for k in kvs]
    for _ in range(25):
        w.call(names, kvs, 4, "bc")
        kvs = [k + w.rng.randint(1, 4) for k in kvs]
    for _ in range(6):
        w.call(names, kvs, 1, "bc")
        kvs = [k + 1 for k in kvs]
    return w.result("batched_bsz3")


def s_prefix_share(K, dev, ring, seed = 7):
    # B reuses A's two full pages (page hash hit) and continues on its own pages; A keeps decoding
    w = World(K, dev, 24, ring, seed)
    w.new_seq("a", 6)
    kva = prefill(w, "a", 0, 700, 512)
    w.new_seq("b", 6, share_from = ("a", 2))
    kvb = prefill(w, "b", 2 * PAGE_SIZE, 2 * PAGE_SIZE + 131, 512)
    for _ in range(20):
        w.call(["a"], [kva], 4, "bc")
        kva += w.rng.randint(1, 4)
        w.call(["b"], [kvb], 4, "bc")
        kvb += w.rng.randint(1, 4)
    w.call(["a", "b"], [kva, kvb], 3, "bc")
    return w.result("prefix_share_full_pages")


def s_midblock_chunks(K, dev, ring, seed = 8):
    # chunk boundaries inside a block (injections, prefill after a mid-page kv): the next chunk reads the
    # partial block from the ring
    w = World(K, dev, 16, ring, seed)
    w.new_seq("a", 8)
    pos = 0
    for n in (1030, 300, 3, 1, 250, 17, 29):
        path = _path(n)
        w.call(["a"], [pos], n, path)
        pos += n
    return w.result("mid_block_chunk_starts")


def s_no_pass(K, dev, ring, seed = 9):
    # full-width rotary (no pass-through segment): the PASS == 0 branch of the pool kernels
    w = World(K, dev, 12, ring, seed, rope_r = D)
    w.new_seq("a", 6)
    kv = prefill(w, "a", 0, 450, 256)
    mtp_rounds(w, "a", None, kv, 20, 3)
    return w.result("rope_full_width")


POSITIVE = [
    s_prefill_decode, s_mtp, s_mtp_max_graph, s_eager_verify_limit, s_replay, s_batched, s_prefix_share,
    s_midblock_chunks, s_no_pass,
]


# ---- negative controls (must mismatch: they prove the comparison detects a ring that loses rows) ------------------

def n_small_ring_graph(K, dev, ring, seed = 11):
    # a ring smaller than one graph call plus the partial block: 8 rows under q_len 16
    w = World(K, dev, 16, 8, seed)
    w.new_seq("a", 6)
    kv = prefill(w, "a", 0, 301, 256)
    mtp_rounds(w, "a", None, kv, 10, 15)
    return w.result("control_ring8_q16_graph")


def n_eager_window_over_limit(K, dev, ring, seed = 12):
    # an eager verify window past the guard (ring + 4 tokens) rewound to one accepted token
    w = World(K, dev, 16, ring, seed)
    w.new_seq("a", 6)
    kv = prefill(w, "a", 0, 302, 256)            # kv % 4 == 2
    w.call(["a"], [kv], ring + 4, "eager")
    w.call(["a"], [kv + 1], 1, "bc")             # needs rows kv - 2 .. kv
    return w.result("control_eager_window_over_limit")


NEGATIVE = [n_small_ring_graph, n_eager_window_over_limit]
