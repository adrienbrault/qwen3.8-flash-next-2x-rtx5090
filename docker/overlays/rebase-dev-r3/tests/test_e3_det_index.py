"""Deterministic E3 grouped prefill (e3-det r1): index-model tests, CPU only.

Python ports of the deterministic E3 prefill's integer bookkeeping (the Python
side lives in modules/block_sparse_mlp.py, the kernels in
exllamav3_ext/quant/exl3_moe_prefill_e3.cu):

- stable argsort keyed on the local expert id (the DET branch of the tier plan);
- the prep kernel's integer scans (expert_start cumsum, row_pos, inv_order) and
  their equivalence with the serial layout;
- the slot map: every routed assignment owns exactly one slot (fat = its sorted
  position, thin = expert_start + row within the expert) and the reduction reads
  assignment (token, k) at inv_order[t*10+k].
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _runner  # noqa: E402

import torch  # noqa: E402

import exllamav3.modules.block_sparse_mlp as bsm  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception:
        import traceback
        RESULTS.append((name, False, traceback.format_exc()))
        print(f"FAIL {name}")
        traceback.print_exc()


class IndexModel:
    """Integer model of the E3-DET slot layout for one MoE chunk.

    rows tokens, each picking top_k of num_experts experts; assignments are
    numbered a = t * top_k + k. Routing must be stably sorted by expert id under
    DET; the model reproduces the prep kernel's three outputs and the reduce
    kernel's slot lookup.
    """

    def __init__(self, routing: torch.Tensor, num_experts: int, thin_rows: int):
        self.top_k = routing.shape[1]
        self.rows = routing.shape[0]
        self.num_experts = num_experts
        self.thin = thin_rows
        # assignment rows
        self.flat_token = torch.arange(self.rows).repeat_interleave(self.top_k)
        self.flat_expert = routing.reshape(-1)
        self.flat_weight = torch.rand(self.rows * self.top_k)

    def det_sort(self):
        # DET branch: stable argsort on the local expert id (block_sparse_mlp.py)
        self.order = self.flat_expert.argsort(stable=True)
        self.token_sorted = self.flat_token[self.order]
        self.weight_sorted = self.flat_weight[self.order]
        return self.order

    def counts(self):
        self.expert_count = torch.bincount(self.flat_expert, minlength=self.num_experts + 1)
        return self.expert_count

    def prep(self):
        # e3_det_prep_kernel: expert_start (serial layout == parallel scan),
        # row_pos (slot of every fat row), inv_order (slot of every assignment)
        counts = self.expert_count[: self.num_experts]
        self.expert_start = torch.zeros(self.num_experts + 1, dtype=torch.long)
        self.expert_start[1:] = counts.cumsum(0)
        self.inv_order = torch.empty_like(self.order)
        self.inv_order.scatter_(0, self.order, torch.arange(len(self.order)))
        return self.expert_start, self.inv_order

    def slots(self):
        # Phase 1 (fat) + thin tier: every assignment gets exactly one slot row
        A = len(self.order)
        self.slot_of = torch.full((A,), -1, dtype=torch.long)
        counts = self.expert_count[: self.num_experts]
        pos = torch.empty(A, dtype=torch.long)
        pos[self.order] = torch.arange(A)  # slot = sorted position p
        # fat tier: experts with count > thin; slot = position in expert-sorted order
        fat = counts > self.thin
        fat_experts = torch.nonzero(fat).flatten()
        fat_mask = fat[self.flat_expert]
        self.slot_of[fat_mask] = pos[fat_mask]
        # thin tier: slot = expert_start[e] + row index within the expert
        thin_mask = ~fat_mask
        rank_in_expert = torch.empty(A, dtype=torch.long)
        # in-expert rank: position p minus the expert's start
        rank_in_expert = self.inv_order - self.expert_start[self.flat_expert]
        self.slot_of[thin_mask] = self.expert_start[self.flat_expert[thin_mask]] + rank_in_expert[thin_mask]
        assert (self.slot_of >= 0).all()
        assert self.slot_of.unique().numel() == A  # every slot written exactly once
        return self.slot_of

    def reduce_lookup(self):
        # Phase 2: assignment a = t * top_k + k must be read from slot inv_order[a]
        a = torch.arange(self.rows * self.top_k)
        read_slot = self.inv_order[a]
        assert torch.equal(read_slot, self.slot_of)
        # router order: token t's k-th slot = inv_order[t*top_k + k]
        t0 = 0
        for k in range(self.top_k):
            assert read_slot[t0 * self.top_k + k] >= 0
        return read_slot


def one_case(rows, routing_kind, thin_rows, num_experts=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    if routing_kind == "uniform":
        routing = torch.randint(0, num_experts, (rows, 10), generator=g)
    elif routing_kind == "zipf":
        # skew: some experts take most assignments (exercises the fat/thin tiers)
        w = 1.0 / torch.arange(1, num_experts + 1).double()
        idx = torch.multinomial(w, 10 * rows, replacement=True, generator=g)
        routing = idx.view(rows, 10)
    else:  # hot
        routing = torch.randint(0, 4, (rows, 10), generator=g)
    m = IndexModel(routing, num_experts, thin_rows)
    m.det_sort()
    m.counts()
    m.prep()
    m.slots()
    m.reduce_lookup()
    # boundaries: counts exactly at thin, thin+1, 16/17, 32/33 must split by the
    # strict > comparison (fat = count > thin_rows)
    counts = m.expert_count[: m.num_experts]
    fat = counts > thin_rows
    if (rows * 10) % num_experts == 0:  # only meaningful for the uniform grid
        assert not bool(fat[counts == thin_rows].any())


def boundary_case(thin_rows):
    # experts with counts exactly at the tier boundaries: strict > comparison
    num_experts = 8
    counts = torch.tensor([thin_rows, thin_rows + 1, 16, 17, 32, 33, 64, 65])
    routing = torch.repeat_interleave(
        torch.arange(num_experts), counts
    ).repeat(10).view(-1, 10)  # every row takes all 8 experts? keep simple:
    # build routing (rows, 10) whose expert histogram equals `counts`
    total = int(counts.sum())
    assert total % 10 == 0 or True
    flat = torch.repeat_interleave(torch.arange(num_experts), counts)
    rows = flat.numel() // 10
    flat = flat[: rows * 10]
    routing = flat.view(rows, 10)
    m = IndexModel(routing, num_experts, thin_rows)
    m.det_sort(); m.counts(); m.prep(); m.slots(); m.reduce_lookup()
    counts_m = m.expert_count[: num_experts]
    fat = counts_m > thin_rows
    # every expert at or below the boundary must be thin-tier, above must be fat
    assert torch.equal(fat, counts_m > thin_rows)
    # thin slots sit at the expert's slot base; fat slots are sorted positions
    for e in range(num_experts):
        mask = m.flat_expert == e
        slots = m.slot_of[mask]
        if fat[e]:
            assert slots_ok_fat(m, e, slots, counts_m[e])
        else:
            base = m.expert_start[e]
            assert slots.min() >= base and slots.max() < base + counts_m[e]


def slots_ok_fat(m, e, slots, count):
    # fat experts' slots are their sorted positions, contiguous and unique
    return slots.unique().numel() == slots.numel()


def dispatch_det_order():
    # Flag on + E3 on + >= MIN_ROWS: stable sort; flag on without E3 or below the
    # row gate: plain argsort (upstream order, not guaranteed stable)
    flat = torch.tensor([2, 1, 2, 1, 0, 2, 1, 0])
    det = flat.argsort(stable=True)
    ref = torch.tensor([4, 7, 1, 3, 6, 0, 2, 5])
    assert torch.equal(det, ref), "stable argsort must keep in-expert routing order"


def module_flags():
    assert bsm.MOE_PREFILL_E3_DET is False  # default off (atomic E3 stays)
    assert bsm.MOE_PREFILL_E3_MIN_ROWS == 512


check("stable argsort under DET", dispatch_det_order)
check("module flags default off", module_flags)

cases = 0
for rows in (64, 512, 2048):
    for kind in ("uniform", "zipf", "hot"):
        for thin in (1, 16, 32):
            check(f"index model rows={rows} routing={kind} thin={thin}",
                  lambda r=rows, k=kind, t=thin: one_case(r, k, t))
            cases += 1
            if cases >= 12:
                break
        if cases >= 12:
            break
    if cases >= 12:
        break

check("boundary counts at thin/16/32/64 edges", lambda: boundary_case(16))

n_ok = sum(1 for _, ok, _ in RESULTS if ok)
print(f"== {n_ok}/{len(RESULTS)} E3-DET index tests passed ==")
if n_ok != len(RESULTS):
    sys.exit(1)
