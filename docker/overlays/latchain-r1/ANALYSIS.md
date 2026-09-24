# latchain r1: latency-bound small-kernel chains in Flash-Next decode

Written 2026-09-24 by the latchain Opus sub-agent; saved by the operator from its final report (the harness refused the agent's .md write). Queued as R712 ([`scripts/r712-latchain-r1.sh`](../../../scripts/r712-latchain-r1.sh)).

Scope: three chains on stack-r2 (slotfix-r1 + hcfast-r1 + moefast-r1, `tabbyapi:stack-r2`, 27-key EXTRA_ENV): the QSA sparse-attention chain, the GDN conv/recurrent/norm chain, and the MoE router. Every lever has its own flag and is off by default. Each is designed to be bitwise identical to the served path. The GPU parity test, the model-level logits hashes, the P1 sequence hashes and fn_greedy are what prove it. Nothing here was compiled or run on a GPU. The compile gate is `cpu-build-check.sh` / `Dockerfile.box`. Every speed number below is either measured from the R680 traces (source given) or labelled as an estimate.

## 1. Per-kernel table (measured)

Source: the R680 nsys traces `traces/trace-{c1d3,c4d3,c8d1}.sqlite` (node mode, 2026-09-24). Capture windows come from `out-roofline/tmp/win-*.json`: 32 steps, where the `draft` phase is the served shape and the `control` phase is the d0 cell. They were processed by `out-latchain/tools/chain_kernels.py`, which reuses `out-roofline/tools/gap.py`.
- "med" is the median kernel duration.
- "per step" sums both cards.
- The median gap between graph nodes inside a graph is 0.38–0.45 µs.
- Step wall is the sum over the two cards: the layer split leaves both-busy ≈ 0 (GAP-TABLE §1), so 1 µs off either card is about 1 µs off the step.

Served untraced ms/iterate, for scale (CONTEXT, R653): c1d3 12.53, c4d3 19.84, c8d1 20.75.

### QSA attention segment, one full-attention layer (card 0, median µs)

| kernel (graph node order) | c1d3 | c8d1 | branch under EXL3_LC_QSA_FORK |
|---|---:|---:|---|
| qkv mgemm (6,1,26) | 21.98 | 22.98 | main |
| deinterleave_qg | 0.99 | 1.02 | main |
| rope (q/k, fused norm) | 2.69 | 2.66 | main |
| quant_cache_paged | 1.60 | 1.66 | main |
| index projection exl3_gemv (20) / exl3_gemm | 10.50 | 12.10 | side |
| _qsa_stage | 1.09 | 1.09 | side |
| _qsa_raw_ring_append | 0.90 | – | side |
| _qsa_pool_update_ring_bc | 2.30 | 2.59 | side |
| rope (indexer q) | 1.50 | 1.60 | side |
| _dsa_indexer_fewq | 2.62 | 3.36 | side |
| dsa_topk | 3.97 | 4.16 | side |
| _dsa_pool_expand | 0.70 | – | side |
| _qsa_sparse_split (8,17) / (32,10) | 22.21 | 50.56 | main (after join) |
| _paged_attn_decode_combine (8) / (32) | 7.55 | 5.15 | main |
| mul_sigmoid gate / o_proj / hc_apply | 0.77 / 18.27 / 0.93 | – | main |

The median attention-segment span is 127 µs at c1d3 and 172 µs at c8d1. At c1d3 the side chain costs about 23.6 µs of kernels plus 8 node gaps. The main q/k/v chain it can overlap costs about 27.3 µs.

Per-step totals (µs/step, both cards, chain_kernels TOTAL; QSA includes the index projection and the MTP layer's attention):

| chain | c1d3 | c4d3 | c8d1 | c1d0 |
|---|---:|---:|---:|---:|
| QSA | 940 | 1,199 | 1,061 | 716 |
| GDN conv/rec/norm/rewind | 733 | 1,334 | 1,224 | 369 |
| ROUTER | 450 | 452 | 433 | 322 |
| GLUE | 75 | 119 | 117 | 16 |

**Sparse split.** At c1d3 and c1d0 it takes 22.2 µs per launch. At c1d0 only 34 CTAs run (2 programs × 17 splits). That works out to about 5.5 µs per dependent BLOCK_N iteration (4 × 32 over split_len 128): per-iteration latency, not bandwidth. At c4d3 and c8d1 it takes 49.5–50.6 µs with 32 × 10 CTAs and 7 iterations.

**Combine.** It is a serial loop over the splits, about 0.35–0.45 µs per split.

### GDN segment, one linear-attention layer (card 0, median µs)

| kernel | c1d3 | c4d3 | c8d1 | c1d0 |
|---|---:|---:|---:|---:|
| qkvz mgemm (20,1,8) | 24.8 | – | 25.86 | – |
| gdn_ba_gemv (served EXL3_GDN_BA_WARP1) | 2.53 | – | 3.97 | – |
| fused_op_3 | 1.25 | 1.79 | 1.63 | 1.38 |
| conv1d_update | 2.88 | 3.74 | 3.94 | 2.37 |
| recurrent _128 | 13.73 (1,48,4) | 26.69 (4,48,1) | 24.29 (8,48,1) | 4.99 (1,48,4) |
| gated_rms_norm | 1.57 | 1.63 | 1.63 | 1.41 |
| out_proj | 17.98 | – | 18.75 | – |

The recurrent kernel runs 36 times per step (27 + 9 at c1d3; 26 + 10 at c4d3 and c8d1). That is 497 µs/step at c1d3, 970 at c4d3, 880 at c8d1 and 180 at c1d0.

### Router (µs)

| kernel | c1d3 | c8d1 | c1d0 |
|---|---:|---:|---:|
| cuBLAS Kernel2 (wmma split-K, grid z 8 / 14) | 4.83 | 4.00 | – |
| splitKreduce | 1.12 | 1.63 | – |
| routing_gemv (R = 1) | – | – | 3.68 |
| routing_std_topk | 2.98 | 3.04 | 2.98 |
| gap before coop_rot (device side) | 4.22 | 6.24 | – |

## 2. Levers (one flag each, default off)

| arm | flag | what changes | why it is bitwise | read |
|---|---|---|---|---|
| RR | `EXL3_LC_GDN_RR=1` | `cuda_recurrent_gated_delta_rule_kernel_128_rr`: the state is carried in registers across the job's tokens (one load of the final state, no per-token re-reads). Token s+1's q/k/v/g/β are prefetched during token s. | Same thread layout, same reduction order, same stores in the same order. Each thread only reads back what it wrote, so the register copy equals the stored value (`lc_state_reload` applies the bf16 store rounding). Every value-producing expression is textually the served one (CPU lint). fp contraction is expected to be identical; the GPU parity test is what proves it. | per launch |
| QF | `EXL3_LC_QSA_FORK=1` | The QSA indexer chain (index projection → stage → raw ring append → pool update → indexer rope → fewq → top-k → expand) is captured on a side stream, as a parallel graph branch next to qkv/rope/cache append, and joined before the sparse split. | Same kernels, same arguments, same host call order, so the same `record_param` sites. The branch GEMM gets the upper half of the xh scratch and the upper half of the tile-lock region. Nothing on the main branch reads what the side branch writes before the join. | at graph capture |
| GF | `EXL3_LC_GDN_FORK=1` | The GDN b/a GEMV is captured on a side branch concurrent with the qkvz projection, joined before fused_op_3. | It reads only x and writes only s.ba, which has no reader before fused_op_3. Same kernel, same arguments. | at graph capture |
| QT | `EXL3_LC_QSA_SPLIT_STAGES=N`, `EXL3_LC_QSA_COMBINE_STAGES=N`, `EXL3_LC_QSA_DIV16=1` | AOT compile options of the sparse split (served num_stages 2) and combine (served 1) kernels, plus 16-divisibility hints on their aligned args. | num_stages only software-pipelines loads, and a divisibility hint only vectorizes aligned loads. num_warps and the tile shapes fix the dot and reduction layouts, and those are unchanged. Triton does not guarantee this, so parity checks every variant bitwise, and P0 may only choose among variants that passed. | at import |

The QT keys are one lever: P0 picks one combination and the gate runs it as one arm.

### Expected gains (all estimates, µs per step, both cards; negative = faster)

| lever | c1d3 | c4d3 | c8d1 | d0 cells | basis |
|---|---:|---:|---:|---:|---|
| RR | −250 … −310 | −200 … −350 | 0 … −150 | ≈ 0 | DRAM bytes are unchanged (one state read, one write per token including history), so the floor is bytes / 1.79 TB/s, × 36 layers. c1d3: 7.5 MB → 4.2 µs vs 13.73 measured; it is latency-bound, so rr should reach 5–7 µs. c4d3: 30 MB → 16.7 µs vs 26.7, so rr reaches 17–21 µs. c8d1: 36 MB → 20.1 µs vs 24.3, little room. |
| QF | −225 … −360 | −200 … −330 | −200 … −320 | −180 … −290 | Overlap = min(side ≈ 24–27 µs, main ≈ 27 µs), minus SM contention (156 + 20 CTAs on 170 SMs), gives 15–24 µs per attention layer. Multiply by 12 target layers plus the MTP layers per step: about 15 at c1d3 and 13.5 at c8d1. |
| GF | −70 … −110 | −90 … −160 | −100 … −160 | −70 … −150 | The b/a GEMV (2.5–4.0 µs) plus one node gap hides under the 25 µs qkvz mgemm, × 36. |
| QT (split) | 0 … −150 | 0 … −270 | 0 … −270 | 0 … −100 | About 5.5 µs per dependent-load iteration. Pipelining the index → page → K/V loads would save 6–10 µs per launch at c1 and 10–20 µs at c4/c8, if Triton's pipeliner handles the indirect loads. It may not, hence the 0 lower bound. |
| QT (combine) | −40 … −55 | −15 … −25 | −15 … −25 | −30 … −45 | With 2 stages, combine drops from 7.5 to 4–5 µs at c1 (17 splits run serially), and from 5.2 to about 4 µs at c8. |
| all (upper bound) | −585 … −985 (4.7–7.9 % of 12.53 ms) | −505 … −1,135 (2.5–5.7 % of 19.84) | −315 … −925 (1.5–4.5 % of 20.75) | | The levers touch disjoint kernels, but QF and RR both shorten card-0 time, so the plain sum is an upper bound. |

Compared with the Track A exposed-over-roofline targets (QSA 705 / 791 µs, GDN 571 / 444, router 362 / 354 at c1d3 / c8d1), r1 addresses about half of QSA, most of the c1 GDN excess, and none of the router.

## 3. Not in r1, and why

- **Router (362 µs exposed at c1d3).** At R > 1 the logits come from `cublasGemmEx`: a cutlass wmma split-K kernel (grid z = 8 at R = 4, 14 at R = 16) followed by splitKreduce.
  - A replacement is bitwise only if it reproduces cuBLAS's split-K partition and its fp32 reduction order. That can only be discovered on the GPU, and it is fragile across cuBLAS versions.
  - A faster top-k alone is worth about 35–58 µs/step (estimate).
  - The top-k → coop_rot gap (4.2 µs on card 0, 9.3 on card 1) is on the device side: the host is about 1.5–1.8 ms ahead, so it is not launch latency. The likely cause is the cross-stream event and the shared-expert branch sharing a hardware queue.
  - **Proposed for r2:**
    - (a) A discovery probe: an mma.sync split-K replica compared bitwise against hgemm at R = 2..24. If it matches, fuse gemm + reduce + top-k. The tie order must be `warp_topk_shared`'s (pos % 32, pos / 32) lexicographic order, not lowest index.
    - (b) A zero-code probe of `CUDA_DEVICE_MAX_CONNECTIONS`.
- **Dependent-chain fusions.** R684 measured about 0: the dependency, not the launch, sets the latency. The forks only remove dependencies that were never real.
- **PDL / L2 prefetch.** R705 measured PDL at about 0 inside graphs, and L2 prefetch cost more than it saved.
- **GDN state rewind.** It costs 38.8 µs but runs only about 0.4 times per step, outside the steady decode step.
- **Glue (75–117 µs).** This is eager torch elementwise work, mostly outside these chains.

## 4. Interaction with out-densegemm

No dense GEMM kernel is modified, but two launches move.

**QF moves the QSA index projection.** It runs `exl3_gemm_gr` on `qsa_qk_proj` on the side branch, with the upper half of xh and `exl3_gemm_set_locks_offset(MAX_TILES_C / 2)`.
- `exl3_gemm.cu` gains a thread-local lock offset: a setter plus one line in `exl3_gemm_gr`.
- If out-densegemm changes how `exl3_gemm_gr` uses the lock buffer (barrier counters, more than MAX_TILES_C / 2 tiles, a different scratch), or which GEMM variant serves the index projection, re-check QF's no-sharing argument.
- The two patches will conflict textually at the `int* locks = ...` line.

**GF moves the GDN b/a GEMV** (`gdn_ba_gemv_gr`, part of the GDN input projection) onto a side branch concurrent with the qkvz mgemm. The kernel is unchanged. A faster qkvz GEMM from out-densegemm shrinks GF's window.

## 5. Risks and which gate step catches them

- **Graph site order.** `Graph::capture_end` matches `record_param` sites to nodes in `cudaGraphGetNodes` order, which must still be creation order across the two capture streams. Cross-stream event record and wait create edges during capture, not nodes. Step 3 of the gate (model-level logits hashes) catches a violation first, then the P1 hashes and fn_greedy. If parity passes but model parity fails, suspect site order rather than a kernel.
- **VRAM.** One stream and two events per device, no allocations: about 0 MiB (estimate). The QT stage counts change shared memory, not VRAM. The gate logs cuda:0 used memory after the ON boot against the ref boot, with a bar of ref + 32 MiB.
- **SM contention.** The side branches compete for SMs with the main branch's GEMMs, which can cost the main branch time. P1 decides whether the net is a gain.
- **The QF xh / lock split.** A TORCH_CHECK at capture time requires the main branch's hadamard scratch (at most max(qkv_num_src, 2) × R × hs) to fit in the lower half. The lock split relies on two facts:
  - exl3_gemm_kernel uses only `locks[0, tiles)` and leaves them at zero;
  - every decode GEMM has far fewer than 512 K tiles.
- **GF capture safety.** `gdn_ba_gemv_gr` launches on `graph->capture_stream` (checked at gdn.cu:2049), so the fork is capture-safe. Every `_gr` call on the QSA branch (exl3_gemm_gr, rope_gr, dsa_topk_gr) does the same.

## 6. Composition with densegemm-r1 (operator note)

densegemm-r1 (R710) rewrites the dense K4 GEMM paths with a per-device workspace whose safety note says: "If later work lets a dense GEMM overlap another, both the served locks and these slots need per-stream copies." QF runs the QSA index projection concurrently with the qkv mgemm. So if both pass their own gates, the combined stack needs a composition review (lock offset + densegemm slots) before one image carries both.
