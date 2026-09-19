# R527: tensor parallelism across the two cards would make decode at most 1.07–1.08× faster; not built

Results directory on the serving host: `results/2026-09-19-r527-tp-bound`. Raw records and the probe source: [`2026-09-19-r527-tp-bound/`](2026-09-19-r527-tp-bound/). Driver: [`scripts/r527-tp-bound.sh`](../../scripts/r527-tp-bound.sh). Date: 2026-09-19.

The served stack splits layers across the two cards, so they take turns. ExLlamaV3 raises `NotImplementedError` for tensor parallelism on `qwen4_exp`; building it would be several rounds of work. This probe bounds the gain before anything is built: it measures how much of each kernel family's time one card keeps when the family's work is halved, and what an all-reduce over the P2P link costs at the served row counts. It then projects a TP decode step from the [R519](r519-profile-2p50.md) kernel budget of the served configuration. The go threshold was 1.25× at 1 stream, draft depth 3. The probe was written for this repository by an Opus agent round and runs in the served image (`tabbyapi:mtp-pruned-r1-tc1-plefix`) with a copy of the served tune cache.

## Measured inputs

P2P between the cards: 28.2 GB/s each way; copies exact; NCCL uses P2P on all four channels.

All-reduce of one row set of the trunk width (fp32 × 2560), per call:

| rows | NCCL eager | NCCL in a CUDA graph | peer copy + add |
| --- | --- | --- | --- |
| 4 (1 stream, draft 3) | 8.55 µs | 12.57 µs | 57.12 µs |
| 16 (4 streams, draft 3) | 18.58 µs | 23.92 µs | 57.61 µs |

A TP decode step needs 96 all-reduces in the trunk plus 6 in the draft: 1.26 ms per step at 1 stream and 2.37 ms at 4, with the graph timings the served path would use.

MoE block, experts and shared expert split in 128-channel units (rank 0 holds 384 of 640 channels, rank 1 holds 256; 0.60 is the structural floor). `s` is the slower half's time over the full block's time:

| layer (expert K) | 1 row | 4 rows | 8 rows | 16 rows |
| --- | --- | --- | --- | --- |
| layer 24 (K=2) | 0.887 | 0.756 | 0.813 | 0.699 |
| layer 0 (K=3) | 0.847 | 0.740 | 0.808 | 0.703 |
| MTP layer | 0.883 | 0.740 | | |

GDN at 1×4 keeps 0.731 of its time, attention 0.870; at 4×4, 0.741 and 0.835.

## Projection

| | 1 stream, draft 3 | 4 streams, draft 3 |
| --- | --- | --- |
| R519 step (profiler mode) | 14.99 ms | 23.62 ms |
| kernels: served → TP | 12.34 → 10.17 ms | 19.19 → 15.16 ms |
| projected TP step, head replicated | 14.30 ms, 1.049× | 22.23 ms, 1.063× |
| projected TP step, vocabulary-sharded head | 13.95 ms, **1.074×** | 21.95 ms, **1.076×** |
| with the hyper-connection mixer sharded (+2 exchanges per site) | 0.913× | 0.906× |
| measured `s`, all-reduce free | 1.150× | 1.190× |
| structural split (GDN, attention, MTP 0.5; MoE 0.6; the rest replicated), measured all-reduce | 1.206× | 1.197× |
| `s` = 0.5 everywhere, all-reduce free | 1.619× | 1.633× |

The host-side time between kernels (2.65 ms at 1 stream, 4.43 ms at 4) is kept as it is; that is an assumption, and TP adds host work rather than removing it. The GDN rewind is assumed to halve with the heads.

Even with a free all-reduce the projection stays under 1.25×. Halved kernels keep 70–89 % of their time at these row counts, and the replicated parts (hyper-connection mixer, head, router: 2.9 ms of the 12.3 ms kernel time at 1 stream) do not shrink. Tensor parallelism is closed for this model on these two cards.

Two earlier tries of the unit stopped in the probe: try 1 on a CUDA context error after a MoE family was unloaded, try 2 because it looked for a K=2 layer only among layers 0–4 and 43–47. On this pack K=2 is layers 12–36; layers 0–11 and 37–47 are K=3.
