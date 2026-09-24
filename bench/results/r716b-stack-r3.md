# R716b, R716c: stack-r3 (hcfast r2, latchain, densegemm r2, moefast r3), promoted; bitwise-identical to stack-r2 at every served decode shape

Results directories on the serving host: `results/2026-09-24-r716b-stack-r3`, `results/2026-09-24-r716c-identity-shapes`, `results/2026-09-24-r716c-b2` and `results/2026-09-24-promote-stack-r3`. Raw records of R716b: [`2026-09-24-r716b-stack-r3/`](2026-09-24-r716b-stack-r3/) (the per-run P1 directories, boot and container logs stay on the host); R716c's records are not copied here. Drivers [`scripts/r716b-stack-r3.sh`](../../scripts/r716b-stack-r3.sh) and [`scripts/r716c-identity-shapes.sh`](../../scripts/r716c-identity-shapes.sh). Image `tabbyapi:stack-r3` = `tabbyapi:stack-r2` plus the patch series in [`docker/overlays/stack-r3`](../../docker/overlays/stack-r3/), built with INCLUDE `mf3 dg2` and empty `DGV2_NVCC_DEFS` (series SHA-256 `fe779eac…`). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 999,424-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft component on the second GPU, draft policy `[[4, 3], [5, 2], [8, 1]]`.

This is the second promotion under the stack-track rule in [`docs/PROMOTION.md`](../../docs/PROMOTION.md#the-stack-track-since-2026-09-24).

## The change

One patch series on `tabbyapi:stack-r2`, applied at fuzz 0 with each patch's SHA-256 pinned in [`series/SERIES`](../../docker/overlays/stack-r3/series/SERIES), and one rebuild of the extension. The build applied patches 01, 02, 03, 05 and 06 ([`build.txt`](2026-09-24-r716b-stack-r3/build.txt)); the served SASS of `stack-r2` is unchanged and 100 kernels were added, each within its served reference's stack frame plus 64 B. The served environment changes these families (the union, U):

| component | keys | admitted by |
| --- | --- | --- |
| HC2, hcfast r2 | `EXL3_HC_MIX_V3=2` with the tables `DOTS_B=1:1,4:2,32:4 UP_B=1:1,8:4,32:8 DOTS_J=1:4,32:8 DOTS_PF=1:1,32:0 UP_Q=1:4,8:2,32:4 PDL=0` (replacing hcfast r1's `=1 DOTS_B=2 UP_B=8`) | [R702](r702-hcfast-r2.md) |
| RR, latchain | `EXL3_LC_GDN_RR=1` | [R712](r712-latchain-r1.md) |
| QT, latchain | `EXL3_LC_QSA_SPLIT_STAGES=2 EXL3_LC_QSA_COMBINE_STAGES=1 EXL3_LC_QSA_DIV16=1` | R712, candidate |
| QF, latchain | `EXL3_LC_QSA_FORK=1` | R712, candidate |
| DG, densegemm r2 | `EXL3_DENSE_V2=1` | [R714](r714-densegemm-r2.md) |
| MF3, moefast r3 | `EXL3_MOE_COOP_V3=3 EXL3_MOE_COOP_V3_MAP=2-4:2 EXL3_SHARED_EXPERT_EARLY=1` (replacing `EXL3_MOE_COOP_V3=2`) | [R713](r713-moefast-r3.md) |

QF and DG=1 share an image only with the series' dense-lcguard patch: a dense call on latchain's QSA side branch is detected by its lock offset and uses its own gemv counters and gemm scratch ([`ANALYSIS.md`](../../docker/overlays/stack-r3/ANALYSIS.md) §3). The launcher's default environment grows from 27 to 39 keys.

## What was measured

R716b ran 2026-09-24 17:16 to 19:28 UTC ([`audit.txt`](2026-09-24-r716b-stack-r3/audit.txt)); R716c and its rerun R716c-b2 ran after it, ending 20:05 UTC.

**Kernel and model parity (R716b).** hcfast, latchain, moefast and densegemm kernel parity passed on the built image (`parity-*.txt`). Logits-level model parity (`lc_model_parity`, in process, 4,096 tokens of context) at batch 1 depth 3, batch 4 depth 3 and batch 8 depth 1: forward digests and tokens identical for OFF, U and OFF2 (96 of 96, 96 of 96, 48 of 48 forwards; `mp-*-compare.txt`). The lcguard wrote its line on both devices at every shape, so the side-branch path ran.

**P1, in-process harness (R716b).** 7 rounds, arms OFF, U, U without each optional component (UnoQT, UnoQF, UnoDG, UnoMF3) and OFF2, rotated per round. Sequence hashes of U, every leave-one-out arm and OFF2 equal OFF in 42 of 42 cells each ([`p1-summary.txt`](2026-09-24-r716b-stack-r3/p1-summary.txt)). U − OFF, ms per iterate, stall-excluded (iterates slower than the run median plus 8 ms dropped from both runs of a pair):

| shape | U − OFF | draft depth 0 cell |
| --- | --- | --- |
| 1 stream, depth 3 | −1.58 (−13.1 %), 7 of 7 | 1 row: −0.70 (−9.0 %), 7 of 7 |
| 4 streams, depth 3 | −1.71 (−8.5 %), 7 of 7 | 4 rows: −1.16 (−11.6 %), 7 of 7 |
| 8 streams, depth 1 | −1.53 (−8.2 %), 7 of 7 | 8 rows: −1.47 (−12.3 %), 7 of 7 |

OFF2 − OFF is flat at every shape. Each leave-one-out marginal (U − Uno&lt;X&gt;) has no same-sign regression and gains in 7 of 7 rounds at: QT −1.2 % at 1 stream and −1.4 % at 8 streams; QF −2.2 % at 1 stream; DG −4.1 % at 1 stream and −2.6 % at 8 streams; MF3 −3.8 % at 1 stream and −2.2 % at 8 streams.

**Served A/B (R716b).** Boots of the served launcher, `NVME_TIER=` on every boot: A = the served configuration (`tabbyapi:stack-r2`, 27 keys), B = `tabbyapi:stack-r3` with the union (39 keys), C = `tabbyapi:stack-r3` with the served environment (flags off). Order A1, B1, A1b (an extra A/A boot, because B1's multi-stream greedy differed from A1), B1 again, A2, B2, A3, B3, C. Every boot ran a 1-to-8-stream decode ramp, the greedy identity set (6 prompts, one of about 100,000 tokens) and multi-stream greedy at 2, 4 and 8 streams (384 tokens per stream); the A and B boots of the three pairs then ran [`bench/fn_gate.sh`](../fn_gate.sh) with 3 recorded rounds: leg A is prose at about 4,000 tokens of context at 1, 4 and 8 streams, leg B is prose at about 26,000 tokens of context at 4 streams, 1,024 forced tokens, greedy. Values are the median per-request decode rate in tokens/s per stream, A → B ([`served.txt`](2026-09-24-r716b-stack-r3/served.txt)):

| cell | pair 1 | pair 2 | pair 3 | mean B/A |
| --- | --- | --- | --- | ---: |
| leg A, prose, 1 stream | 232.8 → 261.5 | 215.2 → 250.4 | 327.0 → 379.2 | 1.149 |
| leg A, prose, 4 streams | 137.2 → 148.2 | 143.3 → 157.3 | 150.8 → 163.7 | 1.088 |
| leg A, prose, 8 streams | 92.1 → 100.8 | 91.9 → 100.1 | 91.7 → 98.7 | 1.087 |
| leg B, prose, 26k context, 4 streams | 136.5 → 150.3 | 133.1 → 147.0 | 137.1 → 151.6 | 1.104 |

The mean over the four cells is 1.107. The 1-stream spread across pairs follows the pairs' prompts, whose tokens per frame are 2.66, 2.46 and 3.71 in both arms; within a pair the prompts are byte-identical.

| check | A (stack-r2) | B (stack-r3, union) |
| --- | --- | --- |
| greedy set against A1 | identical, 6 of 6, every boot | identical, 6 of 6, every boot; C identical |
| free VRAM at the UP line, GPU 0 / GPU 1 | 1,041 / 2,431 MiB | 1,033 / 2,421 MiB |
| free VRAM after the gate, minimum, GPU 0 / GPU 1 | 139 / 1,659 MiB | 121 / 1,631 MiB |
| out-of-memory lines, TORCH_CHECK, tracebacks | 0, 0, 0 | 0, 0, 0 |
| decode ramp, leg B rows | 36 of 36, 12 of 12 | 36 of 36, 12 of 12 |
| lcguard lines | none (no guard in the image) | devices 0 and 1, every boot |

**Multi-stream greedy (R716b)** ([`gstreams-compare.txt`](2026-09-24-r716b-stack-r3/gstreams-compare.txt), texts in `gstreams.jsonl`). At 2 streams all 8 boots are byte-identical. At 4 and 8 streams the served configuration differs from itself: over the 6 pairs of A boots, 4.83 of 8 streams (60 %) and 12.17 of 16 (76 %) diverge, because which requests share a verify batch depends on arrival timing and near-tied tokens flip. B pairs: 6.00 and 12.33; A-B pairs: 4.75 and 13.17. The gate's rule judged each candidate boot against the largest of three A/A divergences from one reference boot; B3 at 4 streams (7 against 6), B2 and C at 8 streams (15 against 13) exceeded it by one stream each, and the unit printed `DECISION UNION: REJECT`.

**The rule's false-alarm rate.** An independent review of the round recomputed the statistics. Over all 280 assignments of the 8 boots to roles (reference, three A/A, four candidates) the rule fails 179 (64 %), and on the four A boots alone it fails the served configuration against itself in 5 of 12 assignments. A pooled label-permutation test (mean between-group minus mean within-group divergence over all splits) shows no effect at 4 streams (−0.47, 28 of 35 splits at or above) and a one-stream shift at 8 streams (+0.94, 3 of 35) that the flags-off boot C shows more strongly (+1.83), so it does not come from the flags.

**R716c: identity at every served decode shape.** The served policy verifies at 4, 8, 12, 16, 15, 12, 14 and 16 rows at 1 to 8 active jobs, and R716b's model parity covered only 4 and 16. R716c ran `lc_model_parity` in process with four arms: OFF, U and OFF2 on `tabbyapi:stack-r3`, and R2 = `tabbyapi:stack-r2` with the served environment, at batch 1 depth 3, 2 depth 3, 3 depth 3, 4 depth 3, 5 depth 2, 6 depth 1, 7 depth 1 and 8 depth 1. Every forward digest and every token was identical in all four arms at all 8 shapes (96 of 96, 72 of 72 or 48 of 48 forwards per arm). Batch 2 depth 3 ran out of memory on GPU 0 at the harness's default `--gpu-split 30,30` in all four arms, `stack-r2` included, and passed at `--gpu-split 28,30` (R716c-b2). The draft passes of batches 1 to 8 cover draft widths 1 to 8, and the R2 arm shows the `stack-r3` image with the served environment computing the same logits as `stack-r2`.

## Verdict

Promoted 2026-09-24 20:10 UTC (22:10 CEST): the launcher's `DAILY_IMG` is `tabbyapi:stack-r3` and its default environment carries 39 keys. The promotion boot read 41 EXL3 keys in the container (the image adds `EXL3_DECODE_FUSE=0` and `EXL3_DECODE_OVERLAP=0`), free VRAM 1,033 / 2,421 MiB at the UP line, the lcguard line on both devices, and greedy output identical to R716b's reference on 6 of 6 prompts (results `2026-09-24-promote-stack-r3`). Rollback is `IMG=tabbyapi:stack-r2` with R701's 27-key `EXTRA_ENV`.

The unit's REJECT came from the multi-stream greedy rule alone; every other bar passed (P1 identity and gain, served ratios, headroom, safety, greedy set). The rule is replaced (`tools/greedy_streams.py --compare` in the overlay): at a concurrency where the A/A boots are identical the check stays strict; above 25 % A/A divergence it is INCONCLUSIVE and identity must come from in-process parity at every row count the draft policy reaches, as R716c did; in between, a pooled permutation test decides. On R716b's data it reads strict PASS at 2 streams and INCONCLUSIVE at 4 and 8. [`docs/PROMOTION.md`](../../docs/PROMOTION.md#identity-at-every-served-shape-since-2026-09-24) carries the rule.

Open, not blocking: every in-process identity cell is a fixed batch composition at 4,096 tokens of context; the densegemm and hcfast kernel parity row lists skip 5 to 7, 9 to 11 and 13 to 15 rows (densegemm) and 10, 11 and 14 (hcfast), which R716c covers only at model level. The gate's defects that the review listed (B1's scored container log overwritten after the A1b reboot, no fn_gate on C, C pooled as a candidate instead of a control) are fixed in the next gate that reuses the script.
