# R814 to R820, R818p: a served process decodes 3 % slower per step at 1 and 2 streams until two new row counts have run through the routed-MoE decode-class path; the launcher now sends three short prompts at boot to put it in the faster state

Results on the serving host, all on 2026-09-30: `2026-09-30-r814-trigger-test` (R814, 02:20 to 02:28 UTC), `2026-09-30-r815-replay-test` (R815, 03:05 to 03:21 UTC), `2026-09-30-r816-staged-trigger` (R816, 04:14 to 04:23 UTC), `2026-09-30-r817-fast-state-rule-0531` (R817 try 2, 05:32 to 05:39 UTC; try 1 in `2026-09-30-r817-fast-state-rule`, 05:21 to 05:29 UTC, is void), `2026-09-30-r818-warmfast-gate` (R818, 06:18 to 07:44 UTC), `2026-09-30b-r818-warmfast-gate` and `2026-09-30c-r818-warmfast-gate` (R818b/c, 07:56 to 08:21 UTC), `2026-09-30-r819-fast-state-bisect` (R819, 07:44 to 07:51 UTC), `2026-09-30-r820-stream-queues-0834` (R820, 08:35 to 08:47 UTC) and `2026-09-30-r818p-promote-warmfast-1152` (R818p, 11:52 to 11:55 UTC). The drivers, the probe overlay and the raw records of these rounds are not in this repository; the served launcher is [`scripts/launch-flashnext.sh`](../../scripts/launch-flashnext.sh), a copy of it is [`scripts/launchers/launch-flashnext-r818-warmfast.sh`](../../scripts/launchers/launch-flashnext-r818-warmfast.sh). Boot logs, readbacks and launcher copies stay on the host.

## Configuration

Served image `tabbyapi:merge-tok-r1` (`sha256:ac16920f72cf`) in every round. SERVED or NEW is the [R809p](r803-r810-prefill-merge.md) launcher (host md5 `6429dfa2…`, 41 keys, `EXL3_PREFILL_MERGE=1`, `EXL3_STASH_ASYNC=1`); KNOBSOFF is the same launcher without those two keys (39 keys); WARM is the R818 launcher (host md5 `f3975d68…`), which is SERVED plus the boot warm-up and nothing else. 8 slots, 901,120-token pool, split `[30, 30]`, draft policy `[[4, 3], [8, 2]]`, NVMe tier off in the measurement rounds; power limits 600 / 575 W, core offset 0, memory offset +4500. Every boot is fresh. `fn` is `fn_bench` at 1 stream on the code prompt (118 tokens), greedy, 1,024 forced tokens, one warm-up request and three recorded requests, reported as the median time per decode step; `fn0` is the value right after boot, `fn1`, `fn2` after each stimulus. The slow level is 10.10 to 10.15 ms per step across all R817 boots, the fast level 9.80 to 9.88 ms.

## Where R813 left it

[R811 and R812b](r811-r813-std-bench-merge.md) found that at 1 and 2 streams the median decode step of a process without `EXL3_PREFILL_MERGE` falls by 2.1 to 3.6 % at one request (ShareGPT request 45, 786 tokens; Spec-Bench request 29, 788 tokens) and stays lower, and that a merge-on process never falls. R814 sent one uncached 786-token prompt of request 45's shape (a nonce-prefixed cut of its text) between two `fn` runs, on four fresh boots: `fn` after / before 0.998 and 1.002 (KNOBSOFF), 1.000 and 0.996 (NEW), SM clock 2,925 to 2,947 MHz and memory clock 16,051 MHz flat. The prompt alone did not reproduce the fall, and the session had no positive control.

## R815: the fall is a state the process keeps

One fresh boot per arm: `fn` ×3, R811's ShareGPT 1-stream cell replayed exactly (400 requests, the same client, sample and order), `fn` ×3; clocks sampled every 200 ms.

| arm | cell ITL p50, requests 46-399 / 0-44 | `fn` after / before | server steps per second |
| --- | ---: | ---: | ---: |
| KNOBSOFF | 0.9739 (10.171 → 9.905 ms) | 0.9712 (10.119 → 9.827 ms) | 97.6 → 100.5 |
| NEW | 0.998 | 1.005 | — |

The faster step is already present in request 45's first 100 decode frames (9.94 against 10.20 ms before), so the switch happens between the end of request 44 and request 45's first decode step: request 45's admission and its split prefill (768 rows, then a 17-row forward). On the 324 requests whose output is byte-identical in both arms, the after / before ratio is 0.998 (NEW) against 0.974 (KNOBSOFF). The state persists after the cell: `fn` run afterwards is 2.9 % faster. SM clock 2,917 to 2,947 MHz, memory 16,051 MHz.

## R816: two short forwards are needed

Three fresh boots with a timing probe mounted into the engine; `fn` between stages.

| boot | stages (`fn` ms per step, × `fn0`) |
| --- | --- |
| A, KNOBSOFF | `fn0` 10.104 → request 45 alone → 10.140 (1.0035) → requests 0-44 → **9.848 (0.9746)** → request 45 (768 cached) → 9.858 (0.9756) |
| B, KNOBSOFF | `fn0` 10.096 → requests 0-44 → 10.162 (1.0065) → request 45 → **9.906 (0.9811)** |
| C, NEW, glibc malloc trim off and mmap thresholds pinned | `fn0` 10.124 → requests 0-44 → 10.137 → request 45 → 10.124 |

- The switch needs both request 40 (its prefill ends in a separate 27-row forward at offset 768) and request 45 (17 rows at 768), in either order. Boot A switches inside request 40's prefill, boot B at request 45's.
- The saving is on the GPU side: the blocking wait after each verify is 0.24 / 0.25 ms shorter, out of 0.28 / 0.26 ms per step. Host issue time, draft, the gap between steps, the PLE lookup, page faults (0 per step) and the CUDA caching allocator's counters do not change.
- Boot C excludes host memory allocation as the cause; the flat gap between steps excludes TabbyAPI.

## R817: two short prompts at boot switch the merge-on process

Try 1 is void: the client required `usage` in the response and TabbyAPI returns `"usage": null` on non-streamed `/v1/completions` as well as on non-streamed chat completions. Its in-process `torch.profiler` arm is void as well: CUPTI moves time between host issue and the GPU wait from the first profiled window on, and the two profiled windows read equal. Try 2:

| boot | plan | `fn` × `fn0` after each stage |
| --- | --- | --- |
| M, served merge-on | `fn0` 10.150 → prompts of 20 and 29 tokens (19 and 28 rows at offset 0) → 2,201 tokens → 2,050 + 2,070 tokens | **0.9695**, 0.9660, 0.9665; no merged prefill ran |
| K1, KNOBSOFF | 786 tokens (17 rows at 768) → 786, another nonce (17 at 768) → 796 (27 at 768) → control | 1.0025, 1.0110, **0.9757**, 0.9759 |
| K2, KNOBSOFF | 20 + 29 tokens → 258 → 788 → control | **0.9724**, 0.9729, 0.9731, 0.9737 |

The offset of the short forward does not matter (K2), a repeated row count does nothing (K1's second 17-row forward), and forwards above 32 rows do not count. The rule the round review drew from this, two new row classes of the hyper-connection mixer, was falsified by R819. `fn_bench` records carry no text, so R817 does not show output identity; R818 does.

## R819: the routed-MoE `run_bszN` path carries the switch

Served merge-on launcher, fresh boots.

| boot | change | stimulus | `fn1` × `fn0` |
| --- | --- | --- | ---: |
| C | none | 20 + 29 tokens (19, 28 rows) | 0.9665, switched |
| HC | mixer at 9 to 32 rows through the tiled prefill kernel | 20 + 29 | 0.9727, switched |
| CLASS | none | 2 + 11 tokens (1, 10 rows) | 0.9731, switched |
| ALL | every ≤ 32-row decode path off | 20 + 29, then 33 + 12 | 0.9960 / 1.0065, not switched |
| MOE | `MAX_BSZN` 32 → 16 (`EXL3_MOE_COOP_ROWS32=0`) | 20 + 29, then 33 + 12 | 1.0017 / 0.9995, not switched |

`fn0` is 10.114 to 10.160 across the boots, and the output text is identical in 15 of 15 phases. With `MAX_BSZN` at 16, the 19- and 28-row forwards leave `BC_BlockSparseMLP::run_bszN` (the routed-expert cooperative launch plus the shared expert started early on its side stream) and take the generic path; only the 11-row forward is a new count through `run_bszN` in that boot. The mixer created the same row buckets in MOE and did not switch, so the mixer-class rule does not hold.

## R820: two new `run_bszN` row counts, and the state is not in ExLlamaV3's code

Served merge-on launcher, fresh boots, stimulus 20 + 29 tokens unless noted.

| boot | change | `fn0` ms | `fn1` ms | reading |
| --- | --- | ---: | ---: | --- |
| C | none | 10.102 | 9.835 | ×0.9736, switched |
| Q32 | `CUDA_DEVICE_MAX_CONNECTIONS=32` | 10.114 | 9.822 | starts slow and switches: aliasing of streams onto hardware connections is not the slow state |
| Q1 | `CUDA_DEVICE_MAX_CONNECTIONS=1` | 10.292 | 9.840 | +1.9 % slower start, the same fast level |
| MOECLASS | `MAX_BSZN` 16, stimulus 2 + 11 tokens | 10.117 | 9.840 | switched: two new `run_bszN` row counts, not the halved scratch geometry |
| SIDE | `EXL3_SHARED_EXPERT_OVERLAP=0`, `EXL3_SHARED_EXPERT_EARLY=0`, `EXL3_MOE_SHARED_COOP=0` | 10.736 | 10.693 | no switch; confounded, because this arm also runs without the side-stream overlap, which is worth 6.3 % by itself |

- Every fast state lands at 9.82 to 9.84 ms per step, whatever the start, the connection count or the `MAX_BSZN` geometry.
- In the served code the shared expert runs eagerly, outside a CUDA graph, exactly at the first call at a new row count, on the per-layer side stream; later calls at that count replay a captured graph. The events that count toward the switch are those eager launches at new counts. A code search found no state that these calls leave behind and that the 4-row decode verify reads: graph slots, workspaces, autotune keys, kernel attribute caches, counters and events are per row count, rewritten every call, or already filled at boot. The persistent state is below the CUDA API, in how the driver or the hardware schedules the side stream against the main stream; it was not traced further.
- C's ShareGPT 8-stream cell ran after C had switched and read the slower of the two 8-stream modes (509.7 against Q32's 529.4 output tok/s, one boot each). The 8-stream two-mode effect below is a separate effect.

## R818: the warm-up launcher gate

The candidate launcher sends, after the existing `Warmup.` request, three raw `/v1/completions` requests of 20, 29 and 11 prompt tokens (synthetic `Note NNNNN:` word lists, temperature 0, `max_tokens` 8), and reads the server's own prompt counts from the container log: each must be uncached and inside its window. A failed check is logged as `FASTWARM FAILED` and never blocks serving. 12 fresh boots, WARM and SERVED alternating (ABBA), tier off.

- `FASTWARM ok` in 6 of 6 WARM boots, none of the three prompts cached.
- Output: `fn_greedy` 6 of 6 and `chat_greedy` 6 of 6 identical to R809's reference in both 1-stream boots per arm; `chat_greedy` run in the slow state (SERVED, before any short prompt) identical to the fast state's, 6 of 6 per pass; ShareGPT 1-stream outputs 400 of 400 byte-identical between SERVED and WARM in both passes, the 158 merged prompts included. At 2 and 8 streams outputs differ between arms as much as between two boots of one arm (batch-composition near-ties), which is not an identity test.

Time per decode step, WARM / SERVED, `fn_bench` median, both passes pooled:

| streams | code | prose |
| ---: | ---: | ---: |
| 1 | 0.9684 | 0.9658 |
| 2 | 0.9767 | 0.9724 |
| 4 | 0.9999 | 1.0080 |
| 8 | 1.0005 | 1.0037 |

ShareGPT V3 (R811's cell), output tok/s, mean of passes A and B:

| streams | WARM | SERVED | WARM / SERVED |
| ---: | ---: | ---: | ---: |
| 1 | 230.8 | 224.9 | 1.026 |
| 2 | 332.9 | 317.6 | 1.048 |
| 8 | 510.7 | 516.8 | 0.988 |

- At 2 streams the output gain is larger than the step gain: TPOT −4.4 %, ITL p50 −3.8 % and TTFT −5.4 % against `fn` −2.3 to −2.8 %. The difference is not explained. The step effect is ×0.966 to ×0.968 at 1 stream and ×0.972 to ×0.977 at 2.
- The unit's 8-way warm-up before each 8-stream cell (R811's protocol) puts SERVED in the fast state too, so that cell compares fast with fast. SERVED also switches during the 8-stream burst of the persistence check (`fn` after / before 0.9656). A served process reaches the fast state on its own at its first episode of 4 or more concurrent streams; the warm-up covers the time from a restart to that episode, and traffic that stays at 1 or 2 streams.
- Persistence: WARM stays fast after merged prefills, a 29,071-token prompt with its stash, 65 s idle and an 8-stream burst (×0.9668 of SERVED before the burst).
- Free VRAM at the UP line: WARM 1,099 / 1,667 MiB, SERVED 1,181 / 1,759 MiB, in 12 of 12 boots. WARM's UP line is read after the three extra forwards. After traffic the arms differ by 6 / 8 MiB (2-stream cells), 6 / 4 MiB (8-stream cells) and 0 / 0 MiB (after the 29k and 120k prompts of the 1-stream boots). Pool 901,120 in every boot, 0 out-of-memory lines.
- The pre-registered rule printed HEADROOM (82 / 92 MiB below SERVED at boot, past the 32 MiB tolerance); with headroom set aside the next clause was REGRESSION on ShareGPT at 8 streams (cost 1.0119 against a 1.01 bar), with SERVED's A/B spread there at 3.25 %.

**R818b/c** (3 more fresh boots per arm, ShareGPT at 8 streams only): WARM 512.0, 529.8, 511.4 and SERVED 507.6, 528.5, 529.5 output tok/s. Over five boots per arm, R818's included: WARM 514.9, SERVED 519.8, WARM / SERVED 0.9906 (cost 1.0095, inside the 1.01 bar). The boots fall into two modes, about 510 and about 529 tok/s (ITL p50 23.9 and 23.3 ms, 3.5 % apart); WARM was in the faster mode in 1 of 5 boots, SERVED in 3 of 5. Which mode a boot lands in is not set by the warm-up (R820 C) and is not explained.

## R818p: promotion

The operator accepted the headroom difference (82 / 92 MiB less free at boot, within 8 MiB after traffic, pool unchanged). The R818 launcher went live at 11:52 UTC. Checks on the served boot: `FASTWARM ok` (20, 29 and 11 tokens, none cached); `fn` 9.797 ms per step (three requests 9.797, 9.718, 10.091; the gate was ≤ 9.95, the slow level 10.10 to 10.15); free VRAM 1,099 / 1,667 MiB; image, 41 keys, pool 901,120 and knob readbacks unchanged; `fn_greedy` 6 of 6 and `chat_greedy` 6 of 6 identical to R809's reference. Rollback: `FASTWARM=0`, or the R809 launcher.

## What this means for the published figures

- The README's decode curve ([R813](r811-r813-std-bench-merge.md), 2026-09-30 01:17 to 01:34 UTC) ran 1 to 8 streams in ascending order on processes without the warm-up. Its 1- and 2-stream points were measured before any stimulus of the kind above, in the slow state. R818 measured the fast state's step at ×0.968 (1 stream) and ×0.972 to ×0.977 (2 streams) on the same prompts; at 4 and 8 streams the step is the same in both states.
- The standard benchmark's 1- and 2-stream cells ([R811](r811-r813-std-bench-merge.md), served image arm) never switched; R818 measured +2.6 % and +4.8 % output tok/s on ShareGPT at 1 and 2 streams with the warm-up. Spec-Bench was not re-measured. At 4 and 8 streams the step does not differ between the states, so those cells are unaffected; in R818 the control arm of each 8-stream cell had switched during the unit's 8-way warm-up before the cell.
- The boot free-VRAM figure is now read after the warm-up. A later headroom gate compares against 1,099 / 1,667 MiB.
