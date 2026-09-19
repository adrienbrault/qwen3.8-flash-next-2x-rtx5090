# R579: the windowed draft cache is served, page pool 999,424 (+3.4 %), and a boot-time decode ramp is what got it there

Results directory on the serving host: `results/2026-09-19-r579-promote-mtp-kv-window`. Raw records: [`2026-09-19-r579-promote-mtp-kv-window/`](2026-09-19-r579-promote-mtp-kv-window/). Driver: [`scripts/r579-promote-mtp-kv-window.sh`](../../scripts/r579-promote-mtp-kv-window.sh). Overlay: [`docker/overlays/mtp-kv-window-r2`](../../docker/overlays/mtp-kv-window-r2).

Served since 2026-09-20 00:29:20 CEST: image `tabbyapi:mtpwin-r2`, `EXL3_MTP_KV_WINDOW=16384` in `EXTRA_ENV`, `cache_size` 966,656 → **999,424**. Slots, KV bits, layer split, chunk and draft policy are unchanged. Rollback: `launch-flashnext.sh.pre-r579`.

## What changed since R575

[R575](r575-promote-mtp-kv-window.md) promoted the same overlay at 1,015,808 and rolled back: after about 1,090 GSM8K requests at 5 concurrent the engine raised `GPU assert: out of memory exllamav3_ext/graph.cu 51`. CUDA graphs are captured per batch size at first use, nothing in that round had decoded at 5 jobs, and the capture landed with 901 MiB free on the first card. The boot-headroom rule only covers what is resident at boot.

This round changed two things:

- one pool step lower, 999,424 instead of 1,015,808;
- **every boot runs a c1 → c8 decode ramp before anything else**, one request per batch size, so each decode graph is captured while the card is otherwise idle. The container log is then checked for `graph.cu` lines. The ramp ran on all four A/B boots and on the gate boot: `graph OOM lines: 0` every time.

## The draft cache and where the memory goes

The MTP draft layer no longer keeps a full-length KV cache; each slot gets a sink page plus 64 pages. That frees draft cache on the first card, and the loader then moves one transformer layer there. The move is worth more than the memory it frees, because the boundary layer 23 is a full-attention layer and takes its share of the page pool with it:

| | free VRAM at boot, MiB | free after the c1 → c8 ramp, MiB |
| --- | --- | --- |
| served before (966,656) | 2,085 / 867 | 1,293 / 347 |
| candidate (999,424) | 1,041 / 2,531 | 251 / 2,013 |

The imbalance does not disappear, it inverts: the second card gains 1,664 MiB and the first becomes the card that bounds the pool.

## A/B, four boots A B B A, tier off

| | served | candidate |
| --- | --- | --- |
| decode, code, 1 stream | 206.0 | 206.8 (**+0.26 %**, 95 % CI −2.66 to +3.22) |
| decode, code, 4 streams | 126.5 | 129.0 (**+1.99 %**, 95 % CI −0.37 to +4.74) |
| decode, prose, 1 stream | 199.6 | 199.5 (−0.10 %) |
| decode, prose, 4 streams | 119.6 | 119.4 (−0.19 %) |
| cold prefill, 45k tokens | 9,535 / 9,553 | 9,486 / 9,159 |
| cold prefill, 90k tokens | 10,202 / 10,072 | 10,505 / 10,718 |

24 paired prompts per shape, per-request decode rate, geometric mean. Decode is flat to slightly positive and prefill is within the run-to-run spread; the round is a pool change, not a speed change.

The one-stream greedy fingerprint changes from `f4add302e176d78e` to `18238d63065ee16c`, because a layer sits on the other card and the reduction order changes with it. The 30k fingerprint `4a255910dee2d9c5` is unchanged. Both are stable across every boot of each arm.

## Gates on the promoted launcher, NVMe tier on

| gate | result |
| --- | --- |
| boot, tier on, c1 → c8 ramp | pool 999,424; free 961 / 2,453 after the ramp; no `graph.cu` line |
| crash-restart restore | 29,952 tokens served from the tier in 0.567 s after a restart against 3.967 s cold, same fingerprint |
| agentic edit, greedy and sampled, 1 and 5 streams | 6/6 each |
| needles, 131k and 240k prompt tokens | 5/5 and 5/5 |
| tool-eval-bench, 69 × 4 | 85.8 |
| GSM8K 5-shot, n=500, no stop strings, 5 concurrent | 0.976 |

The gate that failed in R575 is the one that passed here, on the same overlay, one pool step lower and with every graph captured first.
