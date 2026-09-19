# R548: bf16 GDN recurrent state is served on top of the raw-key ring; the page pool grows to 1,032,192 tokens

Results directory on the serving host: `results/2026-09-19-r548-promote-gdnbf16-ring`. Raw records: [`2026-09-19-r548-promote-gdnbf16-ring/`](2026-09-19-r548-promote-gdnbf16-ring/). Driver: [`scripts/r548-promote-gdnbf16-ring.sh`](../../scripts/r548-promote-gdnbf16-ring.sh). Decode probe: [`bench/mp_decode.py`](../mp_decode.py). Overlay: [`docker/overlays/gdn-state-bf16-r1/`](../../docker/overlays/gdn-state-bf16-r1/). Date: 2026-09-19, 14:32–15:18 UTC.

## What changes

The 36 Gated-DeltaNet layers keep a recurrent state per slot and per draft position (the history for MTP rewinds). With `EXL3_GDN_STATE_BF16=1` that state is stored in bf16 instead of fp32; the recurrence math stays fp32, and the state is rounded once per step on store. At 4 slots and draft depth 3 this frees 864 MiB, which the page pool takes. The CUDA kernels gain a state-type template parameter, so the image rebuilds the extension. The launcher changes four lines: the image (`tabbyapi:nvme-tier-r4-e3det-r6-rawk-gdnbf16`, the [R546](r546-promote-rawk.md) image plus the overlay), the same name in the NVMe tier default, `EXL3_GDN_STATE_BF16=1`, and `cache_size` 1,032,192.

bf16 state is not bit-exact: the c1 greedy fingerprint changes from `e7fb377c987d685c` to `f4add302e176d78e`; the 30k fingerprint `4a255910dee2d9c5` stays. Earlier measurements of the same overlay on the R540 image, results directories named:

- `2026-09-19-r543-gdn-bf16`: kernel harness on both cards; output cosine ≥ 0.99998 against fp32 state after a 32k chunked prefill; GSM8K n=500 0.972 at 868,352 tokens.
- `2026-09-19-r547-mp-decode-bf16`: the first promotion attempt (results `2026-09-19-r545-promote-gdnbf16`) read −13 % code decode at 1 stream from one prompt; R547's 48 paired prompts over 4 boots gave code c1 +3.0 % [−0.3, +6.2], prose c1 +1.4 % [−0.3, +3.0], with a per-prompt range of −11 to +22 %. A single prompt cannot rank two arms whose greedy text differs, so this promotion reads decode from `mp_decode.py` (24 code and 24 prose prompts, paired by prompt).
- `2026-09-19-r548-promote-gdnbf16-ring-try2`: bf16 at 868,352 tokens booted with 1,953 / 637 MiB free against the served 1,973 / 737 and failed GSM8K after tool-eval on `GPU assert: out of memory .../exllamav3_ext/graph.cu 51`: a CUDA graph captured late had no room on cuda:1. Since then a pool step is accepted only when each card's free VRAM at boot stays within 32 MiB of the served boot's.

## Ladder (NVMe tier off)

The served launcher at 983,040 tokens boots with 1,973 / 737 MiB free on cuda:0 / cuda:1, so the floors are 1,941 / 705 MiB.

| pool | free VRAM at boot, MiB | fingerprints | result |
| --- | --- | --- | --- |
| 999,424 | 2,213 / 957 | `f4add302` / `4a255910` | normal placement, 120k prompt + 4-request round survive |
| 1,015,808 | 2,113 / 837 | same | same |
| 1,032,192 | 2,013 / 737 | same | same |
| 1,048,576 | 1,913 / 577 | | below the floors: the ladder stops |

1,032,192 is served: +49,152 tokens over 983,040 and +26 % over the 819,200 served at the start of the day ([R540](r540-promote-r6.md)).

## A/B (4 boots, A B B A, NVMe tier off)

A is the served launcher at 983,040, B the candidate at 1,032,192. `mp_decode.py`: 24 code and 24 prose prompts, each decoded to 512 forced tokens, greedy, at 1 stream and in groups of 4. Paired geometric mean B/A over prompts, 95 % bootstrap interval:

| shape | A per-request (t/s) | B per-request (t/s) | B / A | per-prompt range |
| --- | --- | --- | --- | --- |
| code, 1 stream | 193.6 | 197.6 | +1.88 % [−1.33, +5.03] | −12.4 to +20.3 % |
| code, 4 streams | 121.6 | 124.2 | +2.05 % [−0.26, +4.38] | −9.0 to +14.9 % |
| prose, 1 stream | 195.3 | 196.0 | +0.34 % [−1.35, +1.97] | −7.4 to +8.2 % |
| prose, 4 streams | 115.7 | 117.3 | +1.37 % [−1.09, +3.49] | −14.3 to +10.0 % |

The unit stops a promotion when any shape reads below −1.5 %; none did.

## Gates (NVMe tier on)

| gate | result |
| --- | --- |
| G1 boot | 1,032,192 tokens, 8-bit KV, 4 slots, vision on, normal placement, fingerprints equal to the ladder's |
| G1b cold prefill, salted | 60k target 9,422 t/s, 120k target 9,954 t/s (floors 9,000 and 9,500) |
| GT restart restore | a 30,000-token prompt after a container restart: 29,952 tokens served from the tier in 0.58 s, output identical to the warm run (0.37 s) and the cold run (3.97 s) |
| G2 agentic-edit | 6/6 in four modes; greedy c4 first full wave 518.8 t/s aggregate, 157.8 per stream |
| G3 needles | 5/5 at ~131k and 5/5 at ~240k prompt tokens |
| G4 tool-eval 69×4 | 86.8 ± 2.6 (95 % interval 84.5 to 89.0) |
| G5 GSM8K 5-shot n=500, no stop strings | 0.974 |

Promoted 2026-09-19 15:18 UTC (17:18 CEST); the served launcher then booted with 2,013 / 737 MiB free. Rollback: the R546 launcher.
