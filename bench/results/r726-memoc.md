# R726: the +4500 memory clock offset had reset to 0; restored, +14.3 % DRAM bandwidth and +1.7 to +1.8 % decode at 1 stream

Results directory on the serving host: `results/2026-09-25-r726-memoc`. Raw records: [`2026-09-25-r726-memoc/`](2026-09-25-r726-memoc/) (`bw-*.txt` the bandwidth probe per rung, `memclk-*.txt` the host-side memory clock samples per rung and card, `records.jsonl` one line per decode request, `greedy.jsonl` and `greedy-compare.txt`, `bench-*.log`, `summary.txt`, `audit.txt`; the boot and container logs stay on the host). Driver [`scripts/r726-memoc.sh`](../../scripts/r726-memoc.sh). Bandwidth probe [`bench/membw.py`](../membw.py). Image `tabbyapi:stack-r3-rows32`, model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, the launcher served at 02:22 UTC on 2026-09-25 (42 keys, the windowed MTP draft cache on, 983,040-token page pool, 8 slots, 8-bit KV, draft policy `[[4, 3], [8, 2]]`).

## What was found

At 02:22 UTC on 2026-09-25 both cards read a memory clock offset of 0 (`offsets at entry: [0, 0]` in [`audit.txt`](2026-09-25-r726-memoc/audit.txt)) and ran their memory at 13,801 MHz under the engine. The host's boot-time service had applied +4500 at the last host boot on 2026-09-02, and the offset was recorded intact on 2026-09-03. The host was not rebooted after that. The logs of this model's runs from 2026-09-19 on record 13,801 MHz under load, the stock clock, so the offset reset between 2026-09-03 and 2026-09-19; the cause is not determined.

Every number published in this repository from 2026-09-19 to 2026-09-25 02:22 UTC was measured at the stock memory clock, including the decode curves of [R580](r580-decode-curve.md), [R704](r704-decode-curve.md) and [R719](r719-decode-curve.md). The memory clock of the runs from 2026-09-16 to 2026-09-18 is not established. The README's hardware section stated +4500 on both cards for that whole period.

## Bandwidth ladder

The engine was stopped, and [`bench/membw.py`](../membw.py) ran once per card per rung in a container with one visible GPU: 300 device-to-device copies of a 1 GiB fp32 buffer after 5 warm-up copies, bandwidth = 2 × 1 GiB × 300 / elapsed time, in 10^9 bytes per second. The probe's own clock sampler did not run (`nvml:ModuleNotFoundError`, the image has no `pynvml`), so the memory clock below is the maximum of the host-side samples, which prove the offset was applied and are not a reading during the copy.

| offset | cuda:0, GB/s | cuda:1, GB/s | host memory clock max, MHz | copies exact |
| ---: | ---: | ---: | ---: | --- |
| +0 | 1,531.5 | 1,528.0 | 14,001 | 2 of 2 |
| +4500 | 1,750.8 | 1,752.4 | 16,251 | 2 of 2 |
| +5000 | 1,744.4 | 1,775.8 | 16,501 | 2 of 2 |

+4500 raises the memory clock by 16.1 % and the bandwidth by 14.3 % (cuda:0) and 14.7 % (cuda:1). At +5000 cuda:0 read 0.4 % below +4500 while cuda:1 rose 1.3 %, and the ladder stopped by its pre-registered rule (a rung must add 1 % on every card). One sample per card per rung; the two cards agree within 0.23 % at +0 and 0.09 % at +4500. An offset of +4500 appears as +2,250 MHz in the reported clock at every rung. No new Xid was logged.

## Decode at +0 and +4500

One boot of the served launcher, the offset switched live between four states in the order O0, O45, O0b, O45b. In each state `fn_bench --distinct`, code and prose, at 1, 4 and 8 streams: greedy, 512 forced tokens, one warm-up round and two recorded rounds, prompts of 118 tokens (code) and 106 tokens (prose), NVMe tier off. All 208 recorded requests ended with `finish_reason: length`. Under the engine the memory clock read 13,801 MHz at +0 and 16,051 MHz at +4500.

Decode rate per stream after the first token, median over requests, tokens per second ([`summary.txt`](2026-09-25-r726-memoc/summary.txt)):

| cell | O0 | O45 | O0b | O45b | +4500 / +0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| code, 1 stream | 272.3 | 277.7 | 272.4 | 276.6 | 1.018 |
| code, 4 streams | 161.5 | 162.2 | 163.7 | 167.6 | 1.014 |
| code, 8 streams | 102.9 | 107.1 | 106.6 | 107.2 | 1.023 |
| prose, 1 stream | 272.2 | 276.8 | 271.9 | 276.5 | 1.017 |
| prose, 4 streams | 155.2 | 158.2 | 157.7 | 158.8 | 1.013 |
| prose, 8 streams | 105.3 | 106.9 | 103.3 | 107.0 | 1.025 |

The gain is resolved at 1 stream: the two +0 states agree within 0.1 %, and a fit of the round medians with a linear time term gives 1.018 ± 0.25 % (code) and 1.017 ± 0.11 % (prose). At 4 and 8 streams each state has two rounds, the standard deviation of the round medians is 1.0 to 2.3 %, and the two +0 states differ by up to 3.6 %; with the time term the code gain at 4 streams is 0.999 ± 1.7 %. The 4- and 8-stream cells are consistent with the 1-stream gain and do not resolve it. The O0, O45 order puts the +4500 state after the +0 state in both pairs.

Greedy output is identical in all four states (6 of 6 prompts against O0, [`greedy-compare.txt`](2026-09-25-r726-memoc/greedy-compare.txt)). Draft acceptance is the same in every state (the container log: code 0.620 to 0.625, prose 0.596 to 0.600 accepted per proposed token), and no request revived a cached prefix, so the gain is in the step rate.

A 14.3 % bandwidth gain returning 1.7 to 1.8 % decode means that about 13 % of a 1-stream decode step scales with DRAM bandwidth.

## What changed

The launcher sets the +4500 offset on both cards before every boot and logs the readback (`MEMOC`, default 4500; `MEMOC=` skips it), since 2026-09-25. The +4500 state is the served one. The unit left +4500 applied at its end (`leaving +4500: readback 4500 4500`).
