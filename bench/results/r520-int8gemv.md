# R520: switching off the int8-activation GEMV changes no output; its decode effect is below what this design resolves

Results directory on the serving host: `results/2026-09-19-r520-int8gemv`. Raw records: [`2026-09-19-r520-int8gemv/`](2026-09-19-r520-int8gemv/). Driver: [`scripts/r520-int8gemv.sh`](../../scripts/r520-int8gemv.sh). Date: 2026-09-19.

ExLlamaV3 defaults to `EXL3_INT8_GEMV=2`: single-row linears with the `mul1` codebook run a fused Hadamard + int8-activation dp4a GEMV instead of the fp16 kernel. On a GB10, turning it off was reported at +3 % decode. Arms ON / OFF / ON2 / OFF2 on the served launcher; OFF adds `EXL3_INT8_GEMV=0` to the served environment (checked in the container).

- The c1 greedy fingerprint is `ae890c45d1000582` on all four arms, so the int8 path does not touch the verified output here. The kernel cannot be captured in a CUDA graph, so graphed calls already use the fp16 kernel (`exl3_gemm.cu`, the `mul1 && exl3_gemv_int8_enabled()` branch); [R519](r519-profile-2p50.md) shows it at 0.38 ms per step at 1 stream and almost none at 4.

| load (`bench/probe.py`, 2,048 tokens, 2 boots × 2 runs, mean aggregate) | ON (t/s) | OFF (t/s) | change |
| --- | --- | --- | --- |
| code c1 | 216.7 | 217.9 | +0.5 % |
| code c4 | 509.9 | 512.5 | +0.5 % |
| prose c1 | 190.7 | 191.6 | +0.5 % |
| prose c4 | 473.1 | 477.8 | +1.0 % |

The table above overstates OFF. ON always booted first, and every first run of that first boot read 1–4 % low (code c1 210.2 t/s against 216.8–220.0 for the other ON runs). Without each boot's first run the difference is −0.6 % (code c1), −0.3 % (code c4), +0.1 % (prose c1) and −0.3 % (prose c4). The run-level 95 % confidence intervals are about ±2.5 % wide, so this design cannot resolve a 0.5 % effect in either direction.

`bench/probe.py` now takes `--warmup-runs N`: full-length rounds per shape, run first and not recorded. The re-measurement (`scripts/r520b-int8gemv-precise.sh`) boots 8 times in the order A B B A B A A B, uses one warm-up round per shape, 3 runs at c1 and 2 at c4 per boot, and logs GPU clocks and draft acceptance per boot.
