# R520: switching off the int8-activation GEMV changes no output and moves decode by +0.5 to +1.0 %, inside the run spread

Results directory on the serving host: `results/2026-09-19-r520-int8gemv`. Raw records: [`2026-09-19-r520-int8gemv/`](2026-09-19-r520-int8gemv/). Driver: [`scripts/r520-int8gemv.sh`](../../scripts/r520-int8gemv.sh). Date: 2026-09-19.

ExLlamaV3 defaults to `EXL3_INT8_GEMV=2`: single-row linears with the `mul1` codebook run a fused Hadamard + int8-activation dp4a GEMV instead of the fp16 kernel. On a GB10, turning it off was reported at +3 % decode. Arms ON / OFF / ON2 / OFF2 on the served launcher; OFF adds `EXL3_INT8_GEMV=0` to the served environment (checked in the container).

- The c1 greedy fingerprint is `ae890c45d1000582` on all four arms, so the int8 path does not touch the verified output here. The kernel cannot be captured in a CUDA graph, so graphed calls already use the fp16 kernel (`exl3_gemm.cu`, the `mul1 && exl3_gemv_int8_enabled()` branch); [R519](r519-profile-2p50.md) shows it at 0.38 ms per step at 1 stream and almost none at 4.

| load (`bench/probe.py`, 2,048 tokens, 2 boots × 2 runs, mean aggregate) | ON (t/s) | OFF (t/s) | change |
| --- | --- | --- | --- |
| code c1 | 216.7 | 217.9 | +0.5 % |
| code c4 | 509.9 | 512.5 | +0.5 % |
| prose c1 | 190.7 | 191.6 | +0.5 % |
| prose c4 | 473.1 | 477.8 | +1.0 % |

The ON runs at code c1 range from 210.2 to 220.0 t/s. Not served: the difference is inside that spread.
