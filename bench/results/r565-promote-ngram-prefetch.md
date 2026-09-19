# R565: the n-gram row prefetch is served; +0.8 % code and +0.9 % prose at 1 stream over four boots, output unchanged

Results directory on the serving host: `results/2026-09-19-r565-promote-ngram-prefetch`. Raw records: [`2026-09-19-r565-promote-ngram-prefetch/`](2026-09-19-r565-promote-ngram-prefetch/). Driver: [`scripts/r565-promote-ngram-prefetch.sh`](../../scripts/r565-promote-ngram-prefetch.sh). Overlay: [`docker/overlays/ngram-prefetch-r1/`](../../docker/overlays/ngram-prefetch-r1/). Date: 2026-09-19, 18:00–18:39 UTC.

## What changes

Three launcher lines: the image (`tabbyapi:ngram-prefetch-r1-gdnbf16`, the [R561](r561-promote-slots8.md) image plus the overlay of [R559](r559-ngram-prefetch.md)), the same name in the NVMe tier default, and `EXL3_NGRAM_PREFETCH2=1`. Pool, slots and draft policy are unchanged.

## A/B (4 boots, A B B A, NVMe tier off)

The rule, fixed before the run: fingerprints equal on every boot, each card's free VRAM at boot no more than 32 MiB below A, cold prefill at least 0.95× A, every `mp_decode` shape at or above zero, and at least two shapes with the 95 % interval above zero.

- All four boots: c1 `f4add302e176d78e`, 30k `4a255910dee2d9c5`, 2,085 / 867 MiB free at boot.
- Cold prefill, mean of the B boots over the A boots: 60k target 9,612 / 9,402 t/s (1.022×), 120k target 10,134 / 10,056 t/s (1.008×).

| shape | A per-request (t/s) | B per-request (t/s) | B / A | per-prompt range |
| --- | --- | --- | --- | --- |
| code, 1 stream | 199.2 | 200.8 | +0.81 % [+0.59, +1.02] | −0.6 to +2.3 % |
| code, 4 streams | 126.9 | 127.1 | +0.23 % [−0.76, +1.36] | −4.1 to +9.2 % |
| prose, 1 stream | 197.6 | 199.5 | +0.94 % [+0.56, +1.31] | −1.9 to +3.4 % |
| prose, 4 streams | 118.3 | 119.0 | +0.59 % [−0.25, +1.45] | −2.4 to +4.9 % |

The four-boot read is smaller than R559's one-boot read; it is the served figure.

## Gates (NVMe tier on, live daily)

| gate | result |
| --- | --- |
| G1 boot | 966,656 tokens, 8 slots, flag and image in the container, tier on, fingerprints equal to A |
| GT restart restore | 29,952 of 30,000 prompt tokens served from the tier after a restart in 0.61 s, output identical to the warm run (0.37 s) and the cold run (4.00 s) |
| G2 agentic-edit | 6/6 in four modes |
| G3 needles | 5/5 at ~131k and 5/5 at ~240k prompt tokens |
| G4 tool-eval 69×4 | 84.0 ± 2.4 (95 % interval 82.0 to 86.0) |
| G5 GSM8K 5-shot n=500, no stop strings | 0.978 |

Greedy output is byte-identical to R561, and tool-eval samples at temperature 0.6: its 84.0 against R561's 88.2 lies within the 84.5-to-88.2 range of the twelve configurations before it, which differ by sampling. Promoted 2026-09-19 18:15 UTC (20:15 CEST); gates finished 18:39 UTC. Rollback: the R561 launcher.
