# R482: Cold experts on the CPU

Results directory on the serving host: `results/2026-09-18-r482-cold-experts`. Raw records: [`2026-09-18-r482-cold-experts/`](2026-09-18-r482-cold-experts/). Date: 2026-09-18.


The served image's exllamav3 can keep N of each layer's 512 routed experts in a CPU worker (`cpu_moe_split_experts`), overlapped with the GPU experts, and every 128 decode steps it swaps the most-selected CPU expert with the least-selected GPU one. At 4 slots and 8-bit KV:

| arm | pool that boots | host memory added | code c1 / c4 | prose c1 / c4 | cold prefill, 22,625 / 90,135 tokens | needle 131k / 240k | GSM8K c4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| served, no split | 360,448 | — | 214–217 / 429–444 | 167–172 / 422–436 | 3.12 s / 8.19 s | 5/5 / 5/5 | 0.925 |
| 8 experts on the first 8 layers | 327,680 (360,448 fails) | 2.7 GiB | 147–153 / 408–425 | 149–153 / 353–368 | — | — | — |
| 8 experts on the first 30 layers | 360,448 (393,216 fails) | 3.3 GiB | 140–173 / 338–348 | 131–137 / 345–355 | 5.23 s / 15.27 s | 5/5 / 5/5 | 0.910 |

The split moves about 14 MiB of experts per layer off the card, and its GPU-side buffers take more than that: with eight layers split the served pool no longer fits. Decode drops 20–30 % with 1.6 % of the experts on the CPU, because 16 verification rows with 10 experts each reach a CPU expert in most layers at most steps, and each such layer waits for the worker. The worker holds each split layer's experts in host memory, about 340 MiB per layer. The run stopped at 8 experts per layer by its own rule (code c1 below 161 t/s); 16 and 32 were not run.
