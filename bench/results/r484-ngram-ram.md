# R484: the n-gram table in host RAM is byte-identical and worth 1–2 % at c1, for 30.5 GiB

Results directory on the serving host: `results/2026-09-18-r484-ngram-ram`. Raw records: [`2026-09-18-r484-ngram-ram/`](2026-09-18-r484-ngram-ram/). Driver: [`scripts/r484-ngram-ram.sh`](../../scripts/r484-ngram-ram.sh). Date: 2026-09-18.

`ngram_ram` loads the checkpoint's per-layer n-gram embedding table into anonymous host memory instead of reading it from the page cache. Served configuration (3.05bpw pack, 4 slots, 360,448, 8-bit KV, [30, 30], chunk 2048), arms OFF / ON / OFF2 / ON2, `fn_bench` 2,048 forced tokens, 2 runs.

| arm | fingerprints c1 / 30k | code c1 | code c4 | prose c1 | prose c4 | prefill 22.6k / 90.1k (t/s) | host MemAvailable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | canonical / canonical | 213.8 / 214.7 | 431.8 / 455.4 | 168.7 / 170.0 | 424.5 / 435.5 | 7,273 / 11,080 | 47.2 GiB |
| ON | canonical / canonical | 216.5 / 216.8 | 440.7 / 445.0 | 171.8 / 172.3 | 432.8 / 437.0 | 7,759 / 11,137 | 16.9 GiB |
| OFF2 | canonical / canonical | 212.2 / 216.8 | 428.6 / 442.0 | 166.2 / 172.2 | 422.6 / 434.2 | 7,263 / 11,096 | 47.2 GiB |
| ON2 | canonical / canonical | 216.7 / 217.0 | 442.3 / 444.6 | 170.8 / 172.3 | 434.5 / 436.4 | 7,719 / 11,147 | 16.9 GiB |

ON sits at the top of every OFF range. The c1 gain is 1–2 %, the c4 difference is inside the day's spread, and the 22.6k-token prefill gains 6 %. The price is 30.5 GiB of the 60 GiB host. Not served: the operator ruled out spending host RAM on it.
