# R537: a deeper MTP draft at 1 stream is +5 % on code and −5 to −8 % on prose; not served

Results directory on the serving host: `results/2026-09-19-r537-draft-depth`. Raw records: [`2026-09-19-r537-draft-depth/`](2026-09-19-r537-draft-depth/). Driver: [`scripts/r537-draft-depth.sh`](../../scripts/r537-draft-depth.sh). Date: 2026-09-19.

The served draft policy is `[[4, 3], [8, 1]]`: depth 3 up to 4 decoding jobs. The arms here use `[[1, D], [4, 3], [8, 1]]`: a single decoding job drafts D tokens, 2 to 4 jobs keep depth 3. The GDN recurrent-state history holds one fp32 state copy per slot per draft position, so each extra depth costs VRAM that the page pool gives up. All boots use the served image and environment, with the NVMe tier off.

## Pool cost

Largest pool that boots, stepping down from the served 819,200 in steps of 16,384 tokens; the smaller pools fail with "Insufficient VRAM in split for model and cache":

| policy | largest pool | against the served pool |
| --- | --- | --- |
| D4 | 802,816 | −16,384 |
| D5 | 770,048 | −49,152 |
| D6 | 753,664 | −65,536 |

## Decode at a common pool

Every arm at 753,664 (the D6 pool), so the arms differ only in the policy. 8 boots in the order S D4 D5 D6 D6 D5 D4 S (S: the served policy). Per boot: c1 greedy fingerprint, then `bench/probe.py`, 2,048 forced tokens, greedy, one unrecorded warm-up round, code c1 × 3, prose c1 × 3 and code c4 × 1.

| shape | S | D4 | D5 | D6 |
| --- | --- | --- | --- | --- |
| code c1, t/s (6 runs, sd) | 211.8 (1.2) | 222.1 (1.1), +4.8 % | 222.9 (0.9), +5.2 % | 163.5 (0.6), −22.8 % |
| prose c1, t/s (6 runs, sd) | 192.8 (0.7) | 182.4 (1.1), −5.4 % | 177.1 (0.4), −8.2 % | 182.4 (0.8), −5.4 % |
| code c4 aggregate, t/s (2 runs) | 548.3 | 548.0 | 497.4 | 503.9 |

The code c4 row is one run per boot and was a crash check (no arm crashed). Its request mix differs by arm: in the S and D4 runs one request stopped at 1,807 tokens, in the D5 and D6 runs two stopped at 1,861, and the aggregate includes them.

Depth and pool both change the greedy output. c1 fingerprints: S at 753,664 `4fad2dbfab9374a4`, D4 and D5 `3625375f17a5dbcc`, D6 `fecfb8a8a453c8aa`; the served policy at 819,200 gives `e7fb377c987d685c`. Free VRAM after boot also differs by arm at the same pool: S 769 / 3,161 MiB, D4 569 / 2,939, D5 2,031 / 893, D6 1,829 / 691.

A fixed deeper depth at 1 stream raises code decode and lowers prose decode, and costs 16,384 to 65,536 tokens of pool. It is not served. A depth change would also need the full quality gates, because it changes the output.
