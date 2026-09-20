# R588: `chunk_size` against prefill interference

Results directory on the serving host: `results/2026-09-20-r588-chunk-vs-interference`. Raw records: [`2026-09-20-r588-chunk-vs-interference/`](2026-09-20-r588-chunk-vs-interference/). Driver: [`scripts/r588-chunk-vs-interference.sh`](../../scripts/r588-chunk-vs-interference.sh).

Measurement only. Nothing was promoted.

## Configuration correction, read this before the numbers

The raw `analysis.txt` in this directory prints the header `NVMe tier on`. That label is wrong for all three arms. The driver copied it as a constant from [R585](r585-prefill-interference.md) and did not read the running configuration. **All three arms ran with the NVMe prefix tier off**, because the launcher enabled the tier only when `IMG` matched a literal image name, and [R587](r587-tabby-metrics.md) had changed the promoted image the day these arms ran. The launcher now tests `IMG` against a `DAILY_IMG` variable that the promotion edits, so the two cannot desync.

The three arms are therefore internally comparable to each other — same pool, same victim, same boot path, tier off throughout — and their absolute rates are **not** comparable to R585's, which had the tier on.

## Method

Three boots at `chunk_size` 512, 1024 and 2048, with `cache_size` pinned at 999,424 on every arm. The pin matters: the loader sizes its headroom by running every module on a dummy chunk of `chunk_size` tokens, so a smaller chunk frees pool. Letting the pool float would have confounded chunk against KV size.

The victim is R585's, unchanged: greedy, 3,000 forced tokens at roughly 9,850 prompt tokens, one stream, two runs. Each arm runs it alone and again under R585's interference background — a fresh prompt of roughly 45,000 tokens submitted every 8 seconds, 64 tokens out, each with its own salt so no arm is served another's cached pages.

## Results

| `chunk_size` | decode alone | decode under interference | ratio | interferer TTFT | free VRAM at boot |
| --- | --- | --- | --- | --- | --- |
| 512 | 269.5 t/s | 134.8 t/s | 0.50× | 12.6 s | 337 / 3,275 MiB |
| 1024 | 275.2 t/s | 116.1 t/s | 0.42× | 8.9 s | 1,041 / 2,551 MiB |
| 2048 (served) | 275.4 t/s | 114.7 t/s | 0.42× | 6.1 s | 1,041 / 2,531 MiB |

Decode rates are prose content, greedy, one stream, 3,000 forced tokens at ~9,850 prompt tokens. TTFT is the median over the interferer's own requests in that arm.

## What the numbers say

**`chunk_size` does not change the single-stream rate.** 269.5, 275.2 and 275.4 t/s across a fourfold change in chunk is inside the run-to-run band. A stream that owns the box is unaffected by how prefill is granulated, because there is no prefill to interleave with. Every decode figure published in this repository was measured in that regime, which is why this knob has never shown up in one.

**Under interference the effect appears, and it is a transfer rather than a gain.** chunk 512 delivers the running generation 134.8 t/s against 2048's 114.7, an increase of 17.5 percent. It delivers the arriving request its first token in 12.6 seconds against 6.1, an increase of 107 percent. Finer granulation lets decode steps run sooner between prefill chunks, and the prefill they interleave with takes correspondingly longer to complete. In an agent session the same client is both parties: the session that benefits from the faster running generation is the session waiting on the next prompt.

**chunk 512 also fails the headroom rule.** It boots with 337 MiB free on the first card against the reference 1,041 MiB, because the smaller loader headroom lets the splitter place an additional layer on that card. The rule requires at least the reference minus 32 MiB. At this pool, 512 is not a candidate regardless of the throughput reading.

## A measurement that was not the one intended

The 2048 arm exists as a reproduction check: it is the served configuration, so it should have matched R585's 236.4 t/s alone and 0.65× under the same background. It read 275.4 and 0.42×. The gap is the NVMe tier described above, not chunk size.

That leaves four measurements of one victim shape separated by the tier: 283 t/s ([R583](r583-long-generation.md), tier off), 275.4 (R588, tier off), against 236.4 (R585, tier on). Two independent tier-off readings within 3 percent of each other, and roughly 15 percent above the single tier-on reading. R589 measures that directly, in both directions, since the tier exists to serve prefixes from disk across a restart and a round that measured only the decode cost would be recommending the removal of a cache without measuring what it caches.

R589 reads the tier state from the container and aborts if it disagrees with the arm label. A round that labels its arms from a constant cannot detect a configuration that moved underneath it, which is what happened here.
