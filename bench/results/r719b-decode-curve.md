# R719b: the decode curve of the served configuration after R728 (draft-KV window off) and R726 (+4500 memory offset re-applied)

Results directory on the serving host: `results/2026-09-25-r719-decode-curve` (the R719 driver re-run unchanged on 2026-09-25; the driver stamps its directory with the run date, so R719's own directory is `results/2026-09-24-r719-decode-curve`). Raw records: [`2026-09-25-r719b-decode-curve/`](2026-09-25-r719b-decode-curve/) (`records.jsonl` one line per request, `curve.tsv` and `analysis.txt` from the driver's analysis step, `audit.txt`; the boot and container logs stay on the host). Driver [`scripts/r719-decode-curve.sh`](../../scripts/r719-decode-curve.sh); there is no separate R719b script, the "b" labels the second publication. Instrument `fn_bench` ([`bench/probe.py`](../probe.py)) with `--distinct`. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, image `tabbyapi:stack-r3-rows32`, 41 environment keys (the served set of [R728](r728-promote-window-off.md), without `EXL3_MTP_KV_WINDOW`), 8 slots, 983,040-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft component on the second GPU, draft policy `[[4, 3], [8, 2]]`, memory clock offset +4500 on both cards (launcher readback `4500 4500` at both boots).

## Why

[R719](r719-decode-curve.md) measured the README's decode curve at 00:27 to 00:44 UTC on 2026-09-25. Two things changed on the served configuration after it. [R726](r726-memoc.md) found the +4500 memory clock offset at 0 on both cards at 02:22 UTC, so R719 ran at the stock memory clock, and the launcher re-applies the offset at every boot since then. [R728](r728-promote-window-off.md) removed the windowed MTP draft cache at 03:52 UTC. This round re-measures the curve on the configuration served after both, with R719's driver, prompts and metrics.

| metric | definition |
| --- | --- |
| decode rate per stream | median over requests of (tokens − 1) / (t_last − t_first), the streaming rate after the first token |
| decode aggregate | sum of the decode rates of the requests in one round, averaged over rounds |
| round-wall aggregate (end-to-end burst) | all streams' tokens over the round's wall time, averaged over rounds |
| TTFT | median over requests of the time from sending the request to the first streamed token |

## What was measured

2026-09-25 07:59 to 08:16 UTC, two boots of the served launcher (NEW1, NEW2), each running `fn_bench` at 1 to 8 streams, code and prose: greedy, 1,024 forced tokens, one warm-up round and three recorded rounds per shape, NVMe tier off. Prompts are 118 tokens (code) and 106 tokens (prose), the same as R719's. All 432 recorded requests ended with `finish_reason: length` at 1,024 server-counted tokens, and no request streamed more frames than tokens. The container logs of both boots have 0 out-of-memory lines, and every one of their 722 request lines reads `none cached`: a prompt this short fills no 256-token page, so no request revives a cached prefix. Free VRAM was 1,125 / 1,573 MiB after each boot and 1,019 / 1,465 MiB after each boot's run ([`audit.txt`](2026-09-25-r719b-decode-curve/audit.txt)); cuda:1 has 940 MiB less than in R719 because the draft cache covers the whole page pool since R728.

The table below was recomputed from `records.jsonl` with `bench/plot.py`'s `decode_rates`, and every cell equals `curve.tsv` to the printed precision.

## The served configuration

Both boots pooled (6 rounds per shape):

| streams | code, decode per stream | code, decode aggregate | code, round-wall aggregate | code, TTFT | prose, decode per stream | prose, decode aggregate | prose, round-wall aggregate | prose, TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 298.4 | 298 | 287 | 0.14 s | 278.6 | 279 | 269 | 0.13 s |
| 2 | 207.6 | 415 | 396 | 0.24 s | 204.7 | 412 | 384 | 0.23 s |
| 3 | 182.2 | 541 | 495 | 0.35 s | 181.0 | 542 | 509 | 0.33 s |
| 4 | 157.8 | 632 | 583 | 0.46 s | 159.9 | 643 | 591 | 0.43 s |
| 5 | 137.7 | 691 | 635 | 0.56 s | 138.8 | 694 | 641 | 0.54 s |
| 6 | 121.2 | 727 | 671 | 0.62 s | 122.6 | 735 | 679 | 0.62 s |
| 7 | 112.7 | 784 | 724 | 0.69 s | 113.3 | 792 | 732 | 0.66 s |
| 8 | 105.8 | 845 | 772 | 0.75 s | 106.9 | 853 | 777 | 0.70 s |

All rates are tokens per second. The README's decode figure ([`docs/img/decode-scaling.svg`](../../docs/img/decode-scaling.svg)) draws the two decode columns of this table.

The decode aggregate rises at every step from 1 to 8 streams, 2.84 times on code and 3.06 times on prose. From 5 to 6 streams it rises by 5.2 % on code (691 to 727) and 5.9 % on prose (694 to 735). The per-stream rate falls by 12.7 % (code) and 13.2 % (prose) from 4 to 5 streams, where the policy drops from three draft tokens to two, by 12.0 % and 11.7 % from 5 to 6 streams, and by 12.7 % and 12.8 % from 6 to 8 streams. Code decodes 7.1 % faster than prose at 1 stream; at 8 streams prose decodes 1.0 % faster than code per stream.

From the container logs, a decode step yields 2.99 tokens (code) and 2.82 (prose) at 1 stream, 2.57 to 2.72 at 2 to 4 streams with three draft tokens, and 2.25 to 2.31 at 5 to 8 streams with two draft tokens.

The two boots differ by at most 1.5 % in the per-stream rate (prose at 4 streams) and 1.5 % in the decode aggregate (code at 2 streams, prose at 4 streams).

**Overlap check.** For each round, the window in which every stream is decoding runs from the latest first token to the earliest last token. At 2 to 8 streams that window covers 92.8 % to 100.0 % of the round's mean decode window; the lowest value is code at 8 streams, and every other shape is at 94.9 % or above. The decode aggregate is a sum of per-stream rates, each over its own window, so it overstates the rate the streams sustain together by at most about 8 %. `bench/plot.py` prints the value for every shape.

## Against R719

Same image, driver, prompts, slots, page pool and draft policy. Two changes lie between the rounds: the +4500 memory clock offset (R719 at 0, R719b at +4500) and the windowed MTP draft cache (on in R719, off in R719b). The per-stream rate is 0.7 to 3.2 % higher in all 16 cells and the decode aggregate 0.5 to 3.0 % higher:

| streams | code, per stream | code, R719b / R719 | code, decode aggregate | prose, per stream | prose, R719b / R719 | prose, decode aggregate |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 1 | 291.1 to 298.4 | 1.025 | 292 to 298 | 273.5 to 278.6 | 1.019 | 272 to 279 |
| 2 | 203.1 to 207.6 | 1.022 | 405 to 415 | 201.0 to 204.7 | 1.019 | 402 to 412 |
| 3 | 180.2 to 182.2 | 1.011 | 538 to 541 | 177.4 to 181.0 | 1.020 | 530 to 542 |
| 4 | 153.2 to 157.8 | 1.030 | 615 to 632 | 156.8 to 159.9 | 1.020 | 627 to 643 |
| 5 | 134.1 to 137.7 | 1.027 | 671 to 691 | 135.7 to 138.8 | 1.022 | 679 to 694 |
| 6 | 118.4 to 121.2 | 1.024 | 708 to 727 | 119.9 to 122.6 | 1.022 | 716 to 735 |
| 7 | 110.8 to 112.7 | 1.016 | 771 to 784 | 112.6 to 113.3 | 1.007 | 783 to 792 |
| 8 | 103.4 to 105.8 | 1.024 | 822 to 845 | 103.6 to 106.9 | 1.032 | 831 to 853 |

The round does not separate the two changes. R726 measured the memory offset alone on the same prompts at 512 forced tokens, on one boot with the window on: +1.8 % (code) and +1.7 % (prose) per stream at 1 stream, and +1.3 to +2.5 % at 4 and 8 streams, which that round's two rounds per state do not resolve. The window's prompt-revive loss ([R721](r728-promote-window-off.md#r721-the-window-and-revived-prompts)) cannot occur here, since no request revives a cached prefix. Whether the full-pool draft cache changes the cost of draft attention on fresh prompts was not resolved by R722. TTFT is within 0.02 s of R719's at 1 to 6 streams and 0.00 to 0.07 s lower at 7 and 8 streams.
