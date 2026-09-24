# R704: the decode curve of the served configuration, measured after the first token, against the previous configuration

Results directory on the serving host: `results/2026-09-24-r704-decode-curve-ab`. Raw records: [`2026-09-24-r704-decode-curve-ab/`](2026-09-24-r704-decode-curve-ab/) (`records.jsonl` one line per request, `curve.tsv` and `analysis.txt` from the driver's analysis step, `audit.txt`). Driver [`scripts/r704-decode-curve-ab.sh`](../../scripts/r704-decode-curve-ab.sh). Instrument `fn_bench` ([`bench/probe.py`](../probe.py)). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 999,424-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft component on the second GPU, draft policy `[[4, 3], [5, 2], [8, 1]]`.

## Why

The README's decode figure came from [R580](r580-decode-curve.md) (2026-09-20) and reported the round-wall aggregate: all streams' tokens divided by the wall time of the round. That figure includes the time to the first token of every stream and the tail of the slowest stream. R694 and R701 were promoted on the canonical gate, which reports the per-request median decode rate, so the published curve had not been re-measured since R580. This round re-measures it with R580's instrument on the configuration served before R701 and on the one served since, and records three metrics from the same requests.

| metric | definition |
| --- | --- |
| decode rate per stream | median over requests of (tokens − 1) / (t_last − t_first), the streaming rate after the first token |
| decode aggregate | sum of the decode rates of the requests in one round, averaged over rounds |
| round-wall aggregate (end-to-end burst) | all streams' tokens over the round's wall time, averaged over rounds; the README's headline metric until this round |
| TTFT | median over requests of the time from sending the request to the first streamed token |

`fn_bench` records `t_first_abs` and `t_last_abs` per request since this round, so the overlap of the streams' decode windows can be read from the records.

## What was measured

2026-09-24 09:18 to 09:56 UTC, one session, 2 boots per arm, alternating OLD, NEW, OLD, NEW:

- OLD: the launcher served before R701, image `tabbyapi:slotfix-r1` with the draft component on the second GPU ([R694](r694-mtp-card1.md)).
- NEW: the served launcher, image `tabbyapi:stack-r2` ([R701](r701-stack-r2.md)).

Each boot ran `fn_bench` at 1 to 8 streams, code and prose: greedy, 1,024 forced tokens, one warm-up round and three recorded rounds per shape, NVMe tier off. All 864 recorded requests ended with `finish_reason: length` at 1,024 server-counted tokens. The container logs of the four boots have 0 out-of-memory lines. Free VRAM was 1,041 / 2,429-2,431 MiB after each boot and 961 / 2,347 MiB after each boot's run ([`audit.txt`](2026-09-24-r704-decode-curve-ab/audit.txt)).

## The served configuration

NEW arm, both boots pooled (6 rounds per shape):

| streams | code, decode per stream | code, decode aggregate | code, round-wall aggregate | code, TTFT | prose, decode per stream | prose, decode aggregate | prose, round-wall aggregate | prose, TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 236.7 | 237 | 230 | 0.13 s | 227.0 | 226 | 220 | 0.13 s |
| 2 | 199.2 | 398 | 381 | 0.23 s | 209.4 | 419 | 401 | 0.22 s |
| 3 | 164.4 | 495 | 467 | 0.34 s | 160.5 | 482 | 459 | 0.31 s |
| 4 | 147.9 | 592 | 556 | 0.44 s | 141.1 | 569 | 534 | 0.42 s |
| 5 | 139.1 | 688 | 632 | 0.54 s | 130.3 | 648 | 604 | 0.51 s |
| 6 | 109.7 | 659 | 617 | 0.60 s | 107.2 | 644 | 609 | 0.57 s |
| 7 | 105.8 | 749 | 690 | 0.72 s | 103.1 | 720 | 675 | 0.62 s |
| 8 | 95.9 | 764 | 714 | 0.72 s | 95.3 | 761 | 714 | 0.67 s |

All rates are tokens per second. The README's decode figure ([`docs/img/decode-scaling.svg`](../../docs/img/decode-scaling.svg)) draws the two decode columns of this table.

The decode aggregate falls from 5 to 6 streams on code (688 to 659) and is flat on prose (648 to 644). The draft policy drops from two draft tokens to one at 6 jobs. The per-stream rate falls from 5 to 6 streams by 21 % on code and 18 % on prose, and by a further 13 % and 11 % from 6 to 8 streams.

**Overlap check.** For each round, the window in which every stream is decoding runs from the latest first token to the earliest last token. At 2 to 8 streams that window covers 96.3 % to 100 % of the round's mean decode window for every shape and arm, and 96.8 % to 100 % when code and prose are pooled per stream count. The lowest value is code at 8 streams, 96.3 % in both arms. The decode aggregate is a sum of per-stream rates, each over its own window, so it overstates the rate the streams sustain together by at most about 4 %. `bench/plot.py` prints the value for every shape.

## Against the previous configuration

Decode rate per stream, NEW / OLD, both boots of each arm pooled:

| streams | code, OLD | code, NEW | code, NEW / OLD | prose, OLD | prose, NEW | prose, NEW / OLD |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 214.7 | 236.7 | 1.103 | 205.8 | 227.0 | 1.103 |
| 2 | 176.5 | 199.2 | 1.129 | 184.5 | 209.4 | 1.134 |
| 3 | 143.9 | 164.4 | 1.143 | 140.5 | 160.5 | 1.143 |
| 4 | 139.4 | 147.9 | 1.061 | 132.9 | 141.1 | 1.061 |
| 5 | 127.8 | 139.1 | 1.088 | 122.2 | 130.3 | 1.067 |
| 6 | 95.4 | 109.7 | 1.150 | 93.6 | 107.2 | 1.146 |
| 7 | 99.5 | 105.8 | 1.063 | 97.2 | 103.1 | 1.061 |
| 8 | 91.2 | 95.9 | 1.051 | 90.1 | 95.3 | 1.058 |

At every shape the slower NEW boot is faster than the faster OLD boot: the smallest ratio of the two per-boot medians is 1.041 (prose, 8 streams). The two boots of an arm differ by at most 2.0 %, except OLD code at 5 streams, 5.7 %.

Decode aggregate and round-wall aggregate, OLD to NEW:

| streams | code, decode aggregate | code, round-wall aggregate | prose, decode aggregate | prose, round-wall aggregate |
| ---: | --- | --- | --- | --- |
| 1 | 214 to 237 | 208 to 230 | 205 to 226 | 200 to 220 |
| 2 | 353 to 398 | 339 to 381 | 370 to 419 | 355 to 401 |
| 3 | 434 to 495 | 412 to 467 | 420 to 482 | 402 to 459 |
| 4 | 559 to 592 | 526 to 556 | 534 to 569 | 503 to 534 |
| 5 | 641 to 688 | 590 to 632 | 606 to 648 | 566 to 604 |
| 6 | 573 to 659 | 541 to 617 | 561 to 644 | 533 to 609 |
| 7 | 696 to 749 | 644 to 690 | 680 to 720 | 640 to 675 |
| 8 | 726 to 764 | 680 to 714 | 719 to 761 | 676 to 714 |

TTFT is the same in both arms within 0.01 s at every shape except code at 7 streams (0.67 s OLD, 0.72 s NEW). R701's two overlays change decode kernels only.

## R580's published figure

R580 (2026-09-20, results `2026-09-20-r580-decode-curve-try2`) measured the round-wall aggregate for prose at 8 streams as 630 tokens per second on the configuration served that day. The OLD arm here, the configuration served from R694 to R701, reads 676 on the same metric and instrument, and NEW reads 714. The images promoted between R580 and R694 ([R587](r587-tabby-metrics.md), [R646](r646-verifybatch.md), [R653](r653-stack.md), [R676](r676-slotfix.md)) were gated on greedy fingerprints and per-request medians, and the round-wall curve was not re-measured for them. R580's per-stream column was the per-request wall rate (tokens over the request's wall time, TTFT included), not the decode rate defined above. R580 remains the source of the README's prefill-depth figure.
