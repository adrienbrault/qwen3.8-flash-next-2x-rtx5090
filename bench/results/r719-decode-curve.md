# R719: the decode curve of the served configuration, each stream on its own prompt

Results directory on the serving host: `results/2026-09-24-r719-decode-curve` (the unit was queued at 23:54 UTC on 2026-09-24 and measured on 2026-09-25). Raw records: [`2026-09-24-r719-decode-curve/`](2026-09-24-r719-decode-curve/) (`records.jsonl` one line per request, `curve.tsv` and `analysis.txt` from the driver's analysis step, `audit.txt`). Driver [`scripts/r719-decode-curve.sh`](../../scripts/r719-decode-curve.sh). Instrument `fn_bench` ([`bench/probe.py`](../probe.py)) with `--distinct`. Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, image `tabbyapi:stack-r3-rows32`, 8 slots, 983,040-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft component on the second GPU, draft policy `[[4, 3], [8, 2]]`.

## Why

The README's decode figure came from [R704](r704-decode-curve.md), measured on `tabbyapi:stack-r2`. Two images were promoted after it, `stack-r3` ([R716b](r716b-stack-r3.md)) and `stack-r3-rows32` with a new draft policy ([R717](r717-rows32.md)), and neither was measured with the README's instrument. R704 also sent one prompt to every stream of a round. A review of R707 (results `2026-09-24-r707-draft-depth-curve`, `stack-r2`, R704's instrument) found that rounds in which the streams followed one trajectory ran 5 to 12 % faster per step than rounds in which they did not, in 10 of 11 cells. This round measures the served configuration with R704's instrument and metrics, with one change: `--distinct` appends a per-stream suffix to each request's prompt, so no two streams of a round share a prompt.

| metric | definition |
| --- | --- |
| decode rate per stream | median over requests of (tokens − 1) / (t_last − t_first), the streaming rate after the first token |
| decode aggregate | sum of the decode rates of the requests in one round, averaged over rounds |
| round-wall aggregate (end-to-end burst) | all streams' tokens over the round's wall time, averaged over rounds |
| TTFT | median over requests of the time from sending the request to the first streamed token |

## What was measured

2026-09-25 00:27 to 00:44 UTC, two boots of the served launcher (NEW1, NEW2), each running `fn_bench` at 1 to 8 streams, code and prose: greedy, 1,024 forced tokens, one warm-up round and three recorded rounds per shape, NVMe tier off. Prompts are 118 tokens (code) and 106 tokens (prose) with the suffix. All 432 recorded requests ended with `finish_reason: length` at 1,024 server-counted tokens. The container logs of both boots have 0 out-of-memory lines. Free VRAM was 1,125 / 2,513 MiB after each boot and 1,019 / 2,405 MiB after each boot's run ([`audit.txt`](2026-09-24-r719-decode-curve/audit.txt)).

## The served configuration

Both boots pooled (6 rounds per shape):

| streams | code, decode per stream | code, decode aggregate | code, round-wall aggregate | code, TTFT | prose, decode per stream | prose, decode aggregate | prose, round-wall aggregate | prose, TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 291.1 | 292 | 281 | 0.14 s | 273.5 | 272 | 263 | 0.13 s |
| 2 | 203.1 | 405 | 386 | 0.24 s | 201.0 | 402 | 373 | 0.23 s |
| 3 | 180.2 | 538 | 494 | 0.35 s | 177.4 | 530 | 500 | 0.34 s |
| 4 | 153.2 | 615 | 566 | 0.46 s | 156.8 | 627 | 575 | 0.45 s |
| 5 | 134.1 | 671 | 618 | 0.57 s | 135.7 | 679 | 625 | 0.55 s |
| 6 | 118.4 | 708 | 656 | 0.64 s | 119.9 | 716 | 662 | 0.63 s |
| 7 | 110.8 | 771 | 710 | 0.76 s | 112.6 | 783 | 722 | 0.66 s |
| 8 | 103.4 | 822 | 752 | 0.76 s | 103.6 | 831 | 759 | 0.71 s |

All rates are tokens per second. The README's decode figure ([`docs/img/decode-scaling.svg`](../../docs/img/decode-scaling.svg)) drew the two decode columns of this table from 2026-09-25 until [R719b](r719b-decode-curve.md) replaced it the same day, after the draft-KV window was turned off ([R728](r728-promote-window-off.md)). Both cards ran at the stock memory clock during this round; the +4500 offset had reset to 0 ([R726](r726-memoc.md)).

The decode aggregate rises at every step from 1 to 8 streams, 2.82 times on code and 3.05 times on prose. From 5 to 6 streams it rises by 5.4 % on code (671 to 708) and 5.5 % on prose (679 to 716); the draft policy keeps two draft tokens from 5 to 8 streams. The per-stream rate falls by 12.5 % (code) and 13.4 % (prose) from 4 to 5 streams, where the policy drops from three draft tokens to two, by 11.7 % on both kinds from 5 to 6 streams, and by a further 12.7 % and 13.6 % from 6 to 8 streams. Code decodes 6.4 % faster than prose at 1 stream and within 0.3 % of it at 8 streams.

The two boots differ by at most 2.6 % in the per-stream rate and 3.6 % in the decode aggregate (both code at 3 streams); every other cell is within 1.8 % and 1.3 %.

**Overlap check.** For each round, the window in which every stream is decoding runs from the latest first token to the earliest last token. At 2 to 8 streams that window covers 92.2 % to 99.9 % of the round's mean decode window; the lowest value is code at 8 streams, and every other shape is at 94.4 % or above. The decode aggregate is a sum of per-stream rates, each over its own window, so it overstates the rate the streams sustain together by at most about 8 %. `bench/plot.py` prints the value for every shape.

## Against R704

R704 and R719 are not strictly comparable. Three things changed between them: the image (`stack-r2` to `stack-r3` to `stack-r3-rows32`), the draft policy (`[[4, 3], [5, 2], [8, 1]]` to `[[4, 3], [8, 2]]`) with the page pool (999,424 to 983,040 tokens), and the instrument: R704 sent one prompt to every stream of a round, R719 one prompt per stream, with prompts 17 tokens longer. The effect of the instrument alone was not measured on the same configuration here; R707's review above found shared-trajectory rounds 5 to 12 % faster per step.

Decode rate per stream and decode aggregate, R704's NEW arm against R719, both boots of each pooled:

| streams | code, per stream | code, R719 / R704 | code, decode aggregate | prose, per stream | prose, R719 / R704 | prose, decode aggregate |
| ---: | --- | ---: | --- | --- | ---: | --- |
| 1 | 236.7 to 291.1 | 1.230 | 237 to 292 | 227.0 to 273.5 | 1.205 | 226 to 272 |
| 2 | 199.2 to 203.1 | 1.020 | 398 to 405 | 209.4 to 201.0 | 0.960 | 419 to 402 |
| 3 | 164.4 to 180.2 | 1.096 | 495 to 538 | 160.5 to 177.4 | 1.105 | 482 to 530 |
| 4 | 147.9 to 153.2 | 1.036 | 592 to 615 | 141.1 to 156.8 | 1.112 | 569 to 627 |
| 5 | 139.1 to 134.1 | 0.964 | 688 to 671 | 130.3 to 135.7 | 1.041 | 648 to 679 |
| 6 | 109.7 to 118.4 | 1.079 | 659 to 708 | 107.2 to 119.9 | 1.118 | 644 to 716 |
| 7 | 105.8 to 110.8 | 1.048 | 749 to 771 | 103.1 to 112.6 | 1.092 | 720 to 783 |
| 8 | 95.9 to 103.4 | 1.078 | 764 to 822 | 95.3 to 103.6 | 1.087 | 761 to 831 |

R704's aggregate fell from 5 to 6 streams on code (688 to 659) and was flat on prose (648 to 644), where its policy dropped from two draft tokens to one; R719's rises at that step. TTFT is 0.00 to 0.06 s higher in R719 at every shape, with prompts 17 tokens longer and no two streams of a round sharing a prompt.
