# R552b: the 4-stream code requests reach the forced 2,048 tokens when streamed through the probe

Results directory on the serving host: `results/2026-09-19-r552b-c4-stream-count`. Raw records: [`2026-09-19-r552b-c4-stream-count/`](2026-09-19-r552b-c4-stream-count/). Driver: [`scripts/r552b-c4-stream-count.sh`](../../scripts/r552b-c4-stream-count.sh). Date: 2026-09-19.

Configuration: the served configuration of [R548](r548-promote-gdnbf16-ring.md) (4 slots, 1,032,192-token pool, NVMe tier off). In the code rows at 4 streams of R538, R540 and R546, two of the four requests reported 1,861 tokens although `min_tokens` was 2,048. R552 (non-streamed, same prompt) showed 14 of 14 requests ending with `finish_reason: length` at 2,048 tokens and TabbyAPI returning `usage: null` on non-streamed chat completions. R552b sends the requests through `fn_bench`'s own streamed request function and records the server's token count next to the client's frame count.

- 3 rounds of 4 code requests and 1 request alone: 13 of 13 end with `finish_reason: length` at 2,048 server tokens, carried in 692 to 798 streamed frames.
- The 1,861 does not occur on this image, streamed or not. The earlier round records store only `tokens_median` (1,954.5, i.e. two requests at 1,861), not which counter produced it, so the code aggregates at 4 streams of R538, R540 and R546 may understate throughput by up to 4.6 %.
