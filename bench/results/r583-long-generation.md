# R583: generation length costs nothing, and the draft-cache window is not the reason real sessions are slower

Results directory on the serving host: `results/2026-09-20-r583-long-generation`. Raw records: [`2026-09-20-r583-long-generation/`](2026-09-20-r583-long-generation/). Driver: [`scripts/r583-long-generation.sh`](../../scripts/r583-long-generation.sh).

## Why this round exists

Every decode number in this repository is measured with a forced output of 512 or 1,024 tokens, a freshly salted prompt, greedy sampling, and all streams starting at the same instant. A real agent session does none of those things, and on 2026-09-20 one was measured from the server's own per-request log — 326 requests, 247,544 tokens, 45 minutes, three concurrent agents. It delivered **65.6 tokens/s per stream and 134.4 aggregate** where the table in the README says 213 and 627.

The shape of that session pointed somewhere specific. Nineteen requests running under 50 tokens/s consumed **59 % of all decode seconds** while producing 21 % of the tokens, and the slowest were 2,700–4,000-token generations at 13–20 tokens/s. The two worst sat at only ~10.7k context, so context depth was not the variable. That left generation length, and the newest thing in the serving path was [R579](r579-promote-mtp-kv-window.md)'s windowed MTP draft cache, whose gates — needles, GSM8K, tool-eval and a short decode A/B — are all short-output tests that a length-dependent decay would pass untouched.

This round tests both at once.

## Method

One boot per arm, NVMe tier off, greedy, prompts salted per request so no arm is served another's cached pages.

- **A** — the served daily: windowed draft cache (`EXL3_MTP_KV_WINDOW=16384`), 999,424-token page pool.
- **B** — the previous daily: no window, 966,656-token pool.

The two differ in pool as well as window because the window is what bought the pool step; that is the promotion as it actually happened. Each arm walks 512, 1,024, 2,048 and 3,000 forced tokens at two depths, two runs each, then runs three synchronized streams at 3,000 tokens.

## Generation length is free

Per-request decode tokens/s, one stream:

| forced tokens | prompt tokens | A, window on | B, window off | B/A |
| --- | --- | --- | --- | --- |
| 512 | 9,816 | 279.8 | 292.4 | 1.04 |
| 1,024 | 9,811 | 300.5 | 291.8 | 0.97 |
| 2,048 | 9,786 | 261.4 | 250.3 | 0.96 |
| 3,000 | 9,839 | 284.6 | 278.4 | 0.98 |
| 512 | 49,391 | 281.1 | 282.7 | 1.01 |
| 1,024 | 49,713 | 201.4 | 190.1 | 0.94 |
| 2,048 | 49,689 | 208.8 | 225.1 | 1.08 |
| 3,000 | 49,755 | 209.0 | 205.4 | 0.98 |

At ~9.8k context a 3,000-token generation runs at **1.02×** the rate of a 512-token one. At ~49.7k the rate settles around 205 by 1,024 tokens and stays there; the 512-token row reads high because a short generation amortises the warm first steps over fewer tokens, not because a long one decays.

Three synchronized streams at 3,000 tokens and ~9.8k context: **A 171.7 tokens/s per stream, B 194.0** — the one place the arms separate, by 13 %.

## Both suspects are cleared

The B/A column sits between 0.94 and 1.08 across eight shapes. That is the run-to-run band, and the window does not cost anything that shows up here. The 13 % at three streams is real but is not the effect that has to be explained, and it is not a reason to give back the pool step the window paid for. **R579 stays.**

## What that leaves

The numbers to reconcile, all greedy, all at ~10k context and 3,000 generated tokens:

| condition | tokens/s per stream |
| --- | --- |
| one stream, idle server | 283 |
| three streams started together | 172–194 |
| one stream, fired alongside a real three-agent session | 88.9 |
| the real session's own requests | 65.6 |

The third row is the important one: it was measured with the same greedy probe as the first two, so sampling cannot be what separates 181 from 89. What differs is when the other work arrives. This benchmark starts every stream at the same moment, so they prefill together and then decode as a steady batch. Real agents arrive staggered — one request spends 150–220 seconds inside a long generation while another submits a fresh 30–60k prompt, and that prompt's chunked prefill interleaves with the first one's decode steps. Each interleaved chunk is a forward pass the decoder does not get.

That is the next measurement, and until it lands the published decode figures should be read for what they are: a steady-state batch on an otherwise idle server.
