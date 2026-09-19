# R339: gates on the first served image — long-context retrieval, admission of deep contexts, decode at the requeue boundary

Results directory on the serving host: `results/2026-09-16-r339-gates`. Driver: [`scripts/r339-gates.sh`](../../scripts/r339-gates.sh). Date: 2026-09-16.

## Long-context retrieval


One unique passphrase planted at five positions (8 %, 30 %, 55 %, 80 %, 96 %) of a deterministic document, asked
for by name. Requested depths are labels: the filler's estimate runs ~28 % high, so the actual `prompt_tokens` the
server saw is quoted.

| requested depth | prompt tokens seen | retrieved |
| --- | --- | --- |
| 32,768 | 26,518 | **5 / 5** |
| 131,072 | 105,680 | **5 / 5** |
| 196,608 | 158,452 | **5 / 5** |

A pass at every planted position, not only near the end, is what makes this a gate rather than a demonstration.
`bench/needle.py` speaks the OpenAI chat API, so it runs unchanged against any OpenAI-compatible server.

## Admission of deep contexts


Eight concurrent requests, each carrying 38,283 prompt tokens and 512 forced output tokens.

| | value |
| --- | --- |
| admitted / completed | **8 / 8** |
| TTFT | 59.4–62.4 s (all eight within 3 s of each other) |
| decode | 28.5–33.6 t/s per stream, ~256 t/s aggregate |

All eight arrived together and prefilled concurrently, so the ~60 s is what it costs to put 8 × 38k tokens through
this box at once — an effective prefill of ~5,100 t/s aggregate. **Caveat that the first run did not establish:**
those eight requests shared one 38k-token filler and differed only in a short suffix, so this measures the
realistic same-harness fan-out, not independent context footprint against the 262,144-token pool. `r345-pool`
repeats it with a per-request RNG stream so the requests share no token sequence, and captures the server's own
`cached_tokens` figures as evidence rather than inferring sharing from timings.

## Decode at the requeue boundary — 2,048 forced tokens, results `2026-09-16-r339-gates`


2,048 is TabbyAPI's requeue budget on this config (`chunk_size 2048`, `output_chunking: true`), so these rows sit
exactly at the boundary where the engine's own token accounting used to break.

| kind | concurrency | decode per stream (t/s) | aggregate (t/s) | TTFT (s) |
| --- | --- | --- | --- | --- |
| code | 1 | 210.7–210.8 | 207.7–207.8 | 0.145 |
| code | 4 | 64.4 | 252.4 | 0.511 |
| prose | 1 | 162.4–167.2 | 160.6–165.3 | 0.14–0.15 |
