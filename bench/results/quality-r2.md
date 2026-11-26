# GSM8K and substring retrieval on ExLlamaV3 v1.6.7 r2

Measured 2026-11-26, source `20261126T021547Z-578452-quality-r2-e`, on image `96ab0d09` and launcher `e7ffd0a6`. [Numeric outcomes and validation receipt](2026-11-26-quality-r2/)

| measurement | conditions | outcome |
| --- | --- | --- |
| GSM8K | 500 documents, five-shot, c8, T=0, max 8,192 tokens, no stop strings | 0.978 strict match, 0.980 flexible extract (498 stop and 2 length finishes) |
| substring retrieval | actual prompt 105,680 tokens, five positions, c1, greedy, 64-token cap | 5/5 hits, all length finishes |
| substring retrieval | actual prompt 193,464 tokens, same protocol | 5/5 hits, all length finishes |

The packet compares both GSM8K filters against the 500 audited API responses; the files retain filter outcomes and hashes. Retrieval uses needle positions 0.08, 0.30, 0.55, 0.80 and 0.96 of the context filler and records the actual prompt length and hit flag for each request.
