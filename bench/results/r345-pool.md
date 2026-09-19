# R345: The pool, measured with independent contexts

Results directory on the serving host: `results/2026-09-16-r345-pool`. Raw records: [`2026-09-16-r345-pool/`](2026-09-16-r345-pool/). Driver: [`scripts/r345-pool.sh`](../../scripts/r345-pool.sh). Date: 2026-09-16.


Three arms; the third failed, and its failure is recorded below. Requested depths are labels: `probe.py` records the
`prompt_tokens` the server actually saw, and its filler estimate runs high.

| arm | prompts actually sent | jobs | admitted | TTFT (s) | decode/stream |
| --- | --- | --- | --- | --- | --- |
| shared prefix | 38,283 tokens each (306k total, page-shared) | 8 | **8/8** | 63.6 | 33.2 |
| **unique contexts** | **78,233–79,139 tokens each (~628k total, no page sharing)** | 8 | **8/8** | 43.1–82.2 (median 43.8) | 37.0 |
| unique, deeper | rejected before admission | 4 | 0/4 | — | — |

So eight agents carrying **~628k tokens of mutually unrelated context** — more than twice the 262,144-token pool —
are all admitted and all complete, with the surplus queued rather than refused: the first token arrives in 43 s for
some jobs and 82 s for others, which is the queue draining. The pool schedules; it does not reject.

The third arm is the guard, not a failure: the server answered `400 Prompt length 315,253 exceeds the available
context size of 262,144 tokens` for each request. **That run had a bug in the instrument** — `--unique` prepended
the per-request passage to the shared filler instead of replacing it, so a "120k" request carried 315k tokens. It
is fixed in `bench/probe.py`; the 400 is the server behaving correctly, and it is recorded because a request that
does not fit is refused with a reason rather than silently truncated.

`r345` also captured the server's own log into the results directory, so the cache and length figures above come
from the server rather than from timings.
