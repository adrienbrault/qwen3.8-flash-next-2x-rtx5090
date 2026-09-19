# R367: The slot ladder and 12-agent admission

Results directory on the serving host: `results/2026-09-16-r367-slots`. Driver: [`scripts/r367-slots.sh`](../../scripts/r367-slots.sh). Date: 2026-09-16.


`max_batch_size` is the last untested *config* lever on the weakest axis. TabbyAPI derives 4 slots for a recurrent
model and this seat is served at 8, so the arms below measure whether more slots admit more real work.

**More slots are not reachable here at all.** Both arms above the served value fail to boot:

```
max_batch_size 12 -> RuntimeError: Insufficient VRAM in split for model and cache
max_batch_size 16 -> RuntimeError: Insufficient VRAM in split for model and cache
```

That is the same wall the 393,216-token cache hits, and it has the same cause: the manual layer split has to hold the
weights *and* the whole cache, and the recurrent-state planes grow with the slot count. The ladder therefore does not
rank 8 against 12 or 16: **8 is the maximum this cache size supports**, and any future argument for more slots has to
be an argument for a smaller cache.

With the served 8 slots, offering more concurrency buys nothing (the probe's own aggregate, 512 forced tokens, two runs
each):

| offered concurrency | aggregate t/s | TTFT |
| --- | --- | --- |
| c1 | 163.7 / 167.5 | 0.15 s |
| c4 | 313.0 / 315.1 | 0.46 s |
| c8 | 301.3 / 320.7 | 0.74–0.79 s |
| c12 | 309.6 / 319.0 | 0.97–1.01 s |
| c16 | 314.7 / 313.5 | **7.21 s** |
| 12 agents at ~25.6k prompt tokens | **97.7** | **42.99 s** |

Two readings. **The aggregate is flat from c4 to c16** — roughly 310–320 t/s, which is the ceiling of eight slots at
about 40 t/s each, not a limit the offered concurrency can push. Above 8 the requests queue rather than parallelise,
and c16 pays for it in TTFT: 7.2 s, two waves. **And real agent contexts collapse it**: twelve jobs at ~25k tokens
each return 97.7 t/s aggregate with a 43-second TTFT, which is what "admission" means here — the box does not host
twelve deep agents, it queues them.

The c1 column reads 163.7 against the 207 t/s this repository quotes elsewhere; that is `docs/GOTCHAS.md` #9
(content and generation-length dependence), not a regression: 512 forced tokens of code versus 2,048.
