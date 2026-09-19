# R358: The host KV tier is not a lever

Results directory on the serving host: `results/2026-09-16-r358-hostkv`. Driver: [`scripts/r358-hostkv.sh`](../../scripts/r358-hostkv.sh). Date: 2026-09-16.


`sysmem_kv_cache` was 0 in every measurement above; this boots it at 4096 MiB and runs the three shapes that could
plausibly notice, against the same shapes with no tier. The expectation was written down before the run: a host tier
cannot make more than the 262,144-token VRAM pool fit, so it should not change admission; what it could change is
recomputation.

| shape | tier 0 (served) | tier 4096 MiB |
| --- | --- | --- |
| 152,761-token prompt, first send | decode 150.1, TTFT **32.36 s** | decode 150.9, TTFT **32.52 s** |
| the same prompt again | decode 161.4, TTFT **0.431 s** | decode 161.0, TTFT **0.441 s** |
| 8 × unique ~40k prompts at once | **8/8**, 48.4 agg, TTFT 17.7–116.1 s | **8/8**, 48.4 agg, TTFT 18.5–113.0 s |
| 4 × shared 152,761-token prompts | 51.4 per stream, 190.7 agg, TTFT 1.58 s | 48.8 per stream, 179.4 agg, TTFT 1.74 s |

Nothing moves. The reused prefix already stays in VRAM (0.43 s on the repeat, either way), and the pool is never
spilled to host under these shapes. **A negative result: the last configuration knob that could have addressed the
deep-context weakness does not.** The launcher keeps `SYS_KV` as a knob and defaults it to 0.
