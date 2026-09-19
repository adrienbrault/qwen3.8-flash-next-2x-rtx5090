# R563: the served stack on upstream ExLlamaV3 `dev` costs pool and decode

Results directory on the serving host: `results/2026-09-19-r563-rebase-dev-try4`. Driver: [`scripts/r563-rebase-dev.sh`](../../scripts/r563-rebase-dev.sh).

Every served patch was ported onto upstream `dev` (`1d64111`) and built for sm_120. The result does not replace the served image.

- **Page pool.** The rebase does not boot at the served 966,656 tokens with 8 slots (`Insufficient VRAM in split for model and cache`). Its ladder tops out at 917,504, **−49,152 tokens**, with free VRAM at boot 1,943 / 773 MiB against the served 2,085 / 867.
- **Decode**, paired over 24 prompts per cell: prose at 1 stream −2.86 % [−5.06, −0.62], code at 4 streams −2.72 % [−4.82, −0.79], code at 1 stream −0.35 %, prose at 4 streams −0.48 %. Eight streams: 639 against 649 t/s.
- **Prefill** reads higher at 120k (11,112 against 10,156 t/s). [R568](r568-rebase-prefill.md) isolates that to one upstream commit, which is worth porting on its own.
- **Upstream's fused shared expert** loses 1.5–5 % against the side-stream overlap this stack already runs.

Unchanged from the served tree: every served flag is present in the ported source, so the missing VRAM is upstream `dev`'s own at load time.
