# R568: the upstream rebase's prefill gain is one feature, worth +14 % at 60k

Results directory on the serving host: `results/2026-09-19-r568-rebase-prefill`. Raw records: [`2026-09-19-r568-rebase-prefill/`](2026-09-19-r568-rebase-prefill/). Driver: [`scripts/r568-rebase-prefill.sh`](../../scripts/r568-rebase-prefill.sh).

Porting the served stack onto upstream ExLlamaV3 `dev` costs page pool and decode ([R563](r563-rebase-dev.md)), but its cold prefill read higher. This isolates why: upstream's tiled hyper-connection prefill mix ([`825db5b`](https://github.com/turboderp-org/exllamav3/commit/825db5b), env `EXL3_GR_MIX_TILED`) on and off, against the served image.

Three salted cold prefills per arm and length, prompt tokens per second:

| arm | 60k | 120k |
| --- | --- | --- |
| served image @ 966,656 | 9,537 | 10,088 |
| rebase, tiled mix on @ 917,504 | 10,872 (**1.140×**) | 11,242 (**1.114×**) |
| rebase, tiled mix off @ 917,504 | 9,564 (1.003×) | 9,949 (0.986×) |

With the feature off the rebase matches the served image, so the whole prefill difference is that one commit. Its cost upstream is 302 MiB of free VRAM per card — about two pool steps on the card that limits the pool — which is why it is worth porting on its own terms rather than taking the rebase.
