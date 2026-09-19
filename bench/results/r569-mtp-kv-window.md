# R569: a windowed draft cache frees the card that limits the page pool

Results directory on the serving host: `results/2026-09-19-r569-mtp-kv-window`. Raw records: [`2026-09-19-r569-mtp-kv-window/`](2026-09-19-r569-mtp-kv-window/). Driver: [`scripts/r569-mtp-kv-window.sh`](../../scripts/r569-mtp-kv-window.sh).

The MTP draft layer keeps a KV cache as long as the whole context. `EXL3_MTP_KV_WINDOW=16384` gives it a per-slot ring instead — one sink page plus 64 pages — which the container log states at boot: `draft cache 8 slots x 65 pages = 133120 tokens`.

| boot | free VRAM at boot, MiB | 1-stream fingerprint |
| --- | --- | --- |
| window off @ 966,656 | 2,085 / 867 | `f4add302e176d78e` |
| window 16,384 @ 966,656 | 1,241 / 2,751 | `18238d63065ee16c` |

The window frees about 930 MiB on the first card, and the loader then places one more layer there. That is what matters: the second card, which binds the page pool, gains 1,884 MiB. The fingerprint changes because a different layer split changes which positions are verified together, and greedy text on this engine depends on that layout.

Pool ladder with the window on, each step surviving a cold ~90k-token prefill and an 8-stream round:

| pool | free VRAM, MiB | 8-stream aggregate t/s |
| --- | --- | --- |
| 966,656 | 1,241 / 2,751 | 622 |
| 983,040 | 1,121 / 2,671 | 633 |
| 999,424 | 1,041 / 2,531 | 628 |
| 1,015,808 | 901 / 2,431 | 641 |
| 1,032,192 | 821 / 2,311 | below the floor on the first card |

Three steps, +5.1 %. The floor is the tightest served card's free VRAM minus 32 MiB, applied to both cards. Acceptance past the window's span, needles and the layer split were measured next in [R573](r573-mtp-kv-window-screen.md).
