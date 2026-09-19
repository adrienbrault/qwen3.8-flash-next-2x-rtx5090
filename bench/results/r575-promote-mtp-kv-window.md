# R575: the windowed draft cache passed its A/B and four gates, then rolled back on a graph-capture OOM

Results directory on the serving host: `results/2026-09-19-r575-promote-mtp-kv-window`. Raw records: [`2026-09-19-r575-promote-mtp-kv-window/`](2026-09-19-r575-promote-mtp-kv-window/). Driver: [`scripts/r575-promote-mtp-kv-window.sh`](../../scripts/r575-promote-mtp-kv-window.sh). Overlay: [`docker/overlays/mtp-kv-window-r2`](../../docker/overlays/mtp-kv-window-r2).

## What was tried

Four launcher lines: the image gains the `mtp-kv-window` overlay, `EXL3_MTP_KV_WINDOW=16384` joins `EXTRA_ENV`, and `cache_size` goes 966,656 → **1,015,808 (+5.1 %)**. Slots, KV bits, layer split, chunk and draft policy are unchanged.

The MTP draft layer no longer keeps a full-length KV cache. Each slot gets a sink page plus 64 pages, `8 slots x 65 pages = 133120 tokens` in the boot log. That frees about 930 MiB on the first card; the loader then moves one layer there, and the second card — the one that bounds the page pool — gains 1,884 MiB. [R569](r569-mtp-kv-window.md) laddered the pool, [R573](r573-mtp-kv-window-screen.md) measured acceptance at depth, needles and the layer split.

## Boots (NVMe tier off except G1)

| boot | free VRAM at boot, MiB | fingerprints 1 stream / 30k | cold prefill 60k / 120k, t/s |
| --- | --- | --- | --- |
| A1 served @ 966,656 | 2,085 / 867 | `f4add302e176d78e` / `4a255910dee2d9c5` | 9,416 / 10,274 |
| B2 candidate @ 1,015,808 | 901 / 2,431 | `18238d63065ee16c` / `4a255910dee2d9c5` | 9,880 / 10,660 |
| B3 candidate @ 1,015,808 | 901 / 2,431 | same | 9,963 / 10,397 |
| A4 served @ 966,656 | 2,085 / 867 | canonical | 9,871 / 10,050 |

The 1-stream fingerprint changes because moving a layer changes which positions are verified together; the 30k fingerprint does not. Both candidate boots agree with each other, which is the identity check this round can make. Free VRAM stays above the 835 MiB floor on both cards, and mean prefill rises: 1.029× at 60k, 1.036× at 120k.

## Decode, paired over 24 prompts per cell

| shape | served | candidate | paired |
| --- | --- | --- | --- |
| code, 1 stream | 202.2 t/s | 203.6 | +0.52 % [−2.40, +3.40] |
| code, 4 streams | 127.7 | 129.7 | +1.61 % [−1.02, +4.36] |
| prose, 1 stream | 200.9 | 201.3 | +0.22 % [−1.42, +1.95] |
| prose, 4 streams | 119.9 | 121.5 | +1.32 % [−0.58, +3.53] |

## Gates (NVMe tier on, live daily)

| gate | result |
| --- | --- |
| GT restart restore | a 30,001-token prompt after a container restart: 29,952 tokens served from the tier in 0.59 s, identical output to the warm (0.36 s) and cold (4.08 s) runs; 6/6 checkpoints intact |
| G2 agentic-edit | 6/6 in four modes |
| G3 needles | 5/5 at ~131k and 5/5 at ~240k prompt tokens |
| G4 tool-eval 69×4 | 86.2 ± 2.9 (95 % interval 83.8 to 89.0) |
| G5 GSM8K 5-shot n=500, no stop strings | **FAILED**: the engine aborted with `GPU assert: out of memory exllamav3_ext/graph.cu 51` after about 1,090 requests |

Promoted 2026-09-19 21:21 UTC, rolled back automatically at 21:41 UTC when G5 failed. The daily is the R565 configuration again: 966,656 tokens, no window.

## What the failure means

GSM8K runs 5 requests at a time. Nothing before it in this round did: the A/B measured 1 and 4 streams, the gates ran 1 and 4, and the ladder in [R569](r569-mtp-kv-window.md) ran 8. A batch size that has not been seen yet has no captured CUDA graph, and capturing one needs free VRAM at that moment — 901 MiB on cuda:0 was not enough.

So the window's own numbers stand, and the pool it allows does not: 1,015,808 tokens clears the 835 MiB boot floor on both cards but leaves too little for a graph shape captured later under load. The floor rule (a card's free VRAM at boot, minus 32 MiB against the served configuration) does not cover graph capture at batch sizes the round never ran. The next attempt ladders with a full 1-to-8-stream ramp at each pool, so every decode graph is captured before the gates, and starts a step lower.
