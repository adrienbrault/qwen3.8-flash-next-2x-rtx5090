# R581: `gpu_split` does not move the layer boundary, and 999,424 is the ceiling for this layout

Results directory on the serving host: `results/2026-09-20-r581-split-rebalance`. Raw records: [`2026-09-20-r581-split-rebalance/`](2026-09-20-r581-split-rebalance/). Driver: [`scripts/r581-split-rebalance.sh`](../../scripts/r581-split-rebalance.sh).

## Why free VRAM is lopsided

`gpu_split: [30, 30]` is a per-card budget in GiB for the **weights**, not a layer count. The splitter fills the first card to its budget and spills the rest to the second, which also takes `lm_head` and the 320 MiB pruned draft-embedding mirror. The checkpoint has 48 layers with `full_attention_interval` 4, so the 12 full-attention layers are 3, 7, … 47 — and **KV pages exist only on those**. Whichever card holds more of them pays more of the page pool and sets the ceiling for the whole pool.

Before [R579](r579-promote-mtp-kv-window.md) the live pipeline map read `cuda:0` = layers 0–22 (5 attention layers) and `cuda:1` = layers 23–47 plus the head (7 attention layers), and free VRAM at boot was 2,085 / 867 MiB: the second card bounded the pool while 2 GB sat stranded on the first. R579's windowed draft cache frees draft cache on the first card, the splitter pulls boundary layer 23 — itself an attention layer — across, and the second card gets back both its weights and its share of the pool. Free VRAM became 1,041 / 2,531. The imbalance did not go away, it inverted.

## Does the budget knob move the boundary? No

Three arms, each walking the pool up from the served 999,424, with a 1-to-8-stream decode ramp at every step so all CUDA graphs are captured before a pool is called good.

| split | pool | free at boot | free after the ramp | verdict |
| --- | --- | --- | --- | --- |
| `30, 30` (served) | 999,424 | 1,041 / 2,531 | 251 / 2,013 | ok |
| `30, 30` | 1,032,192 | 821 / 2,311 | **11** / 1,753 | below the floor |
| `29, 31` | 999,424 | 1,041 / 2,531 | 251 / 2,013 | ok |
| `29, 31` | 1,032,192 | 821 / 2,311 | **11** / 1,753 | below the floor |
| `31, 29` | 999,424 | 1,041 / 2,531 | 251 / 2,013 | ok |
| `31, 29` | 1,032,192 | 821 / 2,311 | **11** / 1,753 | below the floor |

Every arm produced identical numbers at the same pool, and the greedy fingerprints (`18238d63065ee16c` at one stream, `4a255910dee2d9c5` at 30k) were the same on all six boots. The budgets are not binding in this range: placement is decided by fit. The only thing that has ever moved the boundary is changing what is resident, which is what the draft cache window did. An earlier sweep of `29, 31` and `29.5, 30.5` on the pre-window layout ([R573](r573-mtp-kv-window-screen.md)) found the same null result.

## Two numbers worth keeping

**The decode graphs cost 790 MiB on the first card and 518 MiB on the second** — free VRAM at boot minus free VRAM after the ramp, at 999,424. That is an upper bound, since it also includes whatever the allocator keeps cached from the ramp itself, but on the card that bounds the pool it is worth about 52k pool tokens if the capture set could be narrowed. Graph memory is a lever, not an accident.

**999,424 is the ceiling for this layout.** One step up ramps clean and ends with 11 MiB free on the first card. [R575](r575-promote-mtp-kv-window.md)'s graph-capture failure at 1,015,808 was not bad luck; it was one step past the wall.

The floor in this round was judged on free VRAM **after** the ramp rather than at boot, which is the headroom that actually has to exist — the boot number does not include the graphs.
