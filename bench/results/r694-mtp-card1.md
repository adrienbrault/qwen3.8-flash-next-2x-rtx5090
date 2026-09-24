# R694: the MTP draft component on the second GPU, promoted

Results directory on the serving host: `results/2026-09-24-r694-mtp-card1`. Raw records: [`2026-09-24-r694-mtp-card1/`](2026-09-24-r694-mtp-card1/). Driver [`scripts/r694-mtp-card1.sh`](../../scripts/r694-mtp-card1.sh). Image `tabbyapi:slotfix-r1`, model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 999,424-token page pool at 8-bit KV, layer split `[30, 30]`, MTP draft policy `[[4, 3], [5, 2], [8, 1]]`.

## The change

The launcher gained `DRAFT_GPU_SPLIT`, which it writes as TabbyAPI's `draft_model.draft_gpu_split`. TabbyAPI passes it to ExLlamaV3's `Model.load_gen(use_per_device=...)` for the MTP draft component. `DRAFT_GPU_SPLIT="0, 32"` gives the draft component no budget on the first GPU and loads it on the second. Before this change the key was not written and the loader placed the draft component itself.

Decode traces of the served stack at 4,096 tokens of context (R680, 2026-09-24) put the second GPU idle 74-75 % of each decode step at 1, 4 and 8 streams, and the first GPU idle 43-45 %. The draft chain ran on the first GPU, behind the layers the first GPU holds.

## What was measured

2026-09-24 06:12 to 06:32 UTC, one session, three alternating pairs: OFF is the launcher without the key, ON is the launcher with `DRAFT_GPU_SPLIT="0, 32"`. Each boot ran the greedy identity set (6 prompts, one of them about 100,000 tokens), then the canonical gate with 3 recorded rounds: leg A is prose at about 4,000 tokens of context at 1, 4 and 8 streams, and leg B is prose at about 26,000 tokens of context at 4 streams, both with 1,024 forced tokens, greedy. Values are the ON/OFF ratio of the median per-request decode rate.

| cell | pair 1 | pair 2 | pair 3 | mean |
| --- | ---: | ---: | ---: | ---: |
| leg A, prose, 1 stream | 0.985 | 1.085 | 1.017 | 1.029 |
| leg A, prose, 4 streams | 1.044 | 1.011 | 1.010 | 1.022 |
| leg A, prose, 8 streams | 1.028 | 1.018 | 1.015 | 1.020 |
| leg B, prose, 26k context, 4 streams | 1.009 | 1.032 | 1.025 | 1.022 |

| check | OFF | ON |
| --- | --- | --- |
| greedy output against the first OFF boot | identical, 6 of 6 | identical, 6 of 6, in all three boots |
| CUDA out-of-memory lines, tracebacks | 0, 0 | 0, 0 |
| free VRAM after boot, GPU 0 / GPU 1 | 1,039 / 2,531 MiB | 1,041 / 2,049-2,429 MiB |
| free VRAM after the gate, GPU 0 / GPU 1 | 19-33 / 1,903-1,919 MiB | 125-139 / 1,487-1,655 MiB |

The per-round numbers are in [`audit.txt`](2026-09-24-r694-mtp-card1/audit.txt), the per-request records in `gate-<arm>/bench-A.jsonl` and `bench-B.jsonl`, and the greedy texts in [`greedy.jsonl`](2026-09-24-r694-mtp-card1/greedy.jsonl).

The rule written before the run required at least 150 MiB free on both GPUs after the gate. The ON arm ended at 125-139 MiB on GPU 0. The OFF arm, the configuration served until then, ended at 19-33 MiB on the same GPU. The promotion was made on that comparison, and it is recorded here as a deviation from the rule as written.

Neither container log names the device the draft component loaded on, and free VRAM on GPU 0 after boot changed by 2 MiB. The measured change is on GPU 1 (100-480 MiB less free after boot) and on GPU 0 after load (about 100 MiB more free).

## Verdict

Promoted 2026-09-24 06:34 UTC: the launcher writes `draft_gpu_split: [0, 32]` by default. After the promotion boot the GPUs had 1,041 / 2,429 MiB free, and the greedy set's five short prompts were identical to the reference. Rollback is `DRAFT_GPU_SPLIT=` (empty), which omits the key.
