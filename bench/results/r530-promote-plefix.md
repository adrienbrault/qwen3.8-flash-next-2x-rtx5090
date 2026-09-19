# R530: ExLlamaV3's PLE checkpoints no longer alias the live slot; every gate passes

Results directory on the serving host: `results/2026-09-19-r530-promote-plefix`. Raw records: [`2026-09-19-r530-promote-plefix/`](2026-09-19-r530-promote-plefix/). Driver: [`scripts/r530-promote-plefix.sh`](../../scripts/r530-promote-plefix.sh). Date: 2026-09-19.

ExLlamaV3's `PLELayerState.stash()` (`exllamav3/modules/ple.py`, also on upstream master) returns `self.id_state[slot, :self.ctx].cpu()`. `id_state` is allocated on the host, so `.cpu()` returns the tensor itself: every recurrent checkpoint stored for a PLE layer holds a view of the live slot, not a copy. Later prefill and decode writes to that slot, and the end-of-sequence fill when the next job starts there, change the checkpoint after it was stored. A job that resumes from it gets the wrong token-id context for its first `ctx` positions. The float half of the state (`conv_state`) is copied off the GPU and is unaffected.

It was found while testing the recurrent-tip checkpoint round (R524, 2026-09-19): a stored decode checkpoint's float state was bit-identical to a fresh one, but its PLE id pair changed from [579, 264] to [271, 4277] after it was stored. It also made every PLE checkpoint's digest in the NVMe tier round (R526) mismatch on restore.

The overlay [`docker/overlays/ple-ckpt-clone-r1/`](../../docker/overlays/ple-ckpt-clone-r1/) replaces `.cpu()` with `.clone()` on that one line. `fix_ple.py` checks the SHA-256 of `ple.py` before and after the edit; `test_ple_stash.py` runs in the build and fails on the unpatched file (the stored window reads `[99, 99, 99]` after the slot is overwritten).

## Promotion gates (promoted 04:18 UTC, 06:18 CEST)

Built on the served image: `tabbyapi:mtp-pruned-r1-tc1-plefix` from `tabbyapi:mtp-pruned-r1-tc1`; launcher unchanged except the image. Rollback: the R529 launcher.

| gate | result |
| --- | --- |
| fingerprints | c1 `e7fb377c987d685c`, 30k `4a255910dee2d9c5`: equal to the daily before it |
| cold prefill, salted, one invocation per length | `fn_bench --ctx` 60k target (45,163 prompt tokens): 9,354 t/s; 120k target (90,040): 10,015 t/s ([R528](r528-promote-mtp-pruned.md): 9,841 and 10,278) |
| decode (one boot, against the [R522](r522-mtp-pruned.md) figures) | code c1 223.5 vs 221.8 t/s, c4 510.3 vs 508.4; prose c1 197.4 vs 194.5, c4 482.1 vs 484.4 |
| agentic-edit | 6/6 greedy at c1 and c4 |
| needles | 5/5 at 131k and 5/5 at 240k |
| tool-eval 69×4 | 84.8 ± 1.3 (trials 118 / 114 / 117 / 117); TC-45 2 points on all 4 trials |
| GSM8K 5-shot n=500, no stop strings | 0.974 |

## Tool-eval between boots

R529 scored 88.0 ± 1.6 on the configuration without this fix; R530 scored 84.8 ± 1.3. The fix changes only what a stored PLE checkpoint holds, and outputs on fresh prompts are byte-identical (fingerprints above). Summed over the four trials, R528 scored 467 points, R529 485 and R530 466. The scenarios that differ are ones whose points already change between trials of one boot:

| scenario | R528 | R529 | R530 |
| --- | --- | --- | --- |
| TC-30 | 0 0 0 2 | 2 0 0 2 | 0 2 0 0 |
| TC-42 | 0 2 2 0 | 2 2 2 2 | 0 0 0 2 |
| TC-45 (`tool_choice=required`) | 0 0 0 0 | 2 2 2 2 | 2 2 2 2 |
| TC-49 | 1 2 1 1 | 2 2 2 2 | 2 1 1 1 |
| TC-58 | 0 0 2 0 | 0 0 2 2 | 0 0 0 0 |
| TC-68 | 2 0 2 0 | 2 2 2 0 | 2 0 0 0 |

The ± in the tool-eval figures is the spread of four trials on one boot and understates the spread between boots: the last six configurations read 84.5 to 88.0. Of R529's gain, TC-45's 8 points (2.9 on the 69-scenario score) are the `tool_choice` enforcement; the rest was that boot.
