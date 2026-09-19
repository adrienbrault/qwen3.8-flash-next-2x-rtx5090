# QSA raw-key ring round 1 — box build, harness, boots, ladder, A/B and tier

Run on flan under the normal GPU queue: source `/srv/qwen5090/lib/gpu-queue.sh` before the flock and chain the steps without restoring the daily in between (`flan/OPERATIONS.md` §12). All output goes under `R=/srv/qwen5090/results/<date>-rNNN-qsa-rawk-ring/`, full raw output, never grep-only. No step below has been run. Probe helpers (`greedy`, `greedy30k`, `served_id`, `cfgline`, `vram`, the decode aggregation) are the ones in `flan/r540-promote-r6.sh`; reuse them.

The change is meant to be bit-exact. The flag-on arm must reproduce the canonical fingerprints; any difference is a failure, not a quality question.

Variables:

```bash
SRC=/srv/qwen5090/overlay-src/qsa-rawk-ring-r1           # rsync of flan/patches/exllamav3/qsa-rawk-ring/r1
LIVE=/srv/qwen5090/launch-flashnext.sh
LIMG=$(sed -n 's/^IMG=\${IMG:-\(.*\)}$/\1/p' "$LIVE")      # tabbyapi:nvme-tier-r4-e3det, or ...-r6 after R540
NIMG="$LIMG-rawkring"
C1=e7fb377c987d685c; C30=4a255910dee2d9c5                # canonical fingerprints (check the live launcher's round)
```

## 1. Build (before the lock)

```bash
cd "$SRC" && sudo docker build -f Dockerfile.box --build-arg BASE="$LIMG" -t "$NIMG" . > "$R/build.log" 2>&1
```

The patch is Python only (four files, no extension rebuild), so one manifest serves the R535 image and the R540 (`-r6`) image: decode-kernels r6 and gdn-state-bf16 r1 touch other files. The build aborts on a patch hash mismatch, a touched file that differs from its pinned baseline, a non-zero `patch -p1 --fuzz=0` exit or a `.rej`, a post-patch hash mismatch, the selector reading anything but `False` with `EXL3_QSA_RAWK_RING` unset, or a missing ring symbol. Keep `build.log`.

## 2. Harness on both cards (lock held, daily down)

```bash
for G in 0 1; do
  sudo docker run --rm --gpus all --entrypoint python3 -v "$R":/out "$NIMG" \
    /opt/qsa-rawk-ring-r1/tests/gpu_rawk_ring.py --device cuda:$G --json /out/harness-gpu$G.json > "$R/harness-gpu$G.log" 2>&1
  echo "gpu$G exit $?"
done
```

Both runs must exit 0. The harness fails (exit 1) on any of:
- `scenarios`: nine generator-shaped schedules with the real Triton kernels (chunked prefill then decode; MTP verify depth 3 with the draft-cache pattern; q_len 16 graph windows with acceptance down to one token and a zero-advance re-run; eager verify windows at the 17-token guard limit; stop-string rewind and replay from the stash page, including a graph-sized replay tail; bsz 3 batches with different start positions; prefix sharing of full pages; mid-block chunk starts; full-width rotary). After every call the ring's pooled plane must be `torch.equal` to the served full plane's, in the graph order (ring append, ring pool) and the eager order (ring pool, ring append).
- `controls`: a ring of 8 rows under q_len 16, and an eager window past the guard, must both mismatch. If either matches, the comparison is blind and the harness fails.
- `update_planes`: the real `QSAIndexer.update_planes`, served branch on a full plane vs ring branch on a ring, over a 5,003-token chunked prefill, 40 MTP windows and 10 q16 windows with rewinds, a second sequence continuing after 12 shared full pages, and bsz 2 decode. Pooled planes and returned indexer queries after every call; at ≥ 8 checkpoints past the sparse threshold, `select_indices_paged` indices and `sparse_attend` outputs, all `torch.equal`. The ring branch must raise on an 18-token verify window.
- `aot`: `_qsa_ring_graph_kernels` (the AOT compile a graph slot performs) for q_len 1/4/16 at rotary widths 64 and 128; q_len 17 must compile (17 + 3 = 20 rows) and q_len 18 must be refused. This is the first compile of the ring kernels as cubins; a failure here would otherwise surface as a boot failure.
- `cache_layer`: `CacheLayer_qsa_quant` with the selector off (`[8, 256, 128]` raw plane) and on (`[8, 20, 128]`), storage difference `8 × 236 × 128 × 2` B, block-aligned `copy_page` copies the pooled rows, a misaligned one raises.

`bench_us` (plane upkeep per call, served vs ring, eager 2,048-token chunk and q1/q4 at bsz 4) is informational.

## 3. Flag-off serving identity (G1)

Candidate launcher: the live launcher with `$LIMG` replaced by `$NIMG` in both places (IMG default and the tier-default condition), exactly as `r540-promote-r6.sh` builds it. With a non-served image name the launcher attaches no NVMe tier, which keeps the daily's tier namespace untouched during the whole experiment.

Boot it with `EXTRA_ENV` unchanged (selector unset) at 819,200. Check `cfgline` (`cache_size: 819200 cache_mode: 8,8 max_batch_size: 4 gpu_split: [30, 30] ... chunk_size: 2048 vision: true`) and the container env. `greedy boot` must equal `$C1` and `greedy30k boot` must equal `$C30`. Record `nvidia-smi --query-gpu=memory.used,memory.free` per card and keep the boot log (layer placement lines). This is the reference for step 4.

## 4. Flag-on identity at 819,200 (G1 for the flag)

Flag-on arm = the step-3 candidate with `EXTRA_ENV="<live env> EXL3_QSA_RAWK_RING=1"`.

1. Boot at 819,200. `greedy` must equal `$C1` and `greedy30k` must equal `$C30`. A difference stops the round.
2. Record per-card used/free VRAM against step 3. Expected drop: 236 B × 819,200 = 184.4 MiB per QSA layer, 13 layers = 2.34 GiB summed over the cards, split by where the 12 main QSA layers and the MTP layer sit. Compare the boot logs' layer placement with step 3; a layer that moved between cards is a confound for the ladder (R489).
3. Long greedy identity, flag on vs the step-3 boot (flag off), both at 819,200, `temperature 0`: (i) the 120k prose haystack of `fn_needle_oai.py` with its question, `max_tokens 512`; (ii) a 240,000-token haystack, `max_tokens 256`; (iii) the R540 c1 code prompt, `max_tokens 2048, min_tokens 2048`. Save the full responses. Each pair must be byte-identical. These exercise thousands of sparse selections, many chunk boundaries and hundreds of MTP rewinds, beyond what the canonical 30k prompt covers. (Run the flag-off halves right after step 3 to save a boot.)

## 5. Pool ladder, flag on

`CACHE=$((819200 + k*16384))` for k = 1, 2, 3, ... until the first boot that fails (`Insufficient VRAM in split for model and cache`, a restart, or no answer). At every step that boots: `cfgline`, per-card free VRAM, `greedy` = `$C1` and `greedy30k` = `$C30` (the pool size does not enter the arithmetic; a difference is a failure), one cold 120,000-token prefill (`fn_bench.py --kind prose --tokens 64 --conc 1 --runs 1 --ctx 120000 --unique`), and one c4 decode round (`fn_bench.py --kind code --tokens 2048 --conc 4 --runs 1`) to prove runtime headroom. `POOL_ON` = the highest step that passes all of these.

Arithmetic expectation (`impl-status.md`, Bytes): equal bytes allow 984,064 tokens, i.e. k = 10 (983,040), if the saving were spread evenly. The two cards' headroom decides the real ceiling: the census (R541) found ~1.2 GiB unused on cuda:0 under a 90k prefill, so the limiting card may reach fewer steps. Record which card fails first. If the ladder stops well short of k = 10 because one card is full while the other has slack, note it as input for a `gpu_split` retune (R541's frontier), not as a failure of this round.

Control, flag off: one boot at `819200 + 16384`. R525 set 819,200 as the flag-off maximum; if this boots and passes the same checks, the ceiling moved and the gain is measured from the new ceiling.

## 6. A/B decode and prefill

Eight boots at 819,200, interleaved OFF/ON/OFF/ON/..., so the pool is the same in both arms. Per boot, the R540 decode block: `fn_bench.py --kind code|prose --tokens 2048 --warmup-runs 1 --conc 1 4 --runs 2`, plus the 30k and 120k cold prefill times from the same `fn_bench.py --ctx` runs. Report per arm and cell the mean and the paired ON−OFF difference with a 95 % CI across the four pairs, as in R538. Expected: neutral (the ring kernels do the same loads and math; the eager pool reads new rows from the staging buffer instead of the plane). Decode regressions beyond −1 % at c1 or c4, or prefill regressions beyond −1 %, fail the round. One flag-on boot at `POOL_ON` repeats the c1/c4 cells; the pool size should not change the decode rate.

## 7. Gates at POOL_ON (flag on)

Bit-exactness makes step 4 the quality gate. At the promoted pool, run:
- G3 needles: `fn_needle_oai.py --ctx-tokens 131072 240000 --fracs 0.08 0.3 0.55 0.8 0.96`; `5/5 retrieved` on both.
- c4 survival: four concurrent 120k-token sessions (`fn_bench.py --kind prose --tokens 256 --conc 4 --runs 1 --ctx 120000 --unique`), no OOM, no restart; this is the case the larger pool exists for.
- G2 agentic-edit (`agentic-edit.py --tag RR --conc 1 4 --modes greedy sampled`), 4 × `6/6 ok`, as a smoke check of the batched and sampled paths.

## 8. NVMe tier compatibility (flag on, scratch directory)

Use a fresh directory so the daily's tier is never opened: `T=/srv/qwen5090/fast/exl3-nvme-rawkring-test`, `NVME_TIER=$T NVME_TIER_GB=16`.

1. Boot flag on with the tier at 819,200. Record the namespace digest and `page image` size from the ` -- nvme tier:` log line (expected about 17 % smaller than the flag-off image: the `raw_k` segment is 5,120 B per page and layer instead of 65,536 B). Send the `greedy30k` prompt; it must equal `$C30`. Wait for the tier to drain on idle.
2. Restart the same arm with the same directory. Send the same prompt. The tier must report restored pages and checkpoints (`restored_pages`, `restored_checkpoints` > 0), the response must equal `$C30`, and the container log must show no `EXL3_QSA_RAWK_RING` error.
3. Send the 30k prompt plus a 2,000-token continuation (a second turn): the first ~30k tokens come from restored/shared full pages, the continuation starts at a page boundary. Save the response; repeat on a flag-off boot with its own scratch directory (step 4's tier-less boot is not enough: it must also restore) and compare, byte-identical.
4. Boot flag off with the step-1 directory. The log must show `stale namespaces removed` ≥ 1 at open (the flag-on namespace is deleted), no restore, and `greedy30k` = `$C30`.
5. Boot flag on again with the same directory: again a new namespace, no hit on the first request.
6. `sudo rm -rf "$T"` and the step-3 flag-off directory.

The namespace separation comes from three served inputs: `engine_identity()` lists `EXL3_QSA_RAWK_RING=1`; the `raw_k` segment's `page_layout` shape is `[20, 128]` instead of `[256, 128]`; `page_image_bytes` differs.

## 9. End of chain

Restore the daily only if no other unit is queued (`daily-restore-retry.sh` / `gpu_queue_others`, OPERATIONS §12). Write `$R/audit.log` with: build result, harness pass/fail per card with the scenario list and `bench_us`, step-3 and step-4 fingerprints, the long greedy identity table, per-card VRAM for both arms, `POOL_ON` and the failing card, the flag-off control, the A/B table, the gates and the tier sequence.

## Promotion notes

A promoted launcher carries `IMG=$NIMG`, `EXL3_QSA_RAWK_RING=1` in `EXTRA_ENV` and `CACHE=$POOL_ON`; the canonical fingerprints stay `$C1`/`$C30`. The image change and the env change both open a new tier namespace, so the daily's tier restarts cold. Rollback = the previous launcher (its tier directory was GC'd by the promoted boot and restarts cold too). The ring caps eager verify windows at 17 tokens (draft depth 16); a deeper MTP experiment needs a larger ring (`rawk_ring_rows`), which costs 16 B per token and layer for every 16 extra rows per page.
