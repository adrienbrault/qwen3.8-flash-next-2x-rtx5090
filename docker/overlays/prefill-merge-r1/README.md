# prefill-merge-r1: the sub-page leftover of a prefill runs in the last forward, and recurrent stashes are copied to the host asynchronously

`tabbyapi:merge-tok-r1` is `tabbyapi:tokenize-offloop-r2` plus [`fix.patch`](fix.patch) on the installed ExLlamaV3 package: `cache/prefill_merge.py` (new), `generator/job.py`, `generator/disk_cache.py`, `modules/gated_delta_net.py` and `modules/ple.py`. Python only, no extension rebuild; TabbyAPI, the decode kernels and the page pool are unchanged. Both changes are behind environment keys that are unset in the image; the served launcher sets both (41 keys). Served since 2026-09-29 22:02 CEST ([R809p](../../../bench/results/r803-r810-prefill-merge.md)).

```sh
B=tabbyapi:tokenize-offloop-r2
docker build -f Dockerfile.box --build-arg BASE=$B --build-arg BASE_ID=$(docker image inspect $B --format '{{.Id}}') \
  --label local.prefillmerge.patch_sha256=$(sha256sum < fix.patch | cut -c1-64) -t tabbyapi:merge-tok-r1 .
```

The label is `a5102079e91099c5b457b0b9cffa956fa9e7c5f8ad869e12e611ff07c1f2adda` for the file in this directory, the value on the served image.

**Build order.** The served image (`ac16920f72cf` on the serving host) was built in the other order: `fix.patch` on `tabbyapi:rebase-dev-r3-loopthink5` first, then `tokenize-offloop-r2`'s two patches on that result, with the same hash checks. The two overlays touch disjoint files (this one five ExLlamaV3 files; `tokenize-offloop-r2` ExLlamaV3's `tokenizer/tokenizer.py` and four TabbyAPI files), so every installed file hashes the same in both orders: `SHA256SUMS.src` here and `tokenize-offloop-r2`'s `SHA256SUMS.*.src`, which this build checks after stacking. The image ID that [`../../build-chain.sh`](../../build-chain.sh) produces differs from the served one. This file stacks on `tokenize-offloop-r2` so that the chain stays one line.

## Why

[R802](../../../bench/results/r803-r810-prefill-merge.md) (2026-09-29, a solo prefill ladder on the served configuration, results `2026-09-29-r802-s23-prefill-ladder-0929`) measured about 40 % of a 2,807-token agent turn's prefill in fixed costs: a separate forward for the rows after the last full 256-token page, 7 to 120 ms depending on its size, and about 20 ms of synchronous recurrent-state copy to the host per last-page stash. [R804](../../../bench/results/r803-r810-prefill-merge.md) put one stash at 8 to 10 ms (73 pageable device-to-host copies, 57 MiB) in a profile, paid twice per agent turn.

## `EXL3_PREFILL_MERGE=1`: the leftover runs in the last forward

**The served rule.** `Job.prefill` runs one forward of up to 2,048 rows per generator iteration. For a model with recurrent layers (Gated-DeltaNet and the PLE layer here), the forward that reaches the last full page of the prompt (`last_page_b`, a multiple of 256 below the prompt end) is cut there, so the recurrent state at that page boundary can be stashed and reused by the next turn; the rows after it (N_p mod 256, N_p = prompt tokens − 1) are one more forward. With A the rows up to `last_page_b`: forwards = ceil(A / 2048) + [N_p mod 256 > 0].

**With the key.** When the remainder of the prompt fits one forward (at most 2,048 rows) and a page boundary lies strictly inside that forward, the forward runs to the prompt end, and every recurrent layer splits itself at the page boundary inside it:

- Gated-DeltaNet: the served conv (`causal_conv1d_update`) and delta rule (`gated_delta_rule_fn`, `_chunked_scan`) run on rows [0, T_s) and store the slot as a forward ending at T_s stores it (the fp32 final state rounded to bf16 by the served store); the slot is copied into a pinned staging slab on the current stream; the same two functions run on rows [T_s, T) from the slot; the two outputs are concatenated.
- PLE: the n-gram embedding is computed once for the whole forward, `forward_streams` runs per part, and the conv and id-context state are stored and captured between the parts.
- After the forward the Job stores the captured state as the last-page stash, with the checks of the served `maybe_stash_recurrent`.

Forwards = ceil(N_p / 2048), none larger than 2,048 rows. The set of stash positions is the one the served rule produces (`tests/test_partition.py`, every N from 2 to 8,193 at 0 and 27,136 cached tokens). Serial forwards per prompt (warm, the prefill pipeline as served):

| N (N_p = N − 1) | served | merged |
|---|---|---|
| 64, 257, 513, 2,049, 4,097, 6,145 | 1 to 3 forwards, no cut | unchanged |
| 258 | 256 + 1 | 257 |
| 757 | 512 + 244 | 756 |
| 2,050 to 2,304 | 2,048 + r (r < 256, no page inside) | unchanged |
| 2,807 | 2,048 + 512 + 246 | 2,048 + 758 |
| 4,095 | 2,048 + 1,792 + 254 | 2,048 + 2,046 |
| 4,400 | 2,048 + 2,048 + 256 + 47 | 2,048 + 2,048 + 303 |
| 8,192 / 12,288 | pipelined window(s) + 2,048 + 1,792 + 255 | window(s) + 2,048 + 2,047 |

Not merged, the served path runs: more than one sequence in the job, embeddings or multimodal input, a forward inside the two-card prefill pipeline's window, a tensor-parallel load, a recurrent layer state other than GDN or PLE, no free staging slab. A remainder of 2,049 to 2,303 rows keeps two forwards (r1 does not rebalance it). If a layer did not capture its state, the stash is skipped, the key turns itself off for the process and the log says so once; that turn's output is unaffected.

**What changes numerically.** Given identical inputs, every byte of the captured stash equals what a forward ending at T_s writes, because the split calls the served functions on the same rows. The inputs are not identical: every layer above the recurrent ones sees T rows instead of T_s, and kernels are chosen by row count (the grouped MoE prefill path runs from 512 rows, the GEMMs, the QSA attention, the tiled hyper-connection mix and the Triton conv each pick by M). So the merged turn's output, and the stash the next turn reuses, move by row-count rounding. R803's GPU gate measured the stash against the served one at the page boundary: relative L2 0.0004 to 0.0034 at the first GDN layer, growing smoothly with depth to 0.058 to 0.126 over the whole stash. The served path prefilled with 256-row chunks instead of 2,048 (another valid partition of the same prompt, no merge code) moves the whole stash by 0.06 to 0.14 ([GOTCHAS 37](../../../docs/GOTCHAS.md)). R809 then gated the output against that partition null on 80 agent-shaped prompts.

## `EXL3_STASH_ASYNC=1`: recurrent stashes copied asynchronously

- `GDNState.stash()` copies every recurrent layer's slot slices device → pinned staging slab with `non_blocking=True` on the current stream and records one CUDA event per device used. It returns a `PendingStash` holding freshly allocated pageable, storage-owning tensors of the served `.cpu()` shapes and dtypes.
- A daemon worker thread waits on the events, copies slab → pageable tensors, returns the slab to the pool and sets a ready flag. The bytes are the bytes the synchronous `.cpu()` copy produces.
- The readers wait for the flag: `GDNState.unstash` (every restore) and the NVMe tier's `serialize_stash`. Nothing else reads stash tensor contents.
- `EXL3_STASH_ASYNC_SLABS` (default 2, not set by the launcher) pinned slabs of one stash each (about 57 MiB, a 64 MiB block from the caching host allocator), allocated at the first stash and reused. With no free slab the stash takes the served synchronous path, counted.
- The generator runs under `torch.inference_mode()`, which is thread-local: the worker fills the destinations under the mode the stash was made in, and the slabs are allocated outside it ([GOTCHAS 36](../../../docs/GOTCHAS.md)).
- `EXL3_PREFILL_MERGE=1` without the async key captures into the same staging slab and finishes the copy on the generator thread before the stash is stored.

R803 measured about 10 ms saved per stash on its prefill ladder and 18 to 22 ms per 2,048-to-4,095-token agent turn under 8-slot load (two stashes per turn).

Both keys are read per call, so one process can switch them. The first use of each path prints one ` -- prefill-merge-r1: ...` line; counters are in `exllamav3.cache.prefill_merge.metrics` (merged prefills, snapshot stashes, asynchronous stashes, no-slab fallbacks, incomplete snapshots).

## Tests

The build runs the landing check, the four offline tests and the served `rebase-dev-r3` CPU suite against the stacked package and its TabbyAPI call-site audit against `/app`, and fails on any failure. On the build host (macOS, Python 3.12, torch 2.14 CPU) on 2026-09-29, `tests/run_offline.sh` read: partition 5 passed, split equivalence 14, forward wiring 7, GPU gate logic 9, the served CPU suite with 0 failures.

- `tests/test_partition.py`: the Job rule over the N classes of the table and every N from 2 to 8,193: served and merged forward counts, no forward over 2,048 rows, the same stash positions, split offsets on page boundaries.
- `tests/test_split_equiv.py`: the GDN and PLE split helpers against two sequential served calls with the torch reference kernels (slot bytes, outputs, captured stash bytes); the asynchronous stash (bytes, readers waiting, the fallback without a slab, under and outside `torch.inference_mode`); the Job-side guards.
- `tests/test_forward_wiring.py`: drives `GatedDeltaNet.forward` and `PLELayer.forward` themselves. With a split the GDN forward calls the split helper once, and the PLE forward equals two separate forwards of the files in `base/`; without one, both make exactly the calls of the base files' forwards.
- `tests/test_gpu_gate_logic.py`: the GPU gate's verdict functions on CPU fixtures; every hard line fails on its own mutation.
- `tests/partition_sim.py`: the chunk-loop replay shared by the partition test and the GPU gate.
- `tests/gpu_stash_equiv.py`: the GPU gate (R803 step 0), one process with the model loaded and the NVMe tier off. Per case (17 cases of 257 to 8,192 new tokens, cold or behind an 8,192-token cached prefix, including a pipelined window followed by a merged tail) it runs OFF, ASYNC, MERGE and MERGE+ASYNC, plus a CTRL arm (OFF at a 256-row chunk size), and compares every stash by position, the counters, the greedy tokens of the turn and of a follow-up turn, and the follow-up's cached count. Hard: ENGAGED, ASYNC == OFF and MERGE+ASYNC == MERGE bitwise, REUSE, MERGE ~ OFF at the first GDN layer within 1e-2 relative L2. Reported: the whole-stash distance, MERGE == OFF, CTRL ~ OFF, greedy agreement.
- `landing_prefill_merge.py`: in the image, the installed files hash to `SHA256SUMS.src`, the hooks are in the functions that carry them, both keys are off by default, the package loads from site-packages.

Outside Docker the tests need torch (CPU), pydantic, safetensors and tokenizers, and the served tree from [`../rebase-dev-r3/prepare-tree.sh`](../rebase-dev-r3/prepare-tree.sh):

```sh
bash docker/overlays/rebase-dev-r3/prepare-tree.sh
bash docker/overlays/prefill-merge-r1/tests/run_offline.sh python3
```

The GPU gate runs inside the image on a host with the checkpoint:

```sh
docker run --gpus all -v <checkpoint dir>:/models/MODEL -v <results dir>:/results --entrypoint python3 tabbyapi:merge-tok-r1 \
  /opt/prefill-merge-r1/tests/gpu_stash_equiv.py --model /models/MODEL --out /results/gpu-equiv \
  --cache-quant 8,8 --gpu-split 30,30 --draft 3 --draft-split 0,32
```

## Files

- [`Dockerfile.box`](Dockerfile.box): FROM `BASE`. Checks that the base carries `tokenize-offloop-r2`, loop-think r5 and `rebase-dev-r3`; hash-checks the installed files against `SHA256SUMS.base`, dry-runs, applies `fix.patch` with `--fuzz=0 --forward` and hash-checks the result against `SHA256SUMS.src`; runs the landing check and the offline tests; re-checks `tokenize-offloop-r2`'s files and landing on the stacked package; runs the served CPU suite and the call-site audit.
- `fix.patch`: `diff -ruN base src`, `-p1` in the installed package directory. `SHA256SUMS.base`, `SHA256SUMS.src`: the touched files before and after.
- `base/`: the four served ExLlamaV3 files the patch changes, upstream `dev` `5783a93` with [`../rebase-dev-r3/ported-vs-dev.patch`](../rebase-dev-r3/ported-vs-dev.patch) applied (MIT); the forward-wiring test loads them as the reference.
- [`mkpatch.sh`](mkpatch.sh): regenerates `fix.patch` and both `SHA256SUMS` files from `base/` and `src/`, checks that `base/` is still the served tree's bytes and that the patch applies at fuzz 0 and reproduces `src/`. `src/` is not in this repository; the script's header says how to rebuild it. Never hand-edit the patch.
