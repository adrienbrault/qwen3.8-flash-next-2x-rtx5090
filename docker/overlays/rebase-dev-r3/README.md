# rebase-dev-r3: the served engine stack on upstream ExLlamaV3 `dev` 5783a93 (v1.5.2)

`tabbyapi:rebase-dev-r3` is served on `:8022` since 2026-09-27 15:01 CEST ([R785](../../../bench/results/r785-promote-rebase-r3.md)). Until then the engine of every served image was ExLlamaV3 v1.5.0 with this repository's layers applied one on top of the other ([`../../README.md`](../../README.md)). This image replaces that whole `exllamav3` package with one tree: upstream `dev` at `5783a9360b749a1d5bde6862dc35e7766e542ca1` (v1.5.2) plus [`ported-vs-dev.patch`](ported-vs-dev.patch), which carries every engine mechanism of `stack-r3-rows32` and the requeue token-count fix of `tokcount-r1`. TabbyAPI is the base image's `/app`, unchanged: `53da7919` with this repository's patches and [`loop-think-r4`](../loop-think-r4/README.md).

## Build

```sh
bash prepare-tree.sh        # fetches 5783a93, applies ported-vs-dev.patch, checks both trees' SHA-256, writes out/rebase-dev-r3/
B=tabbyapi:stack-r3-rows32-tokcount-loopthink4
docker build -f Dockerfile.box --build-arg BASE=$B \
  --build-arg BASE_ID=$(docker image inspect $B --format '{{.Id}}') \
  --build-arg DGV2_NVCC_DEFS="$(docker image inspect $B --format '{{index .Config.Labels "local.stack.dgv2_defs"}}')" \
  --build-arg TREE_SHA=150497b44e6f49b06b8a13f2568e68fc41841bc28964928509e181b633285501 \
  --build-arg MAX_JOBS=4 -t tabbyapi:rebase-dev-r3 .
```

[`../../build-chain.sh`](../../build-chain.sh) runs both steps as its last layer. The build removes the base's `exllamav3` package and its prebuilt `exllamav3_ext*.so`, empties the torch extension cache, installs the tree and rebuilds the extension once for sm_120 with [`rebuild-native.py`](rebuild-native.py). [`landing_r3.py`](landing_r3.py) then checks where each module is loaded from, runs the landings of `rows32-r4`, `stack-r3`, `tokcount-r1` and `loop-think-r4` against the new tree, and imports the 41-key environment served before this image with the real extension; it also asserts that the tiled hyper-connection prefill mix, the launcher's 42nd key, is on by default. The CPU suite in [`tests/`](tests/) runs last with the extension stubbed, including [`test_tabby_callsites.py`](tests/test_tabby_callsites.py), which checks every engine call TabbyAPI's backend and loop-think's two files make against the installed engine. `EXL3_*` selectors are not set in the image; the launcher sets them.

## How the tree was made

- r2 ported the served stack onto upstream `dev` `73a6229`: the served mechanisms added since an earlier port and upstream's 40 commits from `1d64111`, with the pairs where both sides changed the same code resolved by keeping both behind a flag ([`port-map-r2.md`](port-map-r2.md)).
- r3 merged upstream `73a6229..5783a93` into r2: the sm_75 support of #325, the per-device shared-memory budget `f4db698`, v1.5.2 (`988fa66`) and the loader's shared file streams (`5783a93`). One file conflicted. For each of the 30 upstream paths, the added and removed lines of r2 → r3 equal those of upstream's range, so [`r2-to-r3.patch`](r2-to-r3.patch) is upstream's four commits and nothing else ([`impl-status.md`](impl-status.md)).
- By code reading, those four commits change nothing on sm_120: the new architecture gates and the shared-memory ladder select the existing paths on this card ([`impl-status.md`](impl-status.md)).
- exllamav3#337 (the device pinned during layer-split forwards) is not in upstream `5783a93` and is still carried in the patch.

## What changes numerics

Upstream code that the v1.5.0 chain did not have and that the port keeps: GDN prefill projections in fp16, a deterministic router GEMM, the tiled hyper-connection prefill mix (`EXL3_GR_MIX_TILED=1` on the served launcher) and the PLE in-place add ([`port-map-r2.md`](port-map-r2.md), "Deliberate behavior changes"). Greedy output is therefore not identical to the previous image: 4 of 6 `fn_greedy` prompts and 1 of 5 `chat_greedy` prompts match ([R784](../../../bench/results/r784-rebase-dev-r3.md)). Over a 409,400-position corpus the mean NLL differs by −0.04 % and the top-1 agreement is 0.968, without a null run to judge it against.

## Measured

- [R784](../../../bench/results/r784-rebase-dev-r3.md), one ABBA block against `stack-r3-rows32-tokcount-loopthink3` at the served flags: page pool 901,120 tokens at the same per-card headroom (−8.3 %, 983,040 before); cold prefill 1.134× at 90k tokens and 1.159× (pooled) at 22.6k; decode on short prompts −1.3 to +1.6 % per cell at 1, 4 and 8 streams (95 % intervals between −4.1 and +3.7 %), which the pre-registered rule could not resolve (prose at 8 streams, lower bound −2.30 %).
- [R785](../../../bench/results/r785-promote-rebase-r3.md), the promotion gates on the serving port, passed twice: GSM8K 0.982 and 0.980, tool-eval 85.5 and 86.5, needles 5/5 at 131k and 240k, agentic edit 24/24.
- Long-context decode is unresolved: R785's agent replay (median prompt 29,616 tokens) read 123.0 and 123.7 tokens/s per stream against 128.6 for the previous image in R728, on another day and with the NVMe tier off there. A same-session A/B has not run.

## Files

- [`prepare-tree.sh`](prepare-tree.sh): writes `out/rebase-dev-r3/{exllamav3,tests}` (not carried by this repository, 25 MB) and checks the upstream tree (`3f7b139b…`) and the result (`150497b4…`, the image's label `local.rebase.tree_sha256`).
- [`ported-vs-dev.patch`](ported-vs-dev.patch): upstream `5783a93` → the served tree, `git apply` in the upstream checkout.
- [`r2-to-r3.patch`](r2-to-r3.patch): the r2 → r3 delta, for review.
- [`Dockerfile.box`](Dockerfile.box), [`rebuild-native.py`](rebuild-native.py), [`landing_r3.py`](landing_r3.py), [`tests/`](tests/): the build and its checks.
- [`r784_decide.py`](r784_decide.py): R784's pre-registered decision rule, with a self-test (`--selftest`).
- [`impl-status.md`](impl-status.md), [`port-map-r2.md`](port-map-r2.md): the port's record.
