# Tail checkpoint capture inside the prefill pipeline

`tabbyapi:r823c-cachetail-inforward` is `tabbyapi:merge-tok-r1` plus the bundled `trace/`, `tail/` and final in-forward patches. Served since R823p, 2026-10-01; conditions, paired measurements and fidelity limits are in [R823–R823p](../../../bench/results/r823-tail-checkpoints.md).

The three launcher selectors are `EXL3_RECURRENT_CHECKPOINT_INTERVAL_PP=4096`, `EXL3_RECURRENT_CHECKPOINT_TAIL_PP=12288` and `EXL3_RECURRENT_CHECKPOINT_INFORWARD=1`. The interval is 32,768 outside the last 12,288 prompt rows and 4,096 inside; the existing 2,048 near-end interval remains. Alignment is relative to the restored cached prefix. The image sets `EXL3_CACHE_TRACE=1`; the launcher defaults NVMe off.

The planner keeps the coarse LS window and two-job fairness limit. Before stage A, a staging slab and chunk-local descriptor are reserved; GDN/conv/PLE state is captured on each layer's issuing stream and published after page commit. An unavailable slab cuts the window before lookahead and uses the old coherent stash. The real-model CUT/INFORWARD/NO_SLAB equality probe passed the three fixtures; output fidelity under concurrency remains unresolved.

## Build

From the public repository root, with `tabbyapi:merge-tok-r1` already available:

```sh
docker build --network=none -t tabbyapi:r823c-cachetail-inforward docker/overlays/r823c-cachetail-inforward
```

`Dockerfile` applies the trace interval seam and engine/app instrumentation, the bounded tail policy, then `fix.patch`. Every stage checks its before/after SHA-256 manifests, applies at fuzz 0, rejects patch rejects/backups and runs the combined landing check. `trace/` and `tail/` are the exact predecessor build inputs. No native kernels are rebuilt. `docker/build-chain.sh` includes this final layer after `merge-tok-r1`.

`base/` and `src/` are not redistributed: reconstruct the predecessor and result in a copy of the SHA-matching installed package by applying the bundled patches in Dockerfile order. The actual-model probe is [`bench/r823c_checkpoint_equal.py`](../../../bench/r823c_checkpoint_equal.py), run under the GPU queue lock with clients drained, NVMe off and the served selectors. Passing capture equality does not substitute for the HTTP output or production restore gates.
