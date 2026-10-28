# N-gram prefetch staging-set race fixed with identical greedy output and unchanged speed (pf2lock r3, served 2026-10-28)

Results directories on the serving host: `/srv/qwen5090/results/2026-10-28-153418-flashnext-pf2lock-screen-screen-4071598` (screen 1) and `/srv/qwen5090/results/2026-10-28-154823-flashnext-pf2lock-screen-screen-3242976` (screen 2); the CPU validation ran in containers of the served image without a GPU. Driver: the A/B screen kit and its validator in the private infrastructure repository, not in this repository. Image recipe: [`docker/overlays/pf2lock-r3/`](../../docker/overlays/pf2lock-r3/README.md). Raw records: [`2026-10-28-pf2lock-r3/`](2026-10-28-pf2lock-r3/) (each screen's `REPORT.md` and audit log as `audit.txt`, the validator verdict, the CPU stress JSON and logs, the deterministic test logs; every log is stored with a `.txt` suffix).

## The defect

The served `EXL3_NGRAM_PREFETCH2` path (`ngram-prefetch-r1`, carried into `tune268-r2` as `exllamav3/modules/ngram_embedding.py`, SHA256 `b9b227f8…`) stages n-gram embedding rows in two pinned staging sets. The prefetch worker's deferred staging and the issuing thread's inline staging on a lookup miss both chose a free set before either marked it held, so both could write one set and one forward could return the other's rows. The same unguarded `_pending` list let a retire cancel an entry a forward had just matched (`CancelledError`), let the worker read an entry before its future existed, and raised `IndexError` when no free set remained after a retire. Failure paths could leave a set held. The race was found by CPU tests during a rebase onto a newer ExLlamaV3 tree. How often the served image returned wrong rows in production is unknown: the module's prefetch counters are not exported.

The patch (`pf2lock.patch` r3, SHA256 `13a892c1…`, module `b9b227f8…` → `e04ef253…`) adds a lock around set selection and `_pending` membership, a single owner that releases each held set on every path including failures, and a per-entry commit/abort gate for submitted tasks. It changes no arithmetic, no selector and no image ENV; the native extension is byte-identical to the base's. The [overlay README](../../docker/overlays/pf2lock-r3/README.md) describes each round.

## CPU validation, 2026-10-28

`cpu_stress.py` runs 5,000 iterations per scenario, each on a fresh module; an iteration fails on wrong rows, an exception, a hit/miss count other than the asserted one, a pending entry or held set left after the forwards, more than two staging sets, or a hang. The runs used the served image under `docker run --rm --network=none --cpus=4`, torch 2.9.0+cu128, no GPU.

| scenario | `EXL3_NGRAM_PREFETCH2` | served module: failed / 5,000 (wrong rows, exceptions) | patched module: failed / 5,000 |
| --- | --- | ---: | ---: |
| share: `prefetch_ids(a)`, a miss `forward(b)`, a hit `forward(a)` | 1 | 3,430 (1,512, 1,918) | 0 |
| steady: 20 steps of verify prefetch, a spurious prefetch, a miss and a hit | 1 | 2,503 (571, 1,932) | 0 |
| long: long `prefetch()`, a long miss, a long hit | 1 | 3,072 (690, 2,382) | 0 |
| offpath | unset | 0 | 0 |

The served-module rows for share, steady and offpath come from the first harness version (SHA256 `6f487b48…`, not in this repository; its environment is in `cpu-served/env-share-steady-offpath.txt`); the long row and every patched row use `cpu_stress.py` as committed (`2c23e319…`). `pf2lock_tests.py` (24 deterministic interleavings and injected failures, each followed by a check that the same instance serves correct rows) fails 19 on the served module (15 failures, 4 errors) and passes 24/24 on the patched one. The Dockerfile reruns the four scenarios and the 24 tests as build gates; the operator build of the served image passed them ([`build-manifest.json`](../../docker/overlays/pf2lock-r3/build-manifest.json)). Zero failures in these runs show that no failure was observed, not that none remains. The CPU failure rate is not the production rate: every iteration starts with empty sets, and in serving a miss must coincide with a running deferred stage.

## Promotion screens, 2026-10-28

Two screens on port 18022, each one boot sequence B0, A1, B1, B2, A2 under the GPU lock: screen 1 from 15:34 to 15:47 UTC, screen 2 from 15:48 to 16:01 UTC. A is `tabbyapi:tune268-r2` (`sha256:2239954c…`), B is `tabbyapi:tune268-r2-pf2lock-r3` (`sha256:da386167…`); both arms boot the served launcher with its 50 selectors, checkpoint `qwen3.8-flash-next-exl3-2.68bpw-selfcal`, pool 819,200, split `[30, 30]`, eight slots, MTP draft depth 3, the served boot warm-up and no NVMe tier. B0 only gates the candidate's boot. Each measured boot runs `fn_bench` code decode at 1 and 4 streams (greedy, 1,024 forced output tokens, distinct prompts per stream, one warm-up round and two measured rounds), prose decode at 1 stream with thinking on, the six-agent replay, and 24 greedy prompts of 512 forced tokens at temperature 0 streamed over `/v1/completions`.

| metric | screen 1 A / B mean | screen 2 A / B mean | B vs A, pooled over 4 boots per arm | noise half-width |
| --- | ---: | ---: | ---: | ---: |
| code, 1 stream, per-stream decode (t/s) | 248.43 / 248.25 | 248.56 / 247.79 | −0.19 % | 0.40 % |
| code, 4 streams, decode aggregate (t/s) | 588.21 / 589.61 | 591.19 / 588.34 | −0.12 % | 0.99 % |
| six-agent replay span (s) | 46.89 / 46.80 | 46.05 / 45.42 | −0.79 % | 2.29 % |

The predeclared windows were −2 % for each decode figure and +3 % for the replay span. Greedy output was identical in all four boots of each screen (B1 = B2 = A1 = A2, 24/24 prompts, including the A/A and B/B controls), and the seven format checks were identical across arms. The validator read both screens as eligible for promotion. The six-agent replay runs `agent_replay.py` with `--sessions 6 --steps 3 --base 30000 --gen 512 --tool 1200 --think 1 --stagger 1 --seed 20261021`, the arguments of the [rebase repair write-up](flashnext-rebase-repair.md); the screen pins a later revision of [`bench/agent_replay.py`](../agent_replay.py) (SHA256 `0cf9c053…`, not in this repository) that also records each request's finish reason and stream end.

The launcher change is the image tag and its pinned image ID. The image was promoted at about 16:05 UTC on 2026-10-28; the previous launcher is [`scripts/launchers/launch-flashnext-tune268r2.sh`](../../scripts/launchers/launch-flashnext-tune268r2.sh). The served figures in the README were measured on 2026-10-23 on `tune268-r2` ([publish268-served-268](publish268-served-268.md)); these screens measured no change beyond the noise half-widths above.
