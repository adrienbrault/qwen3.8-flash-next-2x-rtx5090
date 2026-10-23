# RUNBOOK — reproducing this stack and every measurement in it

Everything here runs on the `flan` box. Nothing in this file is a description of what someone once did; each block
is the command that produced a results directory named beside it.

## 0. Prerequisites

| what | where |
| --- | --- |
| the box | `ssh flan` (BatchMode works; `sudo docker` over that ssh) |
| checkpoint | `/srv/qwen5090/models/qwen3.8-flash-next-exl3-2.68bpw-selfcal`, turboderp revision `2.68bpw_h6_ng8` |
| images | `tabbyapi:tune268-r2`, built by `bash docker/build-chain.sh`; the served launcher pins measured image `2239954c`, so a reconstruction requires a reviewed pin update |
| probes | flashed from `bench/*.py` to `/srv/qwen5090/probes/` |
| runners | flashed from `bench/*.sh` to `/srv/qwen5090/` |
| GPU lock | `/srv/qwen5090/lib/gpu-queue.sh` + `/srv/qwen5090/gpu-exclusive.lock`; every runner takes it |
| the box's other engine | `bash /srv/qwen5090/daily-restore-retry.sh` (and the two engines cannot coexist) |

Flashing a probe: `scp bench/probe.py flan:/srv/qwen5090/probes/fn_bench.py`. The runners expect that name.

## 1. Serve

```sh
ssh flan 'bash -s' < scripts/launch-flashnext.sh                 # the served baseline
IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]' bash scripts/launch-flashnext.sh   # both levers
SYS_KV=4096 bash scripts/launch-flashnext.sh                      # host KV tier (measured not to matter)
STOP=1 bash scripts/launch-flashnext.sh                           # stop
```

## 2. Measure

Each of these is a runner in `bench/`, all under `systemd-run` so a dropped ssh does not kill a measurement, and
all taking the GPU lock so nothing else runs beside them.

| runner | what it produces | results dir |
| --- | --- | --- |
| `r343-depth.sh` | decode and TTFT against prompt depth | `2026-09-16-r343-depth` |
| `r339-gates.sh` | depth, needle, admission, code/prose, soak | `2026-09-16-r339-gates` |
| `r345-pool.sh` | shared prefix vs genuinely independent deep contexts | `2026-09-16-r345-pool` |
| `r347-soak.sh` | a clean 40-round soak with drift | `2026-09-16-r347-soak` |
| `r348-capabilities.sh` | JSON schema, tools, vision, reasoning | `2026-09-16-r348-capabilities` |
| `r340-ci-depth.sh` | control / parity / treatment for the draft-depth policy, with the byte-equality gate | `2026-09-16-r340-ci-depth` |
| `r341-qsa-ab.sh` | QSA multi-job control vs treatment, concurrent-greedy equality first | `2026-09-16-r341-qsa` |
| `r354-combined.sh` | both levers together against the baseline | `2026-09-16-r354-combined` |
| `r355-fn-gsm8k.sh` | GSM8K, lm-eval, as served | `2026-09-16-r355-fn-gsm8k` |
| `r357-tooleval.sh` | tool-eval 69×4 on the baseline and on the promoted config | `2026-09-16-r357-tooleval` |
| `r358-hostkv.sh` | host KV tier, three shapes | `2026-09-16-r358-hostkv` |
| `r359-swebench.sh [N]` | SWE-bench Verified, first N instances, scored by the official harness | `2026-09-16-r359-swebench-N` |
| `r356-promoted.sh` | the promoted config against the original failing agent request | `2026-09-16-r356-promoted` |

Example:

```sh
ssh flan 'sudo systemd-run --unit=r343-depth --collect -p User=adrienbrault -p RuntimeMaxSec=10800 \
  bash /srv/qwen5090/r343-depth.sh'
```

Read any of them back with the summariser, which works from the records and never from a printed summary:

```sh
python3 /srv/qwen5090/probes/summarize.py /srv/qwen5090/results/<dir>/records-*.jsonl
```

## 3. Rebuild an image

The complete recipe is in this repository's [`docker/`](../docker/README.md). Build the chain on x86_64 Linux with Docker BuildKit; the final two native overlays disable network access and check the patch hashes and predecessor SASS.

```sh
bash docker/build-chain.sh
DRY_RUN=1 bash docker/build-chain.sh
```

The final tag is `tabbyapi:tune268-r2`. Rebuilding changes the image ID; review the installed sources and update the launcher's `R825C_IMAGE_ID` before serving the reconstruction.

## 4. Hand the box back

```sh
ssh flan 'STOP=1 bash /srv/qwen5090/launch-flashnext-r340.sh'   # stop this seat
ssh flan 'bash /srv/qwen5090/daily-restore-retry.sh'            # restore the box's other engine
```

Decide per run whether the restore is needed: the two cannot coexist, and whichever is served owns the box.

## Flash-Next as the selected daily (2026-10-21)

The validated `tabbyapi:r828-prompt-lookup-r3` configuration was restored on :8022 on 2026-10-21. Its checkpoint, 262,144-token window, 901,120-token pool and launcher flags are unchanged. Boot warm-up, model identity and an arithmetic chat smoke passed. No new engine or checkpoint was installed.

`scripts/serve-flashnext-daily.sh` runs on the serving host after installing `scripts/launch-flashnext.sh` as `/srv/qwen5090/launch-flashnext.sh`. It pins that published launcher's SHA256, checks the GPU lock and pending queue, boots with a clean environment, and writes `flashnext` to `/srv/qwen5090/default-daily` after a healthy model check. The infrastructure restoration helpers dispatch this selector to the installed wrapper; they no longer select another engine after an experiment. The existing launcher retains its power and memory-clock policy.

## Served daily, 2026-10-23

The current daily is `qwen3.8-flash-next-exl3-2.68bpw-selfcal` on `tabbyapi:tune268-r2`, pool 819,200, 50 selectors and the served warm-up. The [publication write-up](../bench/results/publish268-served-268.md) records the three measurement runs and their conditions. Install the public launcher and matching `scripts/serve-flashnext-daily.sh` together; the wrapper pins the scrubbed public launcher's SHA256.
