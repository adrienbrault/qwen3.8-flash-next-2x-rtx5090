# RUNBOOK — reproducing this stack and every measurement in it

Everything here runs on the `flan` box. Nothing in this file is a description of what someone once did; each block
is the command that produced a results directory named beside it.

## 0. Prerequisites

| what | where |
| --- | --- |
| the box | `ssh flan` (BatchMode works; `sudo docker` over that ssh) |
| checkpoint | `/srv/qwen5090/models/qwen3.8-flash-next-exl3-3.05bpw` |
| images | `tabbyapi:53da7919-rqcount` (served), `-rqcount-cid`, `qsa-devel`, `qsa-devel-control`, `qsa-cid` |
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

The Dockerfiles live in the sibling `kubernetes-home` repository (`flan/docker/`), because that is where the box's
runner scripts live.

```sh
# the served image: TabbyAPI 53da7919 + exllamav3 v1.5.0 + the requeue token-count fix
scp kubernetes-home/flan/docker/Dockerfile.tabbyapi flan:/srv/qwen5090/docker/ && ssh flan \
  'cd /srv/qwen5090/docker && sudo docker build -f Dockerfile.tabbyapi -t tabbyapi:53da7919-rqcount .'

# the QSA half needs a CUDA devel base: it changes libtorch/attention.cpp, and the serving image has no nvcc
scp kubernetes-home/flan/docker/Dockerfile.tabbyapi-qsa{,-cid} flan:/srv/qwen5090/docker/ # plus the patches
```

Each Dockerfile asserts the engine version and greps for a marker the patch introduces, so a build that silently
did nothing fails instead of serving.

## 4. Hand the box back

```sh
ssh flan 'STOP=1 bash /srv/qwen5090/launch-flashnext-r340.sh'   # stop this seat
ssh flan 'bash /srv/qwen5090/daily-restore-retry.sh'            # restore the box's other engine
```

Decide per run whether the restore is needed: the two cannot coexist, and whichever is served owns the box.
