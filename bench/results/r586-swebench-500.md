# R586, R586d: SWE-bench Verified, all 500 instances with the task containers off the network: 397 resolved

Results directories on the serving host: `results/2026-09-20-r586-swebench-full` (R586) and `results/2026-09-23-r586d-swebench-netnone-redo` (R586d). Raw records: [`2026-09-23-r586-swebench-500/`](2026-09-23-r586-swebench-500/). The drivers are not published. Dates: 2026-09-20 to 2026-09-23.

## What ran

- Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab` on TabbyAPI + ExLlamaV3, port 8022, 8 slots, 999,424-token page pool at 8-bit KV, layer split `[30, 30]`, draft policy `[[4, 3], [5, 2], [8, 1]]`, the served launcher on every boot. Sampler: temperature 0.6, top_k 20, top_p 0.95 from the seat's preset, because mini-SWE-agent sends only `max_tokens` (32,768). One sampled attempt per instance.
- Harness: mini-SWE-agent 2.4.6, the builtin `benchmarks/swebench.yaml` unmodified (the leaderboard's bash-only setting: step_limit 250, three consecutive replies without a tool call end the run, 60 s per command; sha256 `9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e`, which matches the file in the 2.4.6 wheel), plus the overlay [`flashnext-local.yaml`](2026-09-23-r586-swebench-500/flashnext-local.yaml) for the endpoint and the task-container arguments `--rm --memory=3g --memory-swap=3g --network none`. litellm 1.99.0.
- Dataset `princeton-nlp/SWE-bench_Verified`, split test, Hugging Face snapshot `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
- Scorer: the official swebench harness 4.1.0 (`swebench.harness.run_evaluation`, official `sweb.eval` images, per-instance timeout 1,800 s), 0 harness errors in both runs.
- 8 agents on 8 slots until 2026-09-22 20:50 UTC, 12 agents after that.

## Why there are two runs

R586 ran the task containers with network access until 2026-09-20 06:13 UTC. The agent used it: 279 fetch commands across 158 trajectories, including a `pip download` of a later release of the package under test. `--network none` was added to the overlay at 06:13; each trajectory records the arguments its container ran with, and 160 trajectories (dataset indices 0 to 159: 22 astropy, 138 django) have no `--network none`. R586d re-ran exactly those 160 with `--network none` into a fresh directory and left R586's records in place.

On the 160 instances, the run with network resolved 156 and the run without resolved 129: 28 went from resolved to unresolved and 1 (`django__django-11820`) the other way. 5 of the 28 are R586d runs that ended without a patch (below); on the 155 instances with a patch in both runs, 151 against 129. The two runs also differ in image (`mtpwin-r2` against `stack-r1`), in agents (8 against 12) and in the server faults listed below. Within one image and one agent count, R586 resolved 135 of 138 django instances with network (indices 22 to 159) and 49 of 54 django instances without (indices 160 to 218), which are different instances.

## Result

The merged set takes R586d's prediction for the 160 re-run instances and R586's for the other 340. Every one of the 500 scored trajectories records `--network none`.

| | count |
| --- | --- |
| resolved | **397 of 500 (79.4 %)** |
| unresolved, with a report | 96 |
| ended without a patch, counted unresolved | 7 |
| resolved, the 2 server-error exits left out of the denominator | 397 of 498 (79.7 %) |
| resolved, of the 493 with a report | 397 of 493 (80.5 %) |

The 7 without a patch:

- 2 ended on HTTP 503 from the server after the client's retries ran out: `django__django-11555` and `django__django-14539` (R586d, re-run once by the driver's retry pass, 503 again). They have not been run a third time.
- 5 ended with three consecutive replies that held only reasoning, no answer and no tool call: `django__django-11087`, `django__django-12209`, `django__django-12663` (R586d), `sphinx-doc__sphinx-8638`, `sympy__sympy-23413` (R586). R586d's trajectories hold 28 such replies and R586d's container log records 25 generations `stopped because a token loop was detected` by ExLlamaV3's loop detector, so at least 25 of the 28 ended there. The 5 such replies in `sphinx-doc__sphinx-8638` carry the same timestamps, to the second, as the 5 loop-detector stops in R586's last container log (13:25:08 to 13:34:08 UTC on 2026-09-23); the container log for `sympy__sympy-23413`'s window was not kept.

No trajectory reached the 250-step limit (the longest took 178 model calls), no reply stopped at `max_tokens`, and the longest trajectory took 1.13 h against the task container's 2 h lifetime.

## Server faults during the runs

| when (UTC) | image | what |
| --- | --- | --- |
| 2026-09-22 20:59, 21:24, 21:48, 22:18, 22:38 | `stack-r1` | CUDA out of memory in `graph.cu`, the container restarted |
| 2026-09-22 22:58 to 23:25 | `stack-r1` | every request answered 503: the recurrent-state slot pool had lost its slots, fixed since in [R676](r676-slotfix.md) |
| 2026-09-23 12:09, 12:50, 13:16, 13:45 | `stack-r1` | CUDA out of memory in `graph.cu`, the container restarted |
| 2026-09-23 15:01 | `stack-r1` (R586d) | CUDA out of memory in `graph.cu`, the container restarted |
| 2026-09-23 15:01 to 18:46 | `stack-r1` (R586d) | 2,592 requests aborted with 503 after CUDA out of memory in the prefill pipeline on GPU 0 (40 MiB requested, 28.56 MiB free), in 38 separate minutes |

Every instance that ended on a server error in R586, including the 63 that ended during the 22:58 to 23:25 window, was re-run by the driver's retry pass on 2026-09-23 (99 instances: 76 resolved, 22 unresolved, 1 ended on the loop detector). No scored trajectory received a reply between 22:58 and 23:25; 12 sphinx trajectories received their first reply between 23:26:51 and 23:27:53, so their first request may have been retried. R586d re-ran its 11 server-error exits once; 2 of them are the two above.

A trajectory holds only the replies that succeeded. When a request failed, the client sent the same messages again and the reply is a new draw from the same sampler. 167 of the 500 scored trajectories had at least one model call waiting while a fault in the table above was recorded, and 150 R586 trajectories ran on boots whose container log was not kept, so their exposure is unknown. Three trajectories received an empty reply (no text, no reasoning, no tool call) and continued; all three resolved. The per-instance counts are in `instances.csv`.

## Image segments

| segment | image | agents | UTC | instances scored | resolved | without a patch |
| --- | --- | --- | --- | --- | --- | --- |
| R586 | `tabbyapi:mtpwin-r2` | 8 | 2026-09-20 06:13 to 08:42 | 54 | 49 | 0 |
| R586 | `tabbyapi:stack-r1` | 8, then 12 | 2026-09-22 19:49 to 2026-09-23 14:25 | 286 | 219 | 2 |
| R586d | `tabbyapi:stack-r1` | 12 | 2026-09-23 14:43 to 18:55 | 160 | 129 | 5 |

The two images differ by three promotions. `mtpwin-r2-metrics1` added a metrics endpoint with identical greedy fingerprints ([R587](r587-tabby-metrics.md)). `bverify-r1` made the batched draft verifier reachable, for sampled requests as well as greedy ones, and R646 recorded no greedy-output comparison ([R646](r646-verifybatch.md)). `stack-r1` produced byte-identical greedy output to `bverify-r1` on 6 prompts ([R653](r653-stack.md)). Every request in these runs is sampled, so from 2026-09-22 on they were eligible for the batched sampled verify path, which R646 notes draws its random numbers in a different per-row order than the serial path. Checkpoint, slots, page pool, split, draft policy and agent configuration are the same in every segment; the 8 `stack-r1` boots list the same 23 engine environment keys, and the 2026-09-20 boot predates that log line.

## Limits of the number

- One sampled attempt per instance at temperature 0.6.
- Two images and two agent counts, in the proportions above.
- The merged 397 is the union of two runs' per-instance `report.json`; the harness never produced one summary for it. R586's own summary, 424 of 500, includes the 160 runs with network and is not this result.
- Exact reruns of `django__django-11555` and `django__django-14539` would complete the set without server-error exits; they can change the count by at most 2.

## Raw records

In [`2026-09-23-r586-swebench-500/`](2026-09-23-r586-swebench-500/):

- `instances.csv`: one row per instance: the run whose prediction was scored, image, agents, network, first and last reply time (UTC), model calls, exit status, result, the result of the run with network for the 160, calls waiting during a recorded fault, and whether the server log for its window was kept.
- `preds.jsonl`: the 500 scored predictions in dataset order. The scorer's `model_name_or_path` label on the serving host is a fixed string shared with other runs; it is replaced here by the served model id, and the patches are unchanged.
- `preds-networked-160.jsonl`: R586's predictions for the 160 instances re-run as R586d.
- `scorer-summary-r586-fn500.json`, `scorer-summary-r586d-fn500redo.json`: the harness summaries of the two runs; `merged-score.txt`: the merge and the per-instance changes on the 160.
- `flashnext-local.yaml`: the agent overlay, settings as run, comments replaced by a provenance header.
- `audit-r586.txt`, `audit-r586d.txt`, `boots.txt`: the drivers' audit logs and the launcher's boot lines, with host paths and addresses removed.

Left on the serving host: the 660 trajectories (`results/2026-09-20-r586-swebench-full/out/`, `results/2026-09-23-r586d-swebench-netnone-redo/out/`), the scorer's per-instance logs and reports (`logs/run_evaluation/` in each directory), the agents' `run.log`, the scorer's `score.log` and the container logs `docker-final.log`, about 430 MB in all.
