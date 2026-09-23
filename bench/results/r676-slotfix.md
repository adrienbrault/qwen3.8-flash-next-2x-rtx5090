# R676: the recurrent-state slot pool returns a slot when state construction fails, promoted

Results directory on the serving host: `results/2026-09-23-r676-slotfix-gate`. Raw records: [`2026-09-23-r676-slotfix-gate/`](2026-09-23-r676-slotfix-gate/). Driver [`scripts/r676-slotfix-gate.sh`](../../scripts/r676-slotfix-gate.sh), churn load [`bench/fn_churn.py`](../fn_churn.py). Image `tabbyapi:slotfix-r1` = `tabbyapi:stack-r1` plus [`docker/overlays/slotfix-r1/slotfix-r1.patch`](../../docker/overlays/slotfix-r1/slotfix-r1.patch). Model `qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab`, 8 slots, 999,424-token page pool.

## The failure

On 2026-09-22 from 22:58 to 23:25 UTC, under 12 concurrent SWE-bench agents, the server answered `/v1/model` and returned 503 on generation requests with `Cannot create new state: no available slots`, 2,380 times in 27 minutes, until the container restarted. Five CUDA out-of-memory errors in `graph.cu` preceded the first one.

In ExLlamaV3's `cache/cache.py` the recurrent-state pool holds one handle per slot. `get_new_state`, `get_test_state` and `new_from_stashed` each take a handle off `free_list` and then construct the state object. When the constructor raises (a state `clear()` or `unstash()` that fails under memory pressure, or a restored stash that fails its position assert), no object holds the handle, `job.recurrent_state` was never assigned, and job cleanup has nothing to release. The slot is lost until restart. Eight such failures empty an 8-slot pool.

## The patch

- The three allocation paths put the handle back on `free_list` when the constructor raises, log `state construction failed; returned slot N to the pool`, and re-raise, so the request that hit the failure still gets its error.
- `release_state` refuses a handle that is already free (`state slot N released twice; ignoring`). A double release would hand one slot to two live states.
- `reap_failed_job` and `cancel()` in the generator run the draft-window release and the page deallocation under separate exception handlers, so a failure in one does not skip the other.
- The pool-exhausted assert reports issued and reclaimed counts.
- `EXL3_SLOTFIX_TEST_FAULTS=N` with `EXL3_SLOTFIX_TEST_FAULT_FILE=<path>` makes the constructor raise N times while `<path>` exists, for the gate. Unset, the path is a single integer test.

Build-time assert: `exllamav3.cache.cache._SLOTFIX_BUILD == "r1"`.

## What was measured

All on one boot of the candidate, 2026-09-23 20:05 to 20:25 UTC.

| check | result |
| --- | --- |
| fault injection: 6 faults armed after boot, 10 requests | 4 served, 6 returned 503; 6 `returned slot` lines, 0 `no available slots`, 0 `released twice` ([`fault-probe.txt`](2026-09-23-r676-slotfix-gate/fault-probe.txt)) |
| greedy output, 6 prompts including one ~100k-token prompt, against `stack-r1` on the same session | 6 of 6 byte-identical ([`greedy.jsonl`](2026-09-23-r676-slotfix-gate/greedy.jsonl)) |
| ramp c1 to c8 | clean, 0 `graph.cu` lines, free VRAM after 117 / 1,921 MiB ([`ramp.jsonl`](2026-09-23-r676-slotfix-gate/ramp.jsonl)) |
| churn: 12 workers for 18 minutes, about 25 % of requests closed mid-stream, forced generations of 2,049 to 8,192 tokens (each requeues through stash and restore at least once), a prompt sharing one ~8k-token prefix across workers | 1,497 requests: 1,107 completed, 390 cancelled by the client, 0 errors; 0 container restarts ([`churn-summary.txt`](2026-09-23-r676-slotfix-gate/churn-summary.txt), [`churn.jsonl`](2026-09-23-r676-slotfix-gate/churn.jsonl)) |
| c8 after the churn, 256 forced tokens, code | 8 of 8 streams completed ([`post-c8.txt`](2026-09-23-r676-slotfix-gate/post-c8.txt)) |

The first run of the gate, at 18:58, did not boot: the injector was armed from process start, the launcher's 16-token warm-up generation consumed a fault, and the container restarted in a loop. The injector now fires only while the arm file exists, and the gate creates it after boot ([`audit.txt`](2026-09-23-r676-slotfix-gate/audit.txt) holds both runs).

The patch does not touch a numerical path, so the gate measured no throughput.

## Verdict

Promoted 2026-09-23 20:25 UTC: the launcher's `DAILY_IMG` is `tabbyapi:slotfix-r1`, with the same 23-key default environment. Rollback is `IMG=tabbyapi:stack-r1`.
