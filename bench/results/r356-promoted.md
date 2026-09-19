# R356: The promoted configuration, validated on the real request

Results directory on the serving host: `results/2026-09-16-r356-promoted`. Raw records: [`2026-09-16-r356-promoted/`](2026-09-16-r356-promoted/). Driver: [`scripts/r356-promoted.sh`](../../scripts/r356-promoted.sh). Date: 2026-09-16.


`IMG=tabbyapi:qsa-cid DRAFT_POLICY='[[2, 3], [8, 1]]'` booted as the served configuration, then the **exact agent
request that failed** — the DSH payload rebuilt from the session archive, 36 tool schemas and a 10,481-token
prompt — and the retrieval gate at depth.

| check | result on the promoted configuration |
| --- | --- |
| the original failing agent request | `finish_reason=tool_calls`, tool calls parsed: **`skill`, `write`** — it plans, loads a skill and writes the file |
| reasoning channel | 49,417 chars, 0.05 % non-Latin (the pre-fix run of the same request: 19,477 chars of multilingual salad in `content`) |
| needle at 105,680 prompt tokens | **5/5 retrieved** |

So the configuration this repository recommends is the configuration that has been run against the workload, not
only against the instrument. The baseline was restored afterwards: promoting it to the served default is the
operator's call.
