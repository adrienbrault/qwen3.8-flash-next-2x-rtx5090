# R365: upstream #246 changes numerics for a prefill gain within noise; #290 is output-neutral

Results directory on the serving host: `results/2026-09-16-r365-kernels`. Driver: [`scripts/r365-kernels.sh`](../../scripts/r365-kernels.sh). Date: 2026-09-16.

## Upstream #246 changes numerics for a prefill gain within noise


Its assessment said the claim is prefill-only and that ordinary c1/c4/c8 decode cannot use it, so the columns to read
were TTFT at depth and output identity. Both are now measured, and they point the same way.

| arm | output vs control | TTFT ctx 0 | TTFT ctx 30k |
| --- | --- | --- | --- |
| control (unpatched) | — | 0.146 s | 1.428 s |
| patched, feature off | **byte-identical** | 0.146 s | — |
| patched, `EXL3_MOE_ROUTE_PACKED=1` | **all six responses differ** | 0.146 s | 1.403 s (−1.8 %) |

Two readings. First, the disabled path is identical to control, which is what makes the arms a valid comparison.
Second, the enabled path **reorders numerics** and buys, at most, 1.8 % TTFT at 30k in a single run — inside
run-to-run noise on this box, and against outputs that this repository fingerprints. Rejected: a feature that changes
what the model says needs a much larger gain than that to be worth re-validating quality behind it.

The in-run gate line said `FAIL/NOT-RUN control vs patched-off` and was wrong — the same glob-`cmp` defect that
reported a false negative for #290, in a script copy that predated the fix. The gate was recomputed from the captures
with `lib/greedy-compare.sh`; the numbers above are the recomputation.

## Upstream #290's memory fix is output-neutral


Three arms, one native rebuild each from the same v1.5.0 source with a different patch set applied: `unpatched`,
`+OOB fix`, `+OOB fix +reduction`. The question its assessment left open was whether a memory-safety fix in an API
this model does not route through changes anything; the gate is therefore **output identity**, not speed.

| arm | extension sha256 (first 16) | captured | dirhash |
| --- | --- | --- | --- |
| unpatched | `ec7270959395df30` | 6 responses | `18e30f17883a38eb` |
| +fix | `294407485b5b792c` | 6 responses | `18e30f17883a38eb` |
| +fix+reduction | `fb0f3690b357c436` | 6 responses | `18e30f17883a38eb` |

**All three identical.** The three extension hashes differ, which is the precondition that makes the identity
mean anything: three arms sharing one binary would have been an identity result about nothing. Aggregate decode,
fix vs unpatched: c1 220.5 vs 222.3, c4 381.8 vs 381.8, c8 319.3 vs 319.7, i.e. within run-to-run noise. So the fix
can be adopted on correctness grounds with no behavioural or throughput cost.

**Two caveats, both about what these numbers are not.** First, the arms are comparable *to each other* and not to the
served figures: `kernel290:*` is a CUDA-devel image built from v1.5.0 plus the port, which is a different image from
the served `tabbyapi:qsa-cid-pr337`, and the c4 column here (381.8) sits above the served configuration's 250 without
a policy and 338 with one — a cross-image comparison would be a category error. Second, this run's first attempt at
the gate reported `FAIL/NOT-RUN` for both arms and was **wrong**: the control arm captured one response and the
treatments six, because the control ran before a capture bug was fixed, and the comparison then used `cmp` on
mismatched file sets. The gate was recomputed from paired captures rather than read from the log line — see GOTCHAS 11
and [`scripts/r371-290-identity.sh`](../../scripts/r371-290-identity.sh).
