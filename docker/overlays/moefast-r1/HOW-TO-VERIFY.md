# HOW-TO-VERIFY.md (moefast r1)

> **Bars superseded 2026-09-24 by OPERATIONS §16 (stack track).** The effect-size bars below (P0 %, P1 ms) are diagnostic only. A bitwise-identical flag is accepted into the stack on identity (P1 hashes + fn_greedy) plus no same-sign regression at c4d3/c8d1 over ≥ 5 in-process pairs, and is promoted in a batched stack gate (`flan/r701-stack-gate.sh` pattern). Ledger: `flan/STACK.md`.

Written by the MoE-fast Opus sub-agent (2026-09-24). The operator saved it from the agent's final report, because the agent could not write .md files.

All GPU steps are the operator's. `moefast-gate.sh` (queued as `flan/r700-moefast-gate.sh`) runs §0-§5 in order inside the GPU queue lock:

1. parity
2. P0 (the round stops on KILL)
3. ncu
4. P1 ABAB
5. fn_greedy

`finish_restore` restores the daily at the end.

## §0 Build

```
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:slotfix-r1 -t tabbyapi:moefast-r1 .
```

The build must print both of these lines:
- "moefast r1 landed … 45 V3 kernels"
- "300 partition/ksplit cases … PASS"

Landing marker: `exllamav3_ext.moe_coop_v3_revision == 1`.

The CPU-only pre-check has already passed. It reads "identical after rebuild: 193" of 193:

```
docker run --rm -v <pkg>:/opt/moefast-r1:ro -v <out>:/out --entrypoint bash tabbyapi:slotfix-r1 /opt/moefast-r1/cpu-build-check.sh
```

## §1 Parity (bitwise, one card, served env)

```
python3 /opt/moefast-r1/test_moefast_parity.py --model /models/qwen3.8-flash-next-exl3-2.50bpw-r0b0tlab --json /results/parity.json
```

What it compares: the reference (V3 unset) against V3=1 and V3=2. It checks the storage bits of `out`, `act` and `d_out` for active slots, with sentinels.

What it covers:
- real K2 (layer 20) and K3 (layer 3) weights, plus synthetic K2/3/4;
- rows 1/2/3/4/5/8/12/13/16;
- routings served-D, same-top10, pool24 and inactive (including an empty row);
- shared expert on and off;
- a dispatch check that the V3 kernels actually ran (profiler names);
- a stress test: 200 back-to-back launches per mode at rows 4/12/16, with the kernels resetting their own counters as in serving.

The stress test is a weak race detector. The real ordering check is P1's sequence-hash identity over 32 steps.

**Pass:** "PARITY PASS", exit 0. Any FAIL stops the round. Optionally re-run with `EXL3_MOE_COOP_KSPLIT=2`, because the launcher caches KSPLIT per process.

## §2 P0 microbench (events and per-kernel)

```
python3 /opt/moefast-r1/bench_moefast_p0.py --model … --out /results/p0 --extra
```

**Setup:**
- **Cells:** K2 and K3 × (r1 D10 control, r4 D28, r16 D77), plus the D sweep.
- **Arms:** ref, m1 and m2, interleaved over 5 rotating rounds, timed with d0 Timer us_each.
- **Per kernel:** rot/a/b medians from torch.profiler device intervals.

**Pre-registered bars.** Each decision cell (K2/K3 at r4 D28 and at r16 D77) takes the better of m1 and m2.
- **PASS:** ratio ≤ 0.80 at all four decision cells.
- **KILL:** ratio > 0.95 at all four. The round stops.
- **MIXED:** anything else. P1 still runs if the predicted step saving is ≥ 0.4 ms at c1d3 or c8d1; the bench prints it from 25 K2 + 23 K3 layers.
- **Control:** r1 D10 within ±3 % in every arm, since V3 does not engage at bsz 1. Otherwise re-run.

**Diagnostic predictions** (estimates, stated up front):
- The A kernel at r4 D28 drops ≥ 10 % under m1. A drop under 5 % refutes the latency diagnosis.
- The B kernel at K2 r4 D28 drops ≥ 25 % under m2.
- Most likely verdict: MIXED, with r4 m2 at about 0.72-0.85 and r16 at about 0.75-0.90.

## §2b ncu per-PC attribution (ref and the chosen mode)

The gate runs this step: SourceCounters and WarpStateStats, served env, card 0, `--cache-control all`.

- **Why cache-control all:** kernel replay would otherwise warm the r4 weights (35-52 MB) in L2 on later passes.
- **Analysis rules**, with the review's corrections:
  - attribute barrier stalls to the preceding `BAR.SYNC`;
  - attribute membar stalls to the preceding MEMBAR;
  - rank only the `stall_X` columns.
- **Expected in V3:**
  - long_sb at the HMMA falls to the one-slice-old `LDG.E.CONSTANT`;
  - no MOV consumes a weight load;
  - `DEPBAR` waits stay small;
  - B r4's l.429 barrier share drops with m2.

## §3 P1 untraced ABAB (decision)

**Setup:**
- `probes/r465/profile_decode_events.py` at 4k context, in the moefast image with the served env.
- Arms: A = flag unset; B = `-e EXL3_MOE_COOP_V3=<mode from P0>`. Order OFF, ON, OFF, ON.
- Shapes: c4d3 (b4 d3), c8d1 (b8 d1), and c1d3 (b1 d3, informational).

**Bars:**
- **PASS:** OFF−ON ≥ 1.0 ms/iterate in both pairs at c4d3 AND c8d1.
- **KILL:** < 0.4 ms at both.
- **c1d3:** informational; predicted 0.4-1.1 ms.
- **Sequence hashes:** must be IDENTICAL for ON vs OFF in every pair. Any difference is a numerics or ordering bug and stops promotion.

## §4 Served-launcher greedy identity

`fn_greedy.py` against :8022 across three boots:
1. ref (the daily);
2. `IMG=tabbyapi:moefast-r1 EXTRA_ENV_ADD=EXL3_MOE_COOP_V3=<mode>`;
3. `IMG=tabbyapi:moefast-r1` with the flag unset.

**Pass:** divergences = 0 for both "on vs ref" and "off vs ref".

## §5 After

Before FINDINGS, run the independent round review. Promotion also needs the canonical five gates.
