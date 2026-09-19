# R366: Our own 32-row MoE decode envelope: correct, and no effect

Results directory on the serving host: `results/2026-09-16-r366-ourkernel`. Driver: [`scripts/r366-ourkernel.sh`](../../scripts/r366-ourkernel.sh). Date: 2026-09-16.


The hypothesis was that c4/c8 decode falls into a slow fallback path in the MoE dispatch, and that an envelope written
for exactly these shapes would recover it. The change is dispatch-only, so the gate is output identity — **PASS,
byte-identical** — and then the columns:

| | c1 | c2 | c4 | c8 |
| --- | --- | --- | --- | --- |
| control (`tabbyapi:qsa-devel`) | 0.152 / 0.146 / 0.146 | ~0.255 | ~0.48–0.58 | 0.485–1.246 |
| candidate (`tabbyapi:ourkernel`) | 0.153 / 0.145 / 0.145 | ~0.255 | ~0.48–0.58 | 0.485–1.238 |

Three runs per arm: TTFT agrees to within 0.002 s at c1/c2/c4 and 0.01 s at c8. Aggregate decode reads 248–252 at c4
and 288–294 at c8 on both arms. **The envelope is correct and buys nothing measurable**, which moves the c4/c8 question
away from dispatch and back to where the earlier analysis put it: the layer-split duty cycle.

### The measurement that came out of the control arm

24 c8 TTFT samples per arm give a spread the single-run comparisons never showed:

```
control    min 0.485  p50 0.844  max 1.246  ->  2.57x within ONE configuration
candidate  min 0.485  p50 0.858  max 1.238  ->  2.55x within ONE configuration
```

**c8 TTFT varies by 2.6x inside a single configuration**, so any single-run c8 TTFT comparison smaller than that is
noise. That includes the #246 result recorded above: control 1.249 s against feature-on 0.635 s is a ratio of 1.97,
*inside* the within-configuration spread. "The feature halves c8 TTFT" is therefore not supported; the prefill
columns in the same run moved 1.3-1.6 %, which is the size of the effect #246 produced here.

This is a metric-defect finding rather than an engine finding: **TTFT at high concurrency is dominated by queueing, so
it needs many runs or a median-and-spread report, never one sample per arm.** The r375 arms were rewritten around that
before they ran: two arms instead of three (feature-off is byte-identical to control, so it *is* the control), 512
forced tokens because the reading is TTFT, and four runs per concurrency with the spread printed.
