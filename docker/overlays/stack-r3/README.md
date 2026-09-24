# stack-r3

`tabbyapi:stack-r3` was the image served from 2026-09-24 22:10 CEST to 2026-09-25 01:05 CEST ([R716b, R716c](../../../bench/results/r716b-stack-r3.md)) and is the base of `tabbyapi:stack-r3-rows32` ([`../rows32-r4/README.md`](../rows32-r4/README.md)). It is `tabbyapi:stack-r2` ([`../stack-r2/README.md`](../stack-r2/README.md)) with one patch series applied to the installed `exllamav3` package and one rebuild of `exllamav3_ext` for `sm_120`. [`install-stack.sh`](install-stack.sh) checks that the base is `stack-r2`, applies the series with [`series/apply-series.sh`](series/apply-series.sh) (each patch's SHA-256 against [`series/SERIES`](series/SERIES), a dry run and the apply at `--fuzz=0`, any `.rej` fails), rebuilds, checks that every function of the base `.so` has unchanged SASS and that the new kernels are exactly the selected set, bounds every new kernel's stack frame at its served reference plus 64 B, asserts the landing markers with every flag off, and runs the CPU tests.

| order | patch | selected by | what it carries | component result |
| --- | --- | --- | --- | --- |
| 01 | `01-hcfast-r2-on-stack-r2.patch` | always | hcfast r1 → r2 (`EXL3_HC_MIX_V3=2`, per-row-count tables); the diff from `stack-r2` to the tree R702 measured | [R702](../../../bench/results/r702-hcfast-r2.md), [`../hcfast-r2`](../hcfast-r2/) |
| 02 | `02-latchain-r1b.patch` | always | latchain r1 with its `bindings.cpp` revision line re-anchored so it composes with moefast r3; every other hunk byte-identical to [`../latchain-r1/latchain-r1.patch`](../latchain-r1/latchain-r1.patch) | [R712](../../../bench/results/r712-latchain-r1.md) |
| 03 | `03-moefast-r3.patch` | `mf3` | [`../moefast-r3/moefast-r3.patch`](../moefast-r3/moefast-r3.patch), verbatim | [R713](../../../bench/results/r713-moefast-r3.md) |
| 04 | `04-densegemm-r1.patch` | `dg1` | [`../densegemm-r1/densegemm-r1.patch`](../densegemm-r1/densegemm-r1.patch), verbatim; not in the served image | [R710b](../../../bench/results/r710b-densegemm-gv.md) |
| 05 | `05-densegemm-r2.patch` | `dg2` | [`../densegemm-r2/densegemm-r2.patch`](../densegemm-r2/densegemm-r2.patch), verbatim | [R714](../../../bench/results/r714-densegemm-r2.md) |
| 06 | `06-dense-lcguard.patch` | `dg1` or `dg2` | a dense V2 call on latchain's QSA side branch, detected by its lock offset, uses the upper half of the gemv counters and a separate 2 MiB gemm slot region | [`ANALYSIS.md`](ANALYSIS.md) §3 |

[`series/subset-matrix.txt`](series/subset-matrix.txt) is the output of [`series/check-subsets.sh`](series/check-subsets.sh): every allowed INCLUDE subset applies at fuzz 0, the same patches in a permuted order give a byte-identical tree, and the original latchain r1 and moefast r3 patches reject each other in both orders.

Build, from the repository root, after the `stack-r2` chain:

```sh
cd docker/overlays/stack-r3
docker build -f Dockerfile.box --build-arg BASE=tabbyapi:stack-r2 --build-arg INCLUDE="mf3 dg2" \
  --build-arg DGV2_NVCC_DEFS= --build-arg SERIES_SHA=$(sha256sum series/SERIES | cut -c1-64) \
  --build-arg MAX_JOBS=4 -t tabbyapi:stack-r3 .
# the build prints the landing line:
# stack-r3 landed: hcfast r2 (hc_mix_v3_revision 2, build r2), latchain 1 (latchain-r1), moefast 3, densegemm 2 + lcguard; every flag default off
```

The served build applied patches 01, 02, 03, 05 and 06 ([`build.txt`](../../../bench/results/2026-09-24-r716b-stack-r3/build.txt)). `DGV2_NVCC_DEFS` is empty: R714's compile bisect chose plain r2.

Every change in the series is off unless its environment key is set. The served launcher sets these, on top of the 23 keys `stack-r2` inherited from `slotfix-r1`:

| key | component | value served |
| --- | --- | --- |
| `EXL3_HC_MIX_V3` | hcfast r2 | `2` (was `1` in `stack-r2`) |
| `EXL3_HC_MIX_V3_DOTS_B`, `_UP_B` | hcfast r2 | `1:1,4:2,32:4`, `1:1,8:4,32:8` (were `2`, `8`) |
| `EXL3_HC_MIX_V3_DOTS_J`, `_DOTS_PF`, `_UP_Q`, `_PDL` | hcfast r2 | `1:4,32:8`, `1:1,32:0`, `1:4,8:2,32:4`, `0` |
| `EXL3_LC_GDN_RR` | latchain | `1` |
| `EXL3_LC_QSA_FORK` | latchain | `1` |
| `EXL3_LC_QSA_SPLIT_STAGES`, `_COMBINE_STAGES`, `_DIV16` | latchain | `2`, `1`, `1` |
| `EXL3_DENSE_V2` | densegemm r2 | `1` |
| `EXL3_MOE_COOP_V3` | moefast r3 | `3` (was `2`) |
| `EXL3_MOE_COOP_V3_MAP`, `EXL3_SHARED_EXPERT_EARLY` | moefast r3 | `2-4:2`, `1` |

Rollback is `IMG=tabbyapi:stack-r2` with R701's 27-key `EXTRA_ENV`. With those 27 keys, `tabbyapi:stack-r3` computes the same logits as `tabbyapi:stack-r2` at every served decode shape (R716c's R2 arm).

[`ANALYSIS.md`](ANALYSIS.md) is the composition analysis written before the gate, [`HOW-TO-VERIFY.md`](HOW-TO-VERIFY.md) the gate's steps. [`tools/`](tools/) holds the gate's readers: `p1_stack.py` (stall-excluded P1 deltas and leave-one-out marginals), `served_stack.py` (headroom, safety and canonical-gate ratios), `greedy_streams.py` (the multi-stream greedy rule in [`docs/PROMOTION.md`](../../../docs/PROMOTION.md#identity-at-every-served-shape-since-2026-09-24)), `sass_identity_stack.py`, `stack_bound.py`, `compile_bar.py` and `landing.py` (install checks). [`tests/`](tests/) holds the kernel parity tests of each component and `tests/latchain/lc_model_parity.py`, the logits-level model parity R716c ran at every served shape.
