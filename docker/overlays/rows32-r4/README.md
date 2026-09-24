# rows32-r4

`tabbyapi:stack-r3-rows32` is the image served since 2026-09-25 01:05 CEST ([R717, R717b, R717c](../../../bench/results/r717-rows32.md)). It is `tabbyapi:stack-r3` ([`../stack-r3/README.md`](../stack-r3/README.md)) with [`rows32-r4.patch`](rows32-r4.patch) applied to the installed `exllamav3` package and one rebuild of `exllamav3_ext` for `sm_120`. The patch changes host code only. [`install-rows32.sh`](install-rows32.sh) checks that the base is `stack-r3` with densegemm r2 and the lcguard and without rows32, hashes the SASS of every function of the base `.so`, applies the patch at `--fuzz=0` after a dry run (any `.rej` fails), rebuilds with the base image's `DGV2_NVCC_DEFS`, fails unless every base function keeps its SASS and none is added, asserts the landing markers with every flag off and runs the CPU tests ([`test_rows32_cpu.py`](test_rows32_cpu.py) and `stack-r3`'s lcguard test on the patched tree).

Build, from the repository root, after the `stack-r3` chain:

```sh
cd docker/overlays/rows32-r4
B=tabbyapi:stack-r3
docker build -f Dockerfile.box --build-arg BASE=$B \
  --build-arg BASE_ID=$(docker image inspect $B --format '{{.Id}}') \
  --build-arg STACK_INCLUDE="$(docker image inspect $B --format '{{index .Config.Labels "local.stack.include"}}')" \
  --build-arg DGV2_NVCC_DEFS="$(docker image inspect $B --format '{{index .Config.Labels "local.stack.dgv2_defs"}}')" \
  --build-arg PATCH_SHA=$(sha256sum rows32-r4.patch | cut -c1-64) --build-arg MAX_JOBS=4 -t tabbyapi:stack-r3-rows32 .
# the build prints the landing line:
# rows32 r4 landed: moe_rows32_revision 2 on stack-r3 (INCLUDE 'mf3 dg2'); flags default off (caps 16, dense (0, 0)); ...
```

The served build ran inside R717 ([`landed.txt`](../../../bench/results/2026-09-24-r717-rows32-r4/landed.txt), [`install-sass-identity.txt`](../../../bench/results/2026-09-24-r717-rows32-r4/install-sass-identity.txt)): 1,633 functions, 1,633 unchanged, 0 added.

Every change is off unless its environment key is set. The served launcher changes these, on top of the 39 keys of `stack-r3`:

| key | value served | what it does |
| --- | --- | --- |
| `EXL3_DENSE_ROWS32` | `1` | densegemm r2's one-pass 17-to-32-row twins of the dense gemm and mgemm (in `stack-r3` already); with this patch the densegemm side region on latchain's QSA branch is 4 MiB instead of 2 |
| `EXL3_MOE_COOP_ROWS32` | `1` | the routed-expert cooperative MoE decode in one launch up to 32 rows; each output equals two served 16-row calls |
| `EXL3_SHARED_EXPERT_ROWS32` | `1` | the shared expert's `BC_GatedMLP` graphs and statics sized for 32 rows; `EXL3_MOE_COOP_ROWS32` without it is refused |
| `EXL3_MOE_COOP_V3_MAP` | `2-4:2,17-32:2` (was `2-4:2`) | names mode 2 at 17 to 32 rows, where moefast r3's r2 kernels do not run |

The launcher also sets the draft policy `[[4, 3], [8, 2]]`, which makes 6, 7 and 8 jobs verify at depth 2 (18, 21 and 24 rows), and the page pool 983,040. Rollback is `IMG=tabbyapi:stack-r3 CACHE=999424 DRAFT_POLICY='[[4, 3], [5, 2], [8, 1]]'` with the `stack-r3` 39-key `EXTRA_ENV`.

[`ANALYSIS.md`](ANALYSIS.md) is the analysis written before the gate, [`HOW-TO-VERIFY.md`](HOW-TO-VERIFY.md) the gate's steps. [`test_rows32_parity.py`](test_rows32_parity.py) and [`rows32_extra_parity.py`](rows32_extra_parity.py) are the kernel parity tests the gate ran; [`rows32_decide.py`](rows32_decide.py) reads a gate's results directory into its decision; [`greedy_conc.py`](greedy_conc.py) compares greedy output at 8 streams against 1 stream; [`make-patch.sh`](make-patch.sh) regenerates the patch from reconstructed trees and [`verify-patch.sh`](verify-patch.sh) checks by exit code that it applies at fuzz 0 to the `stack-r3` series with moefast r3 and is rejected by the trees without it.
