# latchain r1: how to verify (operator runbook)

Nothing in r1 was compiled or run off-box. Follow the steps in order. Each one is a gate for the next.

## 0. Stage the directory on flan

Copy `out-latchain/latchain-r1/` to `/srv/qwen5090/patches/exllamav3/latchain-r1/`. The gate reads `$LC` from there. Renumber `rXXX-latchain-r1.sh` to the next R number: copy it to `/srv/qwen5090/rNNN-latchain-r1.sh` and set `RN=rNNN` (or edit the default).

## 1. Compile gates (CPU only, no --gpus, not inside another unit's lock per §15)

- Fast check of the new kernel:
  `docker run --rm --entrypoint bash -v $LC:/opt/latchain-r1 -v /tmp/lc-nvcc:/out tabbyapi:stack-r2 /opt/latchain-r1/quick-nvcc.sh /out`
  Expect `nvcc OK`, plus RR/SERVED register lines for the 8 rr instantiations and the served _128 ones. There should be no spill stores on the rr kernels: 512 threads/block caps them at 128 registers, and `st[32]` plus the prefetch registers fit.
- Full rebuild plus identity:
  `docker run --rm --entrypoint bash -v $LC:/opt/latchain-r1 -v /tmp/lc-out:/out tabbyapi:stack-r2 /opt/latchain-r1/cpu-build-check.sh`
  Expect:
  - `latchain r1 landed ... 8 rr kernels`;
  - `SASS IDENTITY OK`: every served CUDA function's SASS is unchanged, and the only new functions are the 8 `_rr` kernels;
  - `CPU PASS` twice: once with the host call-order check against the unpatched tree, once as installed (incl. flag parsing);
  - `CPU BUILD CHECK OK`.
- The patch stores blank context lines as empty lines. `install-latchain.sh` restores the leading space before `patch --fuzz=0`. `make-patch.sh` regenerates and checks the patch from `src/` + hcfast-r1 + moefast-r1 (round trip byte-identical, checked 2026-09-24 off-box).

## 2. The gate (one queued unit, ~3 h GPU estimate)

`rXXX-latchain-r1.sh` builds `tabbyapi:stack-latchain-r1` in the lock, then:

| step | what | pass / what a failure means |
|---|---|---|
| 1 parity (1 card) | `test_latchain_parity.py`: GDN rr vs served over 14 (bsz, rows) shapes × bf16/fp32 × history on/off, 12 chained steps + 6 graph replays each. It checks the kernel name that ran (ON must launch `_128_rr`). QSA compile variants (split stages 2-4 × combine 1-2 × div16) at the 6 served shapes, contexts 4k/32k | `PARITY PASS`. A GDN mismatch is a kernel bug in rr; a QSA mismatch means that variant is not numerics-preserving (drop it from `VARIANTS`) |
| 2 P0 | `bench_latchain_p0.py`: µs per launch served vs rr (36 L2-cold layer copies) and every QSA variant | Diagnostic. It prints the `QT choice` that the P1 QT arm uses; `none` drops the QT arm |
| 3 model parity (2 cards) | `lc_model_parity.py`: sha256 of every target/draft forward output over 24 decode iterations, OFF vs OFF (A/A) vs ALL at b1d3, b4d3, b8d1 | If the A/A matches and ALL differs, the unit stops. For the fork flags this is the check of the graph-site-order assumption (`Graph::capture_end` matching record_param sites in `cudaGraphGetNodes` order across two capture streams): if parity (step 1) passed and this fails, suspect the site order, not a kernel. If the A/A differs, the logits check is unusable on this build and only token-level identity (steps 5, 7) applies |
| 4 nsys | c1d3 and c8d1, OFF vs ALL, node mode | For off-box analysis: `out-latchain/tools/chain_kernels.py nsys-c1d3-ALL.sqlite <start> <end> 32` (window from `out-roofline/tools/window.py`), plus the attention/GDN segment spans (`gap.py` segments) to confirm the branches overlap |
| 5 P1 | untraced harness, 4k, arms OFF RR QF GF [QT] ALL, ROUNDS = a multiple of the arm count (6), order rotated so each arm runs first equally often, at c4d3, c8d1, c1d3 | Sequence hashes of every arm == the round's OFF, in the drafting AND the d0 cell. A single DIFFER rejects the arm |
| 6 summary | per arm vs OFF: pairs, mean ms and %, sign per shape; d0 cells reported | `P1-VERDICT <arm>: SPEED+HASHES OK` needs: no same-sign regression at c4d3/c8d1; a c1d3 same-sign regression only if ≤ 2 % and smaller in ms than a c4d3/c8d1 GAIN; at least one GAIN (gain clause); ≥ 5 pairs per shape |
| 7 fn_greedy | ref (daily) vs IMG ON (union of accepted arms; ALL if none) vs IMG OFF | 0 divergences on and off (gated). Also cuda:0 used MiB after the ON boot ≤ ref + 32 |
| 8 FINAL | per arm ACCEPT/REJECT with reasons; `FINAL stack entries: ...` | Accepted rows go to `flan/STACK.md` with their per-shape deltas (and the c1 cost if any); promotion is batched per §16 |

Budget: 6 arms × 6 rounds × 3 shapes = 108 P1 runs, plus 9 model-parity loads, 4 nsys runs and 3 boots. Running `ARMS="OFF RR QF ALL"` (dropping GF, the smallest estimate, and QT) cuts P1 to 4 × 8 × 3 = 96 runs at 8 rounds, or 4 × 4 × 3 = 48 with `ROUNDS=4`. ROUNDS=4 gives fewer than 5 pairs, so the gate will not accept on it. It is a cheaper look only. Which to run is the operator's call.

## 3. Reading a failure

- Build fails in `install-latchain.sh`: the compile error is in `build.log`. r1 was never compiled, so expect a typo-class error there first.
- `SASS IDENTITY FAIL` listing CHANGED served functions: the rebuild is not reproducing the base .so. Compare against a plain rebuild of stack-r2 without the patch before blaming r1.
- P1 hashes DIFFER for QF/GF only: the fork changed numerics. That means a real data race, e.g. a scratch buffer shared across the branches that the analysis missed. Look at `lc_model_parity`'s first differing forward.
- QF gains at c4/c8 but regresses c1d3 beyond 2 %: SM contention between the index GEMM branch and the qkv mgemm (156 + 20 CTAs vs 170 SMs). This is an r2 item: per-shape enable.
