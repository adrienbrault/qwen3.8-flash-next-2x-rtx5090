#!/usr/bin/env python3
"""rows32 r4 (rows32 r3 rebased on stack-r3), CPU-only checks (no GPU; parts 1-3 and 5 need no torch).

1. Index model of the V2/V3 coop kernels at 1..32 rows (exl3_moe_coop_kernel.cuh build_runs / read_run / rot kernel,
   exl3_moe_coop_v2_kernel.cuh and exl3_moe_coop_v3_kernel.cuh tile_a / tile_b item decode and completion counters,
   exl3_moe_coop.cu geometry and scratch layout), replayed from the source by hand for random and adversarial
   routings (all distinct, all rows on the same 10 experts = runs of 8 slots, a 24-expert pool, inactive slots,
   empty token rows). For every launch it checks:
     - every active (slot, projection, column) of stage A and every active (slot, column) of stage B is computed
       exactly once; runs hold <= ROWS slots of one expert; one thread per slot suffices (slots <= THREADS);
     - every (slot, 128-chunk) counter of A and every (row, 128-chunk) counter of B receives exactly the arrivals
       its last-arrival test expects, and exactly one arrival is elected last;
     - every counter, scratch row and run-table index stays inside the scratch the caller sized (slots_max,
       rows_max, the python allocation formula), and the run table fits the int32 scratch;
     - token rows without an active slot are written exactly once per 128-chunk by the rotation kernel;
     - geometry: at 17..32 rows x top-10 both stages pick the wide tile, the geometry of a served 16-row call, so
       the per-row arithmetic (k partition of each slot's GEMV, the warp-ordered down reduction over a row's own
       slots) is the one a 16-row call applies to that row.
   This is a model of the code, written from it by hand. It catches partition and sizing mistakes, not compiler
   behaviour; test_rows32_parity.py checks the bits on the GPU.
2. Scratch sizes: block_sparse_mlp.py's allocation formula vs exl3_moe_coop_ctr_len / exl3_moe_coop_prepare at
   MAX_BSZN 16 and 32, and the VRAM delta of the flags (estimate, printed).
3. Patch hygiene (when rows32-r4.patch is next to this file): exactly the expected files, no device code added
   (host-only change: the install checks every served function's SASS unchanged and nothing added), the served
   <= 16-row condition short-circuits before the new env read; the r4 additions are present (start_shared bound,
   the lcguard side region sized for 32-row slots, the side-refusal log line).
4. Flags (only where the patched exllamav3 imports, i.e. inside the image): module constants and the C++ cap under
   every flag combination, strict 0/1 parsing, EXL3_MOE_COOP_ROWS32 without EXL3_SHARED_EXPERT_ROWS32 refused.
5. Source model of the patched stack-r3 tree (--tree DIR, else the installed package):
   - MoE mode per launch: moefast r3's EXL3_MOE_COOP_V3 / _V3_MAP parser and the launcher's mode-3 preconditions
     (R2_MAX_ROWS / R2_MAX_SLOTS read from exl3_moe_coop_r2_kernel.cuh, the early return and the "v3 = 2" fallback
     read from exl3_moe_coop.cu), replayed for 1..32 rows with the flag off and on: the A env (stack-r3 union,
     MAP 2-4:2) and the B env (MAP 2-4:2,17-32:2) pick the same mode at every row count, mode 3 is unreachable
     above 16 rows, and the B MAP's 17-32 entry is exactly the fallback the code already takes;
   - lcguard side region at 17-32 rows: index_qk's twin (k 2560, n 640, tag 2, 32-row slots) fits the side region
     at the largest grid the autotuner can pick (min(tiles, 170 SMs)), and would NOT have fit stack-r3's 2 MiB.

Run: python3 test_rows32_cpu.py [--require-flags] [--tree <exllamav3 package dir>]
     (exit 0 and "CPU TESTS PASS" only if everything holds; --require-flags: part 4 must run, as in the gate,
     instead of being skipped where exllamav3 does not import)
6. greedy_conc.py's condition-2 verdict on synthetic records (when greedy_conc.py is next to this file): equal rates
   PASS, one more B divergence than the worse A half x 2 FAILS, an early B divergence A never shows is reported only (R717 review), an error
   FAILS, a pair across prompt sets is unusable.
"""
import os
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Flash-Next MoE (d0_microbench constants; config.json of qwen3.8-flash-next)
TOPK, NE, H, I = 10, 512, 2560, 640
Hi, Ho = H, H
ROWS, THREADS, MAX_SLOTS, WK = 8, 512, 512, 16
COLS = 32


def pick_wide(kslices, slots):
    # exl3_moe_coop.cu moe_coop_pick_wide, Blackwell, EXL3_MOE_COOP_WIDE unset
    return kslices >= 256 or (kslices >= 128 and slots >= 32) or slots >= 128


def tile_cols(wide):
    return 4 * COLS if wide else COLS


def tile_gpc(wide):
    return 1 if tile_cols(wide) >= 128 else 128 // tile_cols(wide)


def ctr_len(slots_max, bsz_max):
    # exl3_moe_coop.cuh exl3_moe_coop_ctr_len
    return slots_max * (I // 128) + bsz_max * (Ho // 128) + 2 + (slots_max + 1) + slots_max


def py_ctr_alloc(max_bszn):
    # block_sparse_mlp.py: coop_ctr = (bszn_rows * (I // 128) + MAX_BSZN * (Ho // 128) + 2 * bszn_rows + 3,)
    bszn_rows = max_bszn * TOPK
    return bszn_rows * (I // 128) + max_bszn * (Ho // 128) + 2 * bszn_rows + 3


class Fail(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise Fail(msg)


def build_runs(local, slots, slots_max):
    """exl3_moe_coop_kernel.cuh build_runs, one 'thread' per t < THREADS. local[s] = local expert or -1."""
    check(slots <= THREADS, f"{slots} slots > {THREADS} threads: build_runs would skip slots")
    check(slots <= MAX_SLOTS, "slots > MAX_SLOTS (int16 sh_order/sh_local)")
    sh_order = [None] * MAX_SLOTS
    n_active = 0
    for t in range(slots):
        my_e = local[t]
        if my_e >= 0:
            rank = sum(1 for u in range(slots) if local[u] >= 0 and (local[u] < my_e or (local[u] == my_e and u < t)))
            check(sh_order[rank] is None, "rank collision")
            sh_order[rank] = t
            n_active = max(n_active, rank + 1)
    is_start = [0] * THREADS
    order = [None] * (slots_max)
    for t in range(n_active):
        e_t = local[sh_order[t]]
        first = sum(1 for u in range(slots) if local[u] >= 0 and local[u] < e_t)
        is_start[t] = 1 if (t == 0 or local[sh_order[t - 1]] != e_t or (t - first) % ROWS == 0) else 0
        check(t < slots_max, "order index past slots_max")
        order[t] = sh_order[t]
    # exclusive prefix over the block (warp scan + warp totals): run_id = number of starts before t
    run_start = [None] * (slots_max + 1)
    total = 0
    for t in range(THREADS):
        if is_start[t]:
            run_start[total] = t
            total += 1
    check(total <= slots_max, "more runs than run_start entries")
    run_start[total] = n_active
    runs = []
    for r in range(total):
        rows = [order[i] for i in range(run_start[r], run_start[r + 1])]
        check(1 <= len(rows) <= ROWS, f"run of {len(rows)} slots")
        check(len({local[s] for s in rows}) == 1, "run mixes experts")
        runs.append(rows)
    # the table must fit the int32 scratch after the counters: [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    return runs, n_active


def launch_model(sel, rw, slots_max, rows_max, gated=True, ksplit=1, merge_v3=True):
    """One V2/V3 launch at bsz = len(sel). Returns the geometry. Raises Fail on any violation."""
    bsz = len(sel)
    slots = bsz * TOPK
    check(slots <= slots_max and bsz <= rows_max, "batch exceeds the scratch (exl3_moe_coop_run would refuse)")
    local = [sel[s // TOPK][s % TOPK] if rw[s // TOPK][s % TOPK] != 0 else -1 for s in range(slots)]
    active_of_row = [sum(1 for k in range(TOPK) if local[r * TOPK + k] >= 0) for r in range(bsz)]
    nproj = 2 if gated else 1
    wide_a = pick_wide(Hi // 16, slots)
    wide_b = pick_wide(I // 16, slots)
    merge_b = merge_v3 and not wide_b             # V3 mode 2
    a_global = bsz > 1
    check(a_global, "the model covers bsz > 1 (V3 engages only there)")
    runs, n_active = build_runs(local, slots, slots_max)
    check(sum(len(r) for r in runs) == n_active == sum(1 for x in local if x >= 0), "runs do not cover the active slots")

    # rotation kernel: warp per (slot, chunk, projection); empty rows written by their first slot's warps
    chunks = Hi // 128
    empty_written = {}
    for item in range(slots * chunks * nproj):
        s = item // (chunks * nproj)
        rem = item % (chunks * nproj)
        c = rem // nproj
        if local[s] < 0:
            row = s // TOPK
            if rem % nproj == 0 and s % TOPK == 0 and active_of_row[row] == 0:
                for oc in range(c, Ho // 128, chunks):
                    empty_written[(row, oc)] = empty_written.get((row, oc), 0) + 1
            continue
        check(s < slots_max, "had_g/had_u row past slots_max")
    for r in range(bsz):
        if active_of_row[r] == 0:
            for oc in range(Ho // 128):
                check(empty_written.get((r, oc), 0) == 1, f"empty row {r} chunk {oc} written {empty_written.get((r, oc), 0)}x")

    # stage A
    TC = tile_cols(wide_a)
    GPC = tile_gpc(wide_a)
    ng = I // TC
    ctr_a = {}
    last_a = {}
    cover_a = {}
    count = len(runs) * ng * ksplit * nproj
    for item in range(count):
        is_gate = gated and item % nproj == 0
        rem = item // nproj
        ks = rem % ksplit
        rem2 = rem // ksplit
        run_idx, group = rem2 // ng, rem2 % ng
        rows = runs[run_idx]
        chunk = group // GPC
        for s in rows:
            key = (s, is_gate, ks, group)
            cover_a[key] = cover_a.get(key, 0) + 1
            ci = s * (I // 128) + chunk
            check(ci < slots_max * (I // 128), "ctr_a index past ctr_a_len")
            old = ctr_a.get(ci, 0)
            ctr_a[ci] = old + 1
            if old == GPC * nproj * ksplit - 1:
                last_a[ci] = last_a.get(ci, 0) + 1
            check((s + ks * slots) < slots_max * max(1, ksplit), "gu row past the scratch")
    for s in range(slots):
        if local[s] < 0:
            continue
        for p in ([True, False] if gated else [False]):
            for g in range(ng):
                for ks in range(ksplit):
                    check(cover_a.get((s, p, ks, g), 0) == 1, f"A: slot {s} proj {p} group {g} covered {cover_a.get((s, p, ks, g), 0)}x")
        for chunk in range(I // 128):
            ci = s * (I // 128) + chunk
            check(ctr_a.get(ci, 0) == GPC * nproj * ksplit, f"A counter {ci}: {ctr_a.get(ci, 0)} arrivals")
            check(last_a.get(ci, 0) == 1, f"A counter {ci}: {last_a.get(ci, 0)} last arrivals")

    # stage B (V3: merged narrow chunks when merge_b, else V2's item per tile)
    TCb = tile_cols(wide_b)
    GPCb = tile_gpc(wide_b)
    ITEM = 128 if merge_b else TCb
    ngb = Ho // ITEM
    ctr_b, last_b, cover_b = {}, {}, {}
    for item in range(len(runs) * ngb * ksplit):
        ks = item % ksplit
        rem = item // ksplit
        run_idx, g = rem // ngb, rem % ngb
        rows = runs[run_idx]
        arrivals = GPCb if merge_b else 1
        chunk = g if merge_b else g // GPCb
        for s in rows:
            check(s + ks * slots < slots_max * max(1, ksplit), "d_out row past the scratch")
            key = (s, ks, g)
            cover_b[key] = cover_b.get(key, 0) + 1
        for s in rows:
            row = s // TOPK
            n = active_of_row[row]
            bi = row * (Ho // 128) + chunk
            check(bi < rows_max * (Ho // 128), "ctr_b index past ctr_b_len")
            old = ctr_b.get(bi, 0)
            ctr_b[bi] = old + arrivals
            if old == n * GPCb * ksplit - arrivals:
                last_b[bi] = last_b.get(bi, 0) + 1
    for r in range(bsz):
        n = active_of_row[r]
        for chunk in range(Ho // 128):
            bi = r * (Ho // 128) + chunk
            if n == 0:
                check(bi not in ctr_b, "B counter of an empty row touched")
                continue
            check(ctr_b[bi] == n * GPCb * ksplit, f"B counter row {r} chunk {chunk}: {ctr_b[bi]} arrivals")
            check(last_b.get(bi, 0) == 1, f"B counter row {r} chunk {chunk}: {last_b.get(bi, 0)} last arrivals")
    for s in range(slots):
        if local[s] >= 0:
            for g in range(ngb):
                for ks in range(ksplit):
                    check(cover_b.get((s, ks, g), 0) == 1, f"B: slot {s} tile {g} covered {cover_b.get((s, ks, g), 0)}x")
    check(all(local[s] >= 0 for (s, _, _) in cover_b), "B computed an inactive slot")
    return wide_a, wide_b, merge_b


def routings(rows, rng):
    out = []
    for D in sorted({TOPK, min(NE, max(TOPK, round(4.6 * rows))), min(NE, TOPK * rows)}):
        perm = rng.sample(range(NE), D)
        out.append((f"served-D{D}", [[perm[(r * TOPK + j) % D] for j in range(TOPK)] for r in range(rows)], None))
    same = rng.sample(range(NE), TOPK)
    out.append(("same-top10", [list(same) for _ in range(rows)], None))
    pool = rng.sample(range(NE), 24)
    out.append(("pool24", [rng.sample(pool, TOPK) for _ in range(rows)], None))
    sel = [rng.sample(range(NE), TOPK) for _ in range(rows)]
    mask = [[rng.random() < 0.3 for _ in range(TOPK)] for _ in range(rows)]
    mask[-1] = [True] * TOPK
    out.append(("inactive", sel, mask))
    sel = [rng.sample(range(NE), TOPK) for _ in range(rows)]
    out.append(("empty-rows", sel, [[r % 2 == 1] * TOPK for r in range(rows)]))
    return out


def part1():
    n = 0
    geo = {}
    for max_bszn in (16, 32):
        slots_max, rows_max = max_bszn * TOPK, max_bszn
        check(ctr_len(slots_max, rows_max) <= py_ctr_alloc(max_bszn), f"python coop_ctr too small at MAX_BSZN {max_bszn}")
        for rows in range(2, max_bszn + 1):
            for seed in range(3):
                rng = random.Random(7919 * rows + 104729 * seed + max_bszn)
                for name, sel, mask in routings(rows, rng):
                    rw = [[0.0 if (mask and mask[r][k]) else 0.1 for k in range(TOPK)] for r in range(rows)]
                    try:
                        g = launch_model(sel, rw, slots_max, rows_max)
                    except Fail as e:
                        raise Fail(f"MAX_BSZN {max_bszn} rows {rows} {name} seed {seed}: {e}")
                    geo.setdefault(rows, set()).add(g)
                    n += 1
    # a 17..32-row call on the served scratch (MAX_BSZN 16) must be refused, not run
    for rows in (17, 24, 32):
        try:
            launch_model([[0] * TOPK] * rows, [[0.1] * TOPK] * rows, 16 * TOPK, 16)
            raise Fail(f"{rows} rows accepted on 16-row scratch")
        except Fail as e:
            if "accepted" in str(e):
                raise
    ref16 = geo[16]
    for rows in range(17, 33):
        check(geo[rows] == ref16 == {(True, True, False)},
              f"geometry at {rows} rows {geo[rows]} differs from the served 16-row call {ref16}")
    print(f"[cpu] index model: {n} launches at 2..32 rows (MAX_BSZN 16 and 32) OK; geometry at 17..32 rows = "
          f"16 rows = wide A, wide B, no merge; 17/24/32 rows refused on 16-row scratch", flush=True)


def part2():
    lines = []
    for mb in (16, 32):
        b = mb * TOPK
        sizes = {
            "moe1_temp_hidden (half)": max(64, b) * Hi * 2,
            "moe1_temp_interm (half; x2 if fp32)": max(64, 2 * b) * I * 2,
            "moe1_temp_activa (half)": max(32, b) * I * 2,
            "moe1_temp_output (fp32)": max(32, b) * Ho * 4,
            "moe1_temp_hidden_u (half)": b * Hi * 2,
            "moe1_coop_ctr (int32)": py_ctr_alloc(mb) * 4,
            "moe1_out_bszn (fp32)": mb * H * 4,
        }
        lines.append((mb, sizes))
    d = {k: lines[1][1][k] - lines[0][1][k] for k in lines[0][1]}
    per_dev = sum(d.values())
    sh_layer = 16 * H * 4            # sh_exp_t (1, MAX_BSZN, H) fp32 per MoE layer: +16 rows
    print("[cpu] VRAM delta per device, flags on (estimate: allocation formulas of block_sparse_mlp.py / mlp.py):")
    for k, v in d.items():
        print(f"[cpu]   {k:38s} +{v / 2**20:6.2f} MiB")
    print(f"[cpu]   sh_exp_t per MoE layer                 +{sh_layer / 2**20:6.2f} MiB x 24-25 layers per card = "
          f"+{25 * sh_layer / 2**20:.1f} MiB")
    print(f"[cpu]   BC_GatedMLP >16-row buffers: views of the shared (2, 32, w) statics, +(2*16*(H + I_sh))*2 B once")
    print(f"[cpu]   total excl. densegemm's own workspace (+4 MiB slots and, with EXL3_LC_QSA_FORK, +2 MiB side slots with "
          f"EXL3_DENSE_ROWS32) and activations: "
          f"~{(per_dev + 25 * sh_layer) / 2**20:.1f} MiB per card")


def part3():
    p = HERE / "rows32-r4.patch"
    if not p.is_file():
        print("[cpu] patch hygiene: rows32-r4.patch not next to this file, skipped")
        return
    txt = p.read_text()
    files = re.findall(r"^\+\+\+ b/(\S+)", txt, re.M)
    want = {"exllamav3_ext/bindings.cpp", "exllamav3_ext/libtorch/blocksparse_mlp.cpp", "exllamav3_ext/libtorch/mlp.cpp",
            "exllamav3_ext/libtorch/mlp.h", "exllamav3_ext/quant/exl3_moe_coop.cu", "exllamav3_ext/quant/exl3_moe_coop.cuh",
            "exllamav3_ext/quant/exl3_dense_v2.cu", "modules/block_sparse_mlp.py", "modules/mlp.py"}
    check(set(files) == want, f"patch touches {sorted(set(files) ^ want)} unexpectedly")
    added = [l[1:] for l in txt.splitlines() if l.startswith("+") and not l.startswith("+++")]
    for l in added:
        check(not re.search(r"__global__|__device__|<<<|__shared__|asm\s*\(", l), f"device code added: {l.strip()}")
    check(any("p_in.bsz <= 16 || p_in.bsz <= exl3_moe_coop_rows_cap()" in l for l in added),
          "the served <= 16-row condition must short-circuit before the env read")
    j = "\n".join(added)
    check('m.attr("moe_rows32_revision") = 2;' in j, "revision marker 2")
    check('"start_shared: bsz out of supported range"' in j and "start_shared: shared expert output scratch too small" in j,
          "start_shared bound (r4)")
    check("#define DENSE_V2_WS_SIDE_BYTES   (dense_v2_rows32() ? ((size_t) 4 << 20) : ((size_t) 2 << 20))" in j,
          "lcguard side region sized for 32-row slots (r4)")
    check("needs %zu B of side slots" in j, "side-refusal log line (r4)")
    removed = [l[1:] for l in txt.splitlines() if l.startswith("-") and not l.startswith("---")]
    check(not any("__global__" in l or "<<<" in l for l in removed), "no device code removed")
    print(f"[cpu] patch hygiene: {len(files)} files as expected, {len(added)} added lines, no device code; r4 additions "
          f"present", flush=True)


def part4():
    probe = ("import torch\n"
             "import exllamav3_ext as e\n"
             "import exllamav3.modules.mlp as m, exllamav3.modules.block_sparse_mlp as b\n"
             "print('RES', m.MAX_BSZN, b.MAX_BSZN, e.exl3_moe_coop_rows_cap(), e.moe_rows32_revision)\n")
    base_env = {k: v for k, v in os.environ.items() if k not in ("EXL3_MOE_COOP_ROWS32", "EXL3_SHARED_EXPERT_ROWS32")}
    # under the gate's real env (A = the stack-r3 union) the flags are absent; the B env adds all three: the module
    # constants are read at import, so each case is a fresh interpreter
    first = subprocess.run([sys.executable, "-c", probe], env=base_env, capture_output=True, text=True)
    if first.returncode != 0:
        if "--require-flags" in sys.argv:
            raise Fail(f"flags: import failed with the flags unset:\n{first.stderr[-2000:]}")
        tail = first.stderr.strip().splitlines()[-1:] or ["?"]
        print(f"[cpu] flags: the patched modules do not import here ({tail[0][:160]}), SKIPPED; the gate reruns this "
              f"part with --require-flags in a GPU container")
        return
    cases = [({}, "RES 16 16 16 2"),
             ({"EXL3_SHARED_EXPERT_ROWS32": "1"}, "RES 32 16 16 2"),
             ({"EXL3_SHARED_EXPERT_ROWS32": "1", "EXL3_MOE_COOP_ROWS32": "1"}, "RES 32 32 32 2"),
             ({"EXL3_SHARED_EXPERT_ROWS32": "0", "EXL3_MOE_COOP_ROWS32": "0"}, "RES 16 16 16 2"),
             ({"EXL3_MOE_COOP_ROWS32": "1"}, "requires EXL3_SHARED_EXPERT_ROWS32=1"),
             ({"EXL3_SHARED_EXPERT_ROWS32": "true"}, "must be 0 or 1"),
             ({"EXL3_SHARED_EXPERT_ROWS32": "1", "EXL3_MOE_COOP_ROWS32": "2"}, "must be 0 or 1")]
    for extra, want in cases:
        r = subprocess.run([sys.executable, "-c", probe], env={**base_env, **extra}, capture_output=True, text=True)
        out = r.stdout + r.stderr
        check(want in out, f"flags {extra}: expected {want!r}, got rc={r.returncode}: {out[-600:]}")
        if want.startswith("RES"):
            check(r.returncode == 0, f"flags {extra}: rc {r.returncode}")
        else:
            check(r.returncode != 0, f"flags {extra}: should have failed")
    print(f"[cpu] flags: {len(cases)} env combinations as expected (module constants, C++ cap, strict parsing, "
          f"4b-requires-4c)", flush=True)


# ------------------------------------------------------------------ part 5: source model of the patched tree
A_V3, A_MAP = "3", "2-4:2"                 # stack-r3 union (R713's MF3 line)
B_MAP = "2-4:2,17-32:2"                    # rows32 r4 B env (addendum: route 17-32 rows to mode 2)
DEVICE_SMS = 170                           # RTX 5090


def tree_dir():
    for i, a in enumerate(sys.argv):
        if a == "--tree" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    import sysconfig
    p = Path(sysconfig.get_paths()["purelib"]) / "exllamav3"
    return p if (p / "exllamav3_ext").is_dir() else None


def v3_mode(bsz, v3, mp):
    """exl3_moe_coop.cu moe_coop_v3_mode, replayed: EXL3_MOE_COOP_V3 then the MAP, first match wins."""
    mode = 0
    if v3 not in ("", "0"):
        check(len(v3) == 1 and v3 in "123", f"EXL3_MOE_COOP_V3={v3!r}")
        mode = int(v3)
    for ent in [e for e in (mp or "").split(",") if e]:
        m = re.fullmatch(r"(\d+)(?:-(\d+))?:([0-3])", ent)
        check(m is not None, f"MAP entry {ent!r} does not parse")
        lo = int(m.group(1)); hi = int(m.group(2) or lo)
        if lo <= bsz <= hi:
            return int(m.group(3))
    return mode


def launched(bsz, v3, mp, cap, r2_rows, r2_slots, topk=10):
    """exl3_moe_coop_launch: 'generic' (not the V2/V3 launch), else the V3 mode that runs (3 = r2 kernels)."""
    if not (1 <= bsz and (bsz <= 16 or bsz <= cap)):
        return "generic"
    m = v3_mode(bsz, v3, mp) if bsz > 1 else 0
    if m == 3 and (bsz * topk > r2_slots or bsz > r2_rows or bsz < 2):
        m = 2
    return m


def part5():
    t = tree_dir()
    if t is None:
        print("[cpu] source model: no tree (--tree DIR) and no installed exllamav3, SKIPPED")
        return
    q = t / "exllamav3_ext" / "quant"
    r2 = (q / "exl3_moe_coop_r2_kernel.cuh").read_text()
    cu = (q / "exl3_moe_coop.cu").read_text()
    dv2 = (q / "exl3_dense_v2.cu").read_text()
    km = (q / "exl3_kernel_map.cuh").read_text()
    r2_rows = int(re.search(r"constexpr int R2_MAX_ROWS = (\d+);", r2).group(1))
    r2_slots = int(re.search(r"constexpr int R2_MAX_SLOTS = (\d+);", r2).group(1))
    check("if (slots > R2_MAX_SLOTS || p.bsz > R2_MAX_ROWS || p.bsz < 2) return false;" in cu,
          "moe_coop_r2_launch's row / slot precondition")
    check(re.search(r"if \(v3 == 3\)\s*\{\s*if \(moe_coop_r2_launch\([^;]*\)\) return;\s*v3 = 2;\s*\}", cu) is not None,
          "the launcher falls back from mode 3 to mode 2")
    check("(p_in.bsz <= 16 || p_in.bsz <= exl3_moe_coop_rows_cap())" in cu, "rows32 launch condition")
    check(r2_rows <= 16, f"R2_MAX_ROWS {r2_rows} > 16: mode 3 could run above 16 rows; pin 17-32 to mode 2 AND add parity")
    n = 0
    for cap in (16, 32):
        for bsz in range(1, 33):
            a = launched(bsz, A_V3, A_MAP, cap, r2_rows, r2_slots)
            b = launched(bsz, A_V3, B_MAP, cap, r2_rows, r2_slots)
            check(a == b, f"cap {cap} rows {bsz}: A env launches {a}, B env {b}")
            if bsz > 16:
                check(a == ("generic" if cap == 16 else 2), f"cap {cap} rows {bsz}: {a}")
                check(v3_mode(bsz, A_V3, B_MAP) == 2, f"B MAP does not pin rows {bsz} to mode 2")
            n += 1
    served = {bsz: launched(bsz, A_V3, A_MAP, 16, r2_rows, r2_slots) for bsz in range(1, 17)}
    print(f"[cpu] source model, MoE mode: R2_MAX_ROWS {r2_rows}, R2_MAX_SLOTS {r2_slots}; {n} (cap, rows) cells: A env "
          f"(V3=3 MAP {A_MAP}) == B env (MAP {B_MAP}) everywhere; 17-32 rows -> mode 2 under both (flag on), generic "
          f"(flag off); served <= 16: " + ", ".join(f"{k}:{v}" for k, v in served.items()), flush=True)

    # lcguard side region at 17-32 rows
    tk = [int(x) for x in re.search(r"#define EXL3_GEMM_TILESIZE_K\s+([0-9, ]+)", km).group(1).split(",")]
    tn = [int(x) for x in re.search(r"#define EXL3_GEMM_TILESIZE_N\s+([0-9, ]+)", km).group(1).split(",")]
    m = re.search(r"#define DENSE_V2_WS_SIDE_BYTES\s+\(dense_v2_rows32\(\) \? \(\(size_t\) (\d+) << 20\) : \(\(size_t\) (\d+) << 20\)\)", dv2)
    check(m is not None, "DENSE_V2_WS_SIDE_BYTES is the r4 rows32-dependent size")
    side32, side16 = int(m.group(1)) << 20, int(m.group(2)) << 20
    K, N = 2560, 640                       # attn.index_qk_proj (the only dense call on the QSA side branch)
    tags = [g for g in (2, 3) if K % tk[g] == 0 and N % tn[g] == 0]   # rows32 twins exist for tags 2 / 3 only
    check(tags == [2], f"index_qk rows32 tags {tags} (expected tag 2 only: n 640 % 256 != 0)")
    tiles = (K // tk[2]) * (N // tn[2])
    grid = min(tiles, DEVICE_SMS)
    need32 = grid * 32 * tn[2] * 4
    need16 = grid * 16 * tn[2] * 4
    check(need32 <= side32, f"index_qk at 17-32 rows needs {need32} B > side region {side32} B")
    check(need16 <= side16, f"index_qk at 9-16 rows needs {need16} B > side region {side16} B")
    print(f"[cpu] source model, lcguard: index_qk tag 2 (tk {tk[2]}, tn {tn[2]}), {tiles} tiles, grid <= {grid}: "
          f"17-32 rows need <= {need32 / 2**20:.2f} MiB (side region {side32 >> 20} MiB with rows32; stack-r3's "
          f"{side16 >> 20} MiB holds {side16 // (32 * tn[2] * 4)} CTAs), 9-16 rows <= {need16 / 2**20:.2f} MiB", flush=True)


def part6():
    g = HERE / "greedy_conc.py"
    if not g.is_file():
        print("[cpu] greedy_conc verdict: greedy_conc.py not next to this file, skipped")
        return
    import json as _json
    import tempfile
    base = "x" * 300

    def recs(path, spec):
        with open(path, "w") as f:
            for tag, st_, divs, early, errs in spec:
                for pid in range(32):
                    txt = base
                    if pid < divs:
                        cut = 10 if pid < early else 200
                        txt = base[:cut] + "y" + base[cut + 1:]
                    err = "boom" if pid >= 32 - errs else None
                    f.write(_json.dumps(dict(tag=tag, set=st_, pid=pid, text=None if err else txt, error=err)) + "\n")

    def verdict(spec):
        with tempfile.TemporaryDirectory() as td:
            p = f"{td}/gc.jsonl"
            recs(p, spec)
            r = subprocess.run([sys.executable, str(g), "--compare", "--out", p, "--pair", "A_P=A2-c8-P:A1-c1-P",
                                "--pair", "A_Q=A1-c8-Q:A2-c1-Q", "--pair", "B_P=B1-c8-P:A1-c1-P",
                                "--pair", "B_Q=B2-c8-Q:A2-c1-Q", "--verdict", "A_P,A_Q", "B_P,B_Q"],
                               capture_output=True, text=True)
            return r.stdout.strip().splitlines()[-1]

    ref = [("A1-c1-P", "P", 0, 0, 0), ("A2-c1-Q", "Q", 0, 0, 0)]
    cases = [
        ("equal", [("A2-c8-P", "P", 2, 0, 0), ("A1-c8-Q", "Q", 3, 0, 0), ("B1-c8-P", "P", 3, 0, 0), ("B2-c8-Q", "Q", 3, 0, 0)], "PASS"),
        ("worse", [("A2-c8-P", "P", 2, 0, 0), ("A1-c8-Q", "Q", 3, 0, 0), ("B1-c8-P", "P", 4, 0, 0), ("B2-c8-Q", "Q", 3, 0, 0)], "FAIL"),
        ("early", [("A2-c8-P", "P", 2, 0, 0), ("A1-c8-Q", "Q", 3, 0, 0), ("B1-c8-P", "P", 1, 1, 0), ("B2-c8-Q", "Q", 1, 0, 0)], "PASS"),   # R717 review: early is reported, never decides
        ("error", [("A2-c8-P", "P", 2, 0, 0), ("A1-c8-Q", "Q", 3, 0, 0), ("B1-c8-P", "P", 0, 0, 1), ("B2-c8-Q", "Q", 0, 0, 0)], "FAIL"),
        ("sets", [("A2-c8-P", "Q", 2, 0, 0), ("A1-c8-Q", "Q", 3, 0, 0), ("B1-c8-P", "P", 0, 0, 0), ("B2-c8-Q", "Q", 0, 0, 0)], "FAIL"),
    ]
    for name, spec, want in cases:
        got = verdict(ref + spec)
        check(got.startswith(f"GREEDYC-VERDICT {want}"), f"greedy_conc verdict case {name}: {got}")
    print(f"[cpu] greedy_conc verdict: {len(cases)} synthetic cases as pre-registered (rate vs the worse A half, early, "
          f"errors, prompt sets)", flush=True)


def main():
    try:
        part1()
        part2()
        part3()
        part4()
        part5()
        part6()
    except Fail as e:
        print(f"[cpu] FAIL: {e}", flush=True)
        print("CPU TESTS FAIL")
        return 1
    print("CPU TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
