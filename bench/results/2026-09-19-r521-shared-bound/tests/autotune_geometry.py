#!/usr/bin/env python3
"""Which kernel geometry does the served shared expert run at each row count? (CPU-only, reads the box's tune cache.)

The shared expert's two quantized GEMMs (src/exllamav3/exllamav3_ext/libtorch/mlp.cpp:40-60 gate+up exl3_mgemm,
:88 down exl3_gemm) are launched with the tile shape and grid that CoopKernelAutotuner picked the first time each
(row bucket, k, n, K, dtype, device) key was seen (exl3_gemm.cu:238-279 / :575-626, coop_autotune.cu). The picks
persist in EXLLAMAV3_TUNE_CACHE/coop_autotune_v1.bin (the daily mounts /srv/qwen5090/.exl3cache there). This tool
recomputes the served keys (Python port of exl3_gemm.cu:40-108 + coop_autotune.cu salt, cross-checked against the
verbatim C++ in hash_ref.cpp by test_cpu.py), looks them up and prints the geometry, plus the k-segmentation of every
output column that geometry implies (slice_partition.py). Any "faster" bit-identical shared-expert kernel would have to
reproduce exactly these segmentations.

Usage (on the box, no GPU needed):
    python3 autotune_geometry.py --cache /srv/qwen5090/.exl3cache/coop_autotune_v1.bin \
        --hidden 2560 --interm 512 --k-gu 5 --k-d 5 --cb 2 --sms 170 --devices 0,1
The GPU probe (shared_expert_bound_gpu.py) prints the real hidden/interm/K/cb values to feed in here.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_partition import BLOCKDIM, TILESIZE_K, TILESIZE_N, segments  # noqa: E402

M64 = (1 << 64) - 1
FNV_OFF = 1469598103934665603
FNV_PRIME = 1099511628211
COOP_AUTOTUNE_VERSION = 4
CC_BLACKWELL = 5  # exl3_devctx.cuh (prop.major >= 10 -> CC_BLACKWELL; sm_120 is major 12)

MAGIC = b"EX3ATUNE"
HEADER = struct.Struct("<8sII")          # DiskCacheHeader: magic, format, record_size
RECORD = struct.Struct("<QiiiiII")       # DiskCacheRecordV1: hash, tag, block_dim, num_sms, concurrency, reserved x2


def roundup_pow2(x: int) -> int:
    if x == 0:
        return 1
    x -= 1
    for s in (1, 2, 4, 8, 16, 32):
        x |= x >> s
    return (x + 1) & M64


def _mix(h: int, v: int) -> int:
    h ^= v & M64
    return (h * FNV_PRIME) & M64


def gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb) -> int:
    h = FNV_OFF
    for v in (min(roundup_pow2(size_m), 16), size_k, size_n, K, 1 if c_fp32 else 0, device, cc, max_num_sms, cb):
        h = _mix(h, v)
    return h


def mgemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb, bszm_in, bszm_out) -> int:
    h = gemm_autotune_hash(size_m, size_k, size_n, K, c_fp32, device, cc, max_num_sms, cb)
    h = _mix(h, min(bszm_in, 24))
    return _mix(h, min(bszm_out, 24))


def salt(h: int) -> int:
    return _mix(h, COOP_AUTOTUNE_VERSION)


def down_key(rows, interm, hidden, K, device, sms, cb, cc=CC_BLACKWELL) -> int:
    # exl3_gemm.cu:241 keys the regular kernel on MAX(size_m, 2); C (the shared output) is fp32
    return salt(gemm_autotune_hash(max(rows, 2), interm, hidden, K, True, device, cc, sms, cb))


def gate_up_key(rows, hidden, interm, K, device, sms, cb, cc=CC_BLACKWELL) -> int:
    # exl3_gemm.cu:578-581: A = x (1, rows, hidden) -> bszm_in 1; C = gu (2, rows, interm) half -> bszm_out 2
    return salt(mgemm_autotune_hash(rows, hidden, interm, K, False, device, cc, sms, cb, 1, 2))


def read_cache(path: str) -> dict[int, tuple[int, int, int, int]]:
    """Same parse as coop_autotune.cu:load_disk_cache_locked (including the embedded-header skip)."""
    out: dict[int, tuple[int, int, int, int]] = {}
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < HEADER.size:
        return out
    magic, fmt, rsize = HEADER.unpack_from(data, 0)
    if magic != MAGIC or fmt != 1 or rsize < RECORD.size:
        return out
    pos = HEADER.size
    while pos + rsize <= len(data):
        chunk = data[pos:pos + rsize]
        if rsize >= HEADER.size and chunk[:8] == MAGIC:
            pos = pos + HEADER.size   # rewind to just after the embedded header
            continue
        h, tag, block_dim, num_sms, conc, _, _ = RECORD.unpack_from(chunk, 0)
        out[h] = (tag, block_dim, num_sms, conc)
        pos += rsize
    return out


def down_path(rows: int, K: int, cb: int, int8_max_k: int = 6, int8_mode: int = 2) -> str:
    """Which kernel family exl3_gemm_gr takes for the shared down projection (exl3_gemm.cu:182-236)."""
    if cb == 2 and int8_mode != 0 and 1 <= K <= int8_max_k and rows <= (1 if int8_mode == 1 else 2):
        return "int8-sq (exl3_gemv_int8.cu:254; plain int8 activations in mode 2)"
    if 2 <= K <= 4 and (K == 4 or cb != 0) and rows <= 8:
        return "qtip-gemv candidate (exl3_gemv.cu heuristic) else autotuned"
    return "autotuned exl3_gemm_kernel"


def describe(rec, size_k, size_n) -> str:
    if rec is None:
        return "NOT IN CACHE (tuned at first use in the probe/serve process)"
    tag, block_dim, num_sms, conc = rec
    tk, tn = TILESIZE_K.get(tag), TILESIZE_N.get(tag)
    if tk is None:
        return f"tag {tag} (unknown shape) block {block_dim} grid {num_sms}x{conc}"
    tiles_k, tiles_n = size_k // tk, size_n // tn
    segs = segments(tiles_k, tiles_n, num_sms)
    nseg = sorted({len(v) for v in segs.values()})
    return (f"shape {tag} (k-tile {tk}, n-tile {tn}, {block_dim} thr) grid x={num_sms} z={conc}; "
            f"tiles {tiles_k}x{tiles_n}; k-segments per column {nseg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--interm", type=int, default=512)
    ap.add_argument("--k-gu", type=int, default=5)
    ap.add_argument("--k-d", type=int, default=5)
    ap.add_argument("--cb", type=int, default=2, help="0 = 3inst, 1 = mcg, 2 = mul1")
    ap.add_argument("--sms", type=int, default=170)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--rows", default="1,2,3,4,8,16")
    a = ap.parse_args()
    cache = read_cache(a.cache)
    print(f"{a.cache}: {len(cache)} records")
    for dev in [int(d) for d in a.devices.split(",")]:
        for rows in [int(r) for r in a.rows.split(",")]:
            gk = gate_up_key(rows, a.hidden, a.interm, a.k_gu, dev, a.sms, a.cb)
            dk = down_key(rows, a.interm, a.hidden, a.k_d, dev, a.sms, a.cb)
            print(f"cuda:{dev} rows {rows:2d}  gate+up mgemm key {gk:016x}: {describe(cache.get(gk), a.hidden, a.interm)}")
            path = down_path(rows, a.k_d, a.cb)
            dd = describe(cache.get(dk), a.interm, a.hidden) if path.startswith("autotuned") else path
            print(f"cuda:{dev} rows {rows:2d}  down gemm     key {dk:016x}: {dd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
