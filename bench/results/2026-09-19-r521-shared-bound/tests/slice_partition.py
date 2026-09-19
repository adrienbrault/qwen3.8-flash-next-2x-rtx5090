"""CPU reference of the served EXL3 GEMM work partition and cross-block reduction order.

Mirrors src/exllamav3/exllamav3_ext/quant/exl3_gemm_inner.cuh (the inner function shared by exl3_gemm_kernel and
exl3_mgemm_kernel, i.e. by both GEMMs of the shared expert):

- :97-106   block b of gridDim.x = num_slices owns the flat tile range
            [tiles_k*tiles_n*b // num_slices, tiles_k*tiles_n*(b+1) // num_slices), tile s -> (k = s % tiles_k, n = s // tiles_k)
- :934-944  a block reduces (reduce()) when it reaches the last k-tile of a column or the end of its range
- :829-871  per column, partial segments are folded in lock order lock_i = tiles_k - k_end - 1, i.e. the segment holding
            the LAST k-tile goes first ("first" writes), each following (lower-k) segment reads the running global
            partial and adds its register sum (read_sum_gl: frag_c += C), the one reaching lock_i + lock_d == tiles_k
            is "last" and writes the output
- the per-segment register sum accumulates its k-tiles in ascending k (the main loop issues the MMAs in order)

So the fp32 value of every output element is a fixed function of (TILESIZE_K, TILESIZE_N) and gridDim.x. The shape and
gridDim.x are what CoopKernelAutotuner picks per (row bucket, k, n, K, dtype, device) (coop_autotune.cu). A kernel that
wants the same bits must reproduce this segmentation exactly; any other grid or tile shape re-associates the fp32 sums.
"""
from __future__ import annotations

import struct


def f32(x: float) -> float:
    """Round a Python float to IEEE binary32 (round-to-nearest-even, like the GPU's fp32 add)."""
    return struct.unpack("f", struct.pack("f", x))[0]


def segments(tiles_k: int, tiles_n: int, num_slices: int) -> dict[int, list[tuple[int, int, int]]]:
    """Per output column tile n: list of (k_begin, k_end_inclusive, block) in the order they are folded."""
    per_col: dict[int, list[tuple[int, int, int]]] = {n: [] for n in range(tiles_n)}
    total = tiles_k * tiles_n
    for b in range(num_slices):
        beg = total * b // num_slices
        end = total * (b + 1) // num_slices
        if end - beg < 1:
            continue
        k = beg % tiles_k
        k0 = k
        n = beg // tiles_k
        iters = end - beg
        while True:
            if k == tiles_k - 1 or iters == 1:
                per_col[n].append((k0, k, b))
                k0 = k + 1
            # advance2()
            k += 1
            iters -= 1
            if k >= tiles_k:
                k = 0
                k0 = 0
                n += 1
            if not iters:
                break
    # fold order: lock_i = tiles_k - k_end - 1 ascending -> k_end descending
    for n in per_col:
        per_col[n].sort(key=lambda seg: tiles_k - seg[1] - 1)
    return per_col


def fold_column(partials: list[float], segs: list[tuple[int, int, int]]) -> float:
    """fp32 result of one output element given its per-k-tile contributions and the column's segment order."""
    acc = None
    for k0, k1, _ in segs:
        s = 0.0
        for k in range(k0, k1 + 1):
            s = f32(s + partials[k])
        acc = s if acc is None else f32(s + acc)
    return acc


def tile_counts(size_k: int, size_n: int, tilesize_k: int, tilesize_n: int) -> tuple[int, int]:
    return size_k // tilesize_k, size_n // tilesize_n


# exl3_kernel_map.cuh: EXL3_GEMM_TILESIZE_K / _N / _BLOCKDIM, indexed by shape (tag) 1..4
TILESIZE_K = {1: 16, 2: 32, 3: 32, 4: 16}
TILESIZE_N = {1: 128, 2: 128, 3: 256, 4: 512}
BLOCKDIM = {1: 256, 2: 512, 3: 512, 4: 256}
