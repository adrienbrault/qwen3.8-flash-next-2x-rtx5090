#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "hadamard_inner.cuh"
#include "exl3_moe_prefill_e3.cuh"

// MiaAI-Lab E3 transferred to the served EXL3 layout used by Qwen4-Exp:
//   * independently dispatched 2-, 3-, or 4-bit trellises, mul1 codebook
//   * independent gate/up SUH vectors (the original K4/MCG kernel shared one)
//   * served half-precision rounding boundaries around Hadamards/SiLU
//   * one device metadata launch before the three grouped compute phases
// The output scatter remains atomic, as in the source E3 design.

namespace {

constexpr int E3_THREADS = 256;
constexpr int E3_WARPS = E3_THREADS / 32;
constexpr int E3_META_THREADS = 512;
constexpr int E3_EXPERTS = 512;
constexpr int E3_TILE_N = 128;
constexpr int E3_TILE_K = 32;
constexpr int E3_STAGES = 4;
constexpr int E3_CB = 2;  // mul1
constexpr float E3_HAD_SCALE = 0.088388347648f;
constexpr int E3_MB = 4;  // 64 rows
constexpr int E3_TILE_M = E3_MB * 16;

template <int MB, int AS, int BITS0, int BITS1>
constexpr int e3_smem_bytes()
{
    constexpr int a_stage = AS * MB * 16 * E3_TILE_K * 2;
    // Each of the two B streams carries both 16-column halves for every warp.
    // The streams may have different trellis widths.
    constexpr int b_stage = 2 * E3_WARPS * 16 * (BITS0 + BITS1) * 2;
    constexpr int pipe = E3_STAGES * (a_stage + b_stage);
    constexpr int epi = 16 * 2 * E3_TILE_N * 4;
    return pipe > epi ? pipe : epi;
}

__device__ __forceinline__ int e3_swz(int row, int chunk)
{
    return chunk ^ ((row >> 1) & 3);
}

// Same half boundaries and Hadamard operation order as had_hf_r_128_inner:
// half input -> fp32 butterfly -> half normalized result.
__device__ __forceinline__ half4 e3_had_half(half4 v, int lane)
{
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, lane);
    half4 o;
    o.x = __floats2half2_rn(h0 * E3_HAD_SCALE, h1 * E3_HAD_SCALE);
    o.y = __floats2half2_rn(h2 * E3_HAD_SCALE, h3 * E3_HAD_SCALE);
    return o;
}

__device__ __forceinline__ float4 e3_had_float_raw(half4 v, int lane)
{
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float4 o = make_float4(s0 + s1, d0 + d1, s0 - s1, d0 - d1);
    shuffle_had_f4x32(o.x, o.y, o.z, o.w, lane);
    return o;
}

__device__ __forceinline__ half2 e3_silu(half2 x)
{
    const half2 one = __float2half2_rn(1.0f);
    return __hmul2(x, h2rcp(__hadd2(one, h2exp(__hneg2(x)))));
}

__device__ __forceinline__ half4 e3_float4_to_half4(const float4& v)
{
    return half4(__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w));
}

// Build compact fat rows and 64-row expert segments directly from the GPU
// histogram. Thread 0 performs two 512-element scans; all lanes copy rows and
// write segment records in parallel. This is the fourth launch relative to
// Mia's three phases because served routing does not expose E3 segment tables.
__global__ __launch_bounds__(E3_META_THREADS)
void e3_metadata_kernel(
    const int64_t* __restrict__ expert_count,
    const int64_t* __restrict__ token_sorted,
    const half* __restrict__ weight_sorted,
    int64_t* __restrict__ row_token,
    half* __restrict__ row_weight,
    int* __restrict__ row_expert,
    int* __restrict__ seg_expert,
    int* __restrict__ seg_row0,
    int* __restrict__ seg_rows,
    int* __restrict__ num_rows,
    int* __restrict__ num_segs,
    int thin_rows)
{
    __shared__ int starts[E3_EXPERTS];
    __shared__ int fat_base[E3_EXPERTS];
    __shared__ int seg_base[E3_EXPERTS];

    if (threadIdx.x == 0)
    {
        int src = 0;
        int rows = 0;
        int segs = 0;
        #pragma unroll 1
        for (int e = 0; e < E3_EXPERTS; ++e)
        {
            const int c = static_cast<int>(expert_count[e]);
            starts[e] = src;
            src += c;
            fat_base[e] = rows;
            seg_base[e] = segs;
            if (c > thin_rows)
            {
                rows += c;
                segs += (c + E3_TILE_M - 1) / E3_TILE_M;
            }
        }
        *num_rows = rows;
        *num_segs = segs;
    }
    __syncthreads();

    const int e = threadIdx.x;
    const int c = static_cast<int>(expert_count[e]);
    if (c <= thin_rows) return;
    const int src0 = starts[e];
    const int dst0 = fat_base[e];
    for (int r = 0; r < c; ++r)
    {
        row_token[dst0 + r] = token_sorted[src0 + r];
        row_weight[dst0 + r] = weight_sorted[src0 + r];
        row_expert[dst0 + r] = e;
    }
    const int ns = (c + E3_TILE_M - 1) / E3_TILE_M;
    for (int s = 0; s < ns; ++s)
    {
        const int i = seg_base[e] + s;
        const int r0 = s * E3_TILE_M;
        seg_expert[i] = e;
        seg_row0[i] = dst0 + r0;
        seg_rows[i] = (c - r0 < E3_TILE_M) ? (c - r0) : E3_TILE_M;
    }
}

// Rotate/gather gate and up independently. Unlike Mia's w13 representation,
// served gate/up EXL3 tensors are not required to share SUH.
__global__ __launch_bounds__(E3_THREADS)
void e3_gather_kernel(
    const half* __restrict__ x,
    const int64_t* __restrict__ row_token,
    const int* __restrict__ row_expert,
    const half* const* __restrict__ gate_suh,
    const half* const* __restrict__ up_suh,
    half* __restrict__ h13_gate,
    half* __restrict__ h13_up,
    const int* __restrict__ num_rows,
    int hidden)
{
    const int live_rows = *num_rows;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.y;
    for (int row = blockIdx.x * E3_WARPS + warp;
         row < live_rows;
         row += gridDim.x * E3_WARPS)
    {
        const int64_t token = row_token[row];
        const int e = row_expert[row];
        const half* src = x + token * static_cast<int64_t>(hidden) + blk * 128 + lane * 4;
        const half4 xv = *reinterpret_cast<const half4*>(src);

        const half4 sg = *reinterpret_cast<const half4*>(gate_suh[e] + blk * 128 + lane * 4);
        half4 vg;
        vg.x = __hmul2(xv.x, sg.x);
        vg.y = __hmul2(xv.y, sg.y);
        vg = e3_had_half(vg, lane);
        *reinterpret_cast<half4*>(h13_gate + row * static_cast<int64_t>(hidden) + blk * 128 + lane * 4) = vg;

        const half4 su = *reinterpret_cast<const half4*>(up_suh[e] + blk * 128 + lane * 4);
        half4 vu;
        vu.x = __hmul2(xv.x, su.x);
        vu.y = __hmul2(xv.y, su.y);
        vu = e3_had_half(vu, lane);
        *reinterpret_cast<half4*>(h13_up + row * static_cast<int64_t>(hidden) + blk * 128 + lane * 4) = vu;
    }
}

// AS is the number of A streams: gate/up uses two; down shares one A stream
// between its adjacent 128-column output halves. BITS0/BITS1 are independent,
// which is required when gate/up/down do not all have the same trellis width.
template <int MB, int AS, int NB_STRIDE, int BITS0, int BITS1>
__device__ __forceinline__ void e3_mainloop(
    const half* const (&a)[AS],
    int size_k,
    int row0,
    int rows,
    const uint16_t* const (&packed)[2],
    int tiles_n,
    int n_block0,
    half* sh_a,
    uint16_t* sh_b,
    FragC (&acc)[2][MB][2])
{
    constexpr int TILE_M = MB * 16;
    constexpr int A_ONE = TILE_M * E3_TILE_K;
    constexpr int A_STAGE = AS * A_ONE;
    constexpr int A_CHUNKS_ONE = TILE_M * 4;
    constexpr int A_CHUNKS = AS * A_CHUNKS_ONE;
    constexpr int A_ITERS = (A_CHUNKS + E3_THREADS - 1) / E3_THREADS;
    constexpr int PW0 = BITS0 * 16;
    constexpr int PW1 = BITS1 * 16;
    constexpr int PC0 = PW0 / 8;
    constexpr int PC1 = PW1 / 8;
    constexpr int B_STREAM0 = 2 * E3_WARPS * PW0;
    constexpr int B_STREAM1 = 2 * E3_WARPS * PW1;
    constexpr int B_STAGE = B_STREAM0 + B_STREAM1;
    constexpr int B_CHUNKS0 = 2 * E3_WARPS * PC0;
    constexpr int B_CHUNKS1 = 2 * E3_WARPS * PC1;
    constexpr int B_CHUNKS = B_CHUNKS0 + B_CHUNKS1;
    constexpr int B_ITERS = (B_CHUNKS + E3_THREADS - 1) / E3_THREADS;

    const int t = threadIdx.x;
    const int warp = t >> 5;
    const int lane = t & 31;
    const int k_tiles = size_k / E3_TILE_K;

    #pragma unroll
    for (int s = 0; s < 2; ++s)
        #pragma unroll
        for (int mb = 0; mb < MB; ++mb)
        {
            acc[s][mb][0] = {};
            acc[s][mb][1] = {};
        }

    auto load_stage = [&](int stage, int kt)
    {
        half* sa = sh_a + stage * A_STAGE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
        {
            const int c = i * E3_THREADS + t;
            if (c < A_CHUNKS)
            {
                const int as = c / A_CHUNKS_ONE;
                const int ar = c - as * A_CHUNKS_ONE;
                const int row = ar >> 2;
                const int chunk = ar & 3;
                const int src_row = row < rows ? row : rows - 1;
                const half* src = a[as] + static_cast<int64_t>(row0 + src_row) * size_k
                                + kt * E3_TILE_K + chunk * 8;
                half* dst = sa + as * A_ONE + row * E3_TILE_K + e3_swz(row, chunk) * 8;
                cp_async(dst, src);
            }
        }
        uint16_t* sb = sh_b + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < B_ITERS; ++i)
        {
            const int c = i * E3_THREADS + t;
            if (c < B_CHUNKS)
            {
                const int s = c >= B_CHUNKS0;
                const int cs = s ? c - B_CHUNKS0 : c;
                const int pc = s ? PC1 : PC0;
                const int pw = s ? PW1 : PW0;
                const int j = cs / (E3_WARPS * pc);
                const int q0 = cs - j * E3_WARPS * pc;
                const int nb = q0 / pc;
                const int q = q0 - nb * pc;
                const uint16_t* src = packed[s]
                    + (static_cast<int64_t>(kt * 2 + j) * tiles_n
                       + n_block0 + s * NB_STRIDE + nb) * pw + q * 8;
                const int stream_base = s ? B_STREAM0 : 0;
                uint16_t* dst = sb + stream_base + j * E3_WARPS * pw
                                + nb * pw + q * 8;
                cp_async(dst, src);
            }
        }
    };

    #pragma unroll
    for (int s = 0; s < E3_STAGES - 1; ++s)
    {
        if (s < k_tiles) load_stage(s, s);
        cp_async_fence();
    }

    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int a_chunk_hi = lane >> 4;
    for (int kt = 0; kt < k_tiles; ++kt)
    {
        cp_async_wait<E3_STAGES - 2>();
        __syncthreads();
        const int nk = kt + E3_STAGES - 1;
        if (nk < k_tiles) load_stage(nk % E3_STAGES, nk);
        cp_async_fence();

        const int stage = kt % E3_STAGES;
        const half* sa = sh_a + stage * A_STAGE;
        const uint16_t* sb = sh_b + stage * B_STAGE;
        #pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            FragB fb[2][2];
            const uint32_t* wb0 = reinterpret_cast<const uint32_t*>(
                sb + j * E3_WARPS * PW0 + warp * PW0);
            const uint32_t* wb1 = reinterpret_cast<const uint32_t*>(
                sb + B_STREAM0 + j * E3_WARPS * PW1 + warp * PW1);
            dq_dispatch<BITS0, E3_CB>(wb0, lane << 3, fb[0][0], fb[0][1]);
            dq_dispatch<BITS1, E3_CB>(wb1, lane << 3, fb[1][0], fb[1][1]);
            #pragma unroll
            for (int mb = 0; mb < MB; ++mb)
            {
                #pragma unroll
                for (int s = 0; s < 2; ++s)
                {
                    FragA fa;
                    const int as = AS == 1 ? 0 : s;
                    const int row = mb * 16 + a_row;
                    const int chunk = j * 2 + a_chunk_hi;
                    ldsm4(fa, sa + as * A_ONE + row * E3_TILE_K + e3_swz(row, chunk) * 8);
                    ptx_mma_m16n8k16(fa, fb[s][0], acc[s][mb][0]);
                    ptx_mma_m16n8k16(fa, fb[s][1], acc[s][mb][1]);
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();
}

template <int NS>
__device__ __forceinline__ void e3_stage_acc(
    float* sh_c, FragC (&acc0)[2], FragC (&acc1)[2], int warp, int lane)
{
    constexpr int W = NS * E3_TILE_N;
    const int r0 = lane >> 2;
    const int col = (lane & 3) * 2 + warp * 16;
    float* d0 = sh_c + r0 * W + col;
    float* d1 = sh_c + (r0 + 8) * W + col;
    d0[0] = acc0[0][0]; d0[1] = acc0[0][1]; d0[8] = acc0[1][0]; d0[9] = acc0[1][1];
    d1[0] = acc0[0][2]; d1[1] = acc0[0][3]; d1[8] = acc0[1][2]; d1[9] = acc0[1][3];
    if constexpr (NS == 2)
    {
        d0 = sh_c + r0 * W + E3_TILE_N + col;
        d1 = sh_c + (r0 + 8) * W + E3_TILE_N + col;
        d0[0] = acc1[0][0]; d0[1] = acc1[0][1]; d0[8] = acc1[1][0]; d0[9] = acc1[1][1];
        d1[0] = acc1[0][2]; d1[1] = acc1[0][3]; d1[8] = acc1[1][2]; d1[9] = acc1[1][3];
    }
}

template <int GATE_BITS, int UP_BITS>
__global__ __launch_bounds__(E3_THREADS, 2)
void e3_gateup_kernel(
    const half* __restrict__ h13_gate,
    const half* __restrict__ h13_up,
    const uint16_t* const* __restrict__ gate_trellis,
    const uint16_t* const* __restrict__ up_trellis,
    const half* const* __restrict__ gate_svh,
    const half* const* __restrict__ up_svh,
    const half* const* __restrict__ down_suh,
    half* __restrict__ h2,
    const int* __restrict__ seg_expert,
    const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs,
    int hidden,
    int intermediate,
    float act_limit)
{
    constexpr int AS = 2;
    extern __shared__ __align__(16) unsigned char smem[];
    half* sh_a = reinterpret_cast<half*>(smem);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(
        sh_a + E3_STAGES * AS * E3_TILE_M * E3_TILE_K);
    float* sh_c = reinterpret_cast<float*>(smem);

    const int live_segs = *num_segs;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tiles_n = intermediate / 16;
    const int n_base = blockIdx.x * E3_TILE_N;
    for (int seg = blockIdx.y; seg < live_segs; seg += gridDim.y)
    {
        const int e = seg_expert[seg];
        const int row0 = seg_row0[seg];
        const int rows = seg_rows[seg];
        const half* const a[AS] = {h13_gate, h13_up};
        const uint16_t* const packed[2] = {gate_trellis[e], up_trellis[e]};
        FragC acc[2][E3_MB][2];
        e3_mainloop<E3_MB, AS, 0, GATE_BITS, UP_BITS>(
            a, hidden, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);

        #pragma unroll
        for (int mb = 0; mb < E3_MB; ++mb)
        {
            const int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            e3_stage_acc<2>(sh_c, acc[0][mb], acc[1][mb], warp, lane);
            __syncthreads();
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                const int r = warp + rr * E3_WARPS;
                if (r < rows_mb)
                {
                    const float* src = sh_c + r * (2 * E3_TILE_N) + lane * 4;
                    half4 g = e3_float4_to_half4(*reinterpret_cast<const float4*>(src));
                    half4 u = e3_float4_to_half4(*reinterpret_cast<const float4*>(src + E3_TILE_N));
                    g = e3_had_half(g, lane);
                    u = e3_had_half(u, lane);
                    const half4 sg = *reinterpret_cast<const half4*>(gate_svh[e] + n_base + lane * 4);
                    const half4 su = *reinterpret_cast<const half4*>(up_svh[e] + n_base + lane * 4);
                    g.x = __hmul2(g.x, sg.x); g.y = __hmul2(g.y, sg.y);
                    u.x = __hmul2(u.x, su.x); u.y = __hmul2(u.y, su.y);
                    g.x = e3_silu(g.x); g.y = e3_silu(g.y);
                    if (act_limit != 0.0f)
                    {
                        const half2 lo = __float2half2_rn(-act_limit);
                        const half2 hi = __float2half2_rn(act_limit);
                        u.x = __hmin2(__hmax2(u.x, lo), hi);
                        u.y = __hmin2(__hmax2(u.y, lo), hi);
                        g.x = __hmin2(g.x, hi);
                        g.y = __hmin2(g.y, hi);
                    }
                    g.x = __hmul2(g.x, u.x); g.y = __hmul2(g.y, u.y);
                    const half4 sd = *reinterpret_cast<const half4*>(down_suh[e] + n_base + lane * 4);
                    g.x = __hmul2(g.x, sd.x); g.y = __hmul2(g.y, sd.y);
                    g = e3_had_half(g, lane);
                    half* dst = h2 + static_cast<int64_t>(row0 + mb * 16 + r) * intermediate
                                + n_base + lane * 4;
                    *reinterpret_cast<half4*>(dst) = g;
                }
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void e3_atomic_float4(float* dst, const float4& v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    atomicAdd(reinterpret_cast<float4*>(dst), v);
#else
    atomicAdd(dst + 0, v.x);
    atomicAdd(dst + 1, v.y);
    atomicAdd(dst + 2, v.z);
    atomicAdd(dst + 3, v.w);
#endif
}

template <int DOWN_BITS>
__global__ __launch_bounds__(E3_THREADS, 2)
void e3_down_kernel(
    const half* __restrict__ h2,
    const uint16_t* const* __restrict__ down_trellis,
    const half* const* __restrict__ down_svh,
    float* __restrict__ out,
    const int64_t* __restrict__ row_token,
    const half* __restrict__ row_weight,
    const int* __restrict__ seg_expert,
    const int* __restrict__ seg_row0,
    const int* __restrict__ seg_rows,
    const int* __restrict__ num_segs,
    int intermediate,
    int hidden)
{
    constexpr int AS = 1;
    constexpr int TILE_N2 = 2 * E3_TILE_N;
    extern __shared__ __align__(16) unsigned char smem[];
    half* sh_a = reinterpret_cast<half*>(smem);
    uint16_t* sh_b = reinterpret_cast<uint16_t*>(
        sh_a + E3_STAGES * AS * E3_TILE_M * E3_TILE_K);
    float* sh_c = reinterpret_cast<float*>(smem);

    const int live_segs = *num_segs;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int tiles_n = hidden / 16;
    const int n_base = blockIdx.x * TILE_N2;
    for (int seg = blockIdx.y; seg < live_segs; seg += gridDim.y)
    {
        const int e = seg_expert[seg];
        const int row0 = seg_row0[seg];
        const int rows = seg_rows[seg];
        const half* const a[AS] = {h2};
        const uint16_t* const packed[2] = {down_trellis[e], down_trellis[e]};
        FragC acc[2][E3_MB][2];
        e3_mainloop<E3_MB, AS, E3_TILE_N / 16, DOWN_BITS, DOWN_BITS>(
            a, intermediate, row0, rows, packed, tiles_n, n_base / 16, sh_a, sh_b, acc);

        #pragma unroll
        for (int mb = 0; mb < E3_MB; ++mb)
        {
            const int rows_mb = rows - mb * 16;
            if (rows_mb <= 0) break;
            e3_stage_acc<2>(sh_c, acc[0][mb], acc[1][mb], warp, lane);
            __syncthreads();
            #pragma unroll
            for (int rr = 0; rr < 2; ++rr)
            {
                const int r = warp + rr * E3_WARPS;
                if (r < rows_mb)
                {
                    const int fat_row = row0 + mb * 16 + r;
                    const float route = __half2float(row_weight[fat_row]);
                    const float* src = sh_c + r * TILE_N2;
                    float* dst = out + row_token[fat_row] * static_cast<int64_t>(hidden)
                                 + n_base + lane * 4;
                    #pragma unroll
                    for (int s = 0; s < 2; ++s)
                    {
                        half4 hv = e3_float4_to_half4(
                            *reinterpret_cast<const float4*>(src + s * E3_TILE_N + lane * 4));
                        float4 v = e3_had_float_raw(hv, lane);
                        const half4 scale = *reinterpret_cast<const half4*>(
                            down_svh[e] + n_base + s * E3_TILE_N + lane * 4);
                        const float r_scale = E3_HAD_SCALE * route;
                        v.x *= r_scale; v.y *= r_scale; v.z *= r_scale; v.w *= r_scale;
                        v.x *= __low2float(scale.x); v.y *= __high2float(scale.x);
                        v.z *= __low2float(scale.y); v.w *= __high2float(scale.y);
                        e3_atomic_float4(dst + s * E3_TILE_N, v);
                    }
                }
            }
            __syncthreads();
        }
    }
}

struct E3GateUpLaunch
{
    const half* h13_gate;
    const half* h13_up;
    const uint16_t* const* gate_trellis;
    const uint16_t* const* up_trellis;
    const half* const* gate_svh;
    const half* const* up_svh;
    const half* const* down_suh;
    half* h2;
    const int* seg_expert;
    const int* seg_row0;
    const int* seg_rows;
    const int* num_segs;
    int hidden;
    int intermediate;
    float act_limit;
    int grid_y;
    cudaStream_t stream;
};

template <int GATE_BITS, int UP_BITS>
void e3_launch_gateup(const E3GateUpLaunch& a)
{
    constexpr int smem = e3_smem_bytes<E3_MB, 2, GATE_BITS, UP_BITS>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        e3_gateup_kernel<GATE_BITS, UP_BITS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    e3_gateup_kernel<GATE_BITS, UP_BITS>
        <<<dim3(a.intermediate / E3_TILE_N, a.grid_y), E3_THREADS, smem, a.stream>>>(
            a.h13_gate, a.h13_up, a.gate_trellis, a.up_trellis,
            a.gate_svh, a.up_svh, a.down_suh, a.h2,
            a.seg_expert, a.seg_row0, a.seg_rows, a.num_segs,
            a.hidden, a.intermediate, a.act_limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int GATE_BITS>
void e3_dispatch_up(int up_bits, const E3GateUpLaunch& a)
{
    switch (up_bits)
    {
        case 2: e3_launch_gateup<GATE_BITS, 2>(a); break;
        case 3: e3_launch_gateup<GATE_BITS, 3>(a); break;
        case 4: e3_launch_gateup<GATE_BITS, 4>(a); break;
        default: TORCH_CHECK(false, "E3 up K must be 2, 3, or 4");
    }
}

void e3_dispatch_gateup(int gate_bits, int up_bits, const E3GateUpLaunch& a)
{
    switch (gate_bits)
    {
        case 2: e3_dispatch_up<2>(up_bits, a); break;
        case 3: e3_dispatch_up<3>(up_bits, a); break;
        case 4: e3_dispatch_up<4>(up_bits, a); break;
        default: TORCH_CHECK(false, "E3 gate K must be 2, 3, or 4");
    }
}

struct E3DownLaunch
{
    const half* h2;
    const uint16_t* const* down_trellis;
    const half* const* down_svh;
    float* out;
    const int64_t* row_token;
    const half* row_weight;
    const int* seg_expert;
    const int* seg_row0;
    const int* seg_rows;
    const int* num_segs;
    int intermediate;
    int hidden;
    int grid_y;
    cudaStream_t stream;
};

template <int DOWN_BITS>
void e3_launch_down(const E3DownLaunch& a)
{
    constexpr int smem = e3_smem_bytes<E3_MB, 1, DOWN_BITS, DOWN_BITS>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        e3_down_kernel<DOWN_BITS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    e3_down_kernel<DOWN_BITS>
        <<<dim3(a.hidden / (2 * E3_TILE_N), a.grid_y), E3_THREADS, smem, a.stream>>>(
            a.h2, a.down_trellis, a.down_svh, a.out, a.row_token, a.row_weight,
            a.seg_expert, a.seg_row0, a.seg_rows, a.num_segs, a.intermediate, a.hidden);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void e3_dispatch_down(int down_bits, const E3DownLaunch& a)
{
    switch (down_bits)
    {
        case 2: e3_launch_down<2>(a); break;
        case 3: e3_launch_down<3>(a); break;
        case 4: e3_launch_down<4>(a); break;
        default: TORCH_CHECK(false, "E3 down K must be 2, 3, or 4");
    }
}

int e3_grid_y(int64_t capacity)
{
    if (capacity < 1) return 1;
    return static_cast<int>(capacity > 512 ? 512 : capacity);
}

}  // namespace

void exl3_moe_prefill_e3_cuda(
    const at::Tensor& x,
    const at::Tensor& out,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,
    const at::Tensor& gate_trellis,
    const at::Tensor& gate_suh,
    const at::Tensor& gate_svh,
    const at::Tensor& up_trellis,
    const at::Tensor& up_suh,
    const at::Tensor& up_svh,
    const at::Tensor& down_trellis,
    const at::Tensor& down_suh,
    const at::Tensor& down_svh,
    const at::Tensor& h13_gate,
    const at::Tensor& h13_up,
    const at::Tensor& h2,
    const at::Tensor& row_token,
    const at::Tensor& row_weight,
    const at::Tensor& row_expert,
    const at::Tensor& seg_expert,
    const at::Tensor& seg_row0,
    const at::Tensor& seg_rows,
    const at::Tensor& num_rows,
    const at::Tensor& num_segs,
    int gate_bits,
    int up_bits,
    int down_bits,
    int thin_rows,
    float act_limit)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    e3_metadata_kernel<<<1, E3_META_THREADS, 0, stream>>>(
        expert_count.data_ptr<int64_t>(),
        token_sorted.data_ptr<int64_t>(),
        reinterpret_cast<const half*>(weight_sorted.data_ptr()),
        row_token.data_ptr<int64_t>(),
        reinterpret_cast<half*>(row_weight.data_ptr()),
        row_expert.data_ptr<int>(),
        seg_expert.data_ptr<int>(),
        seg_row0.data_ptr<int>(),
        seg_rows.data_ptr<int>(),
        num_rows.data_ptr<int>(),
        num_segs.data_ptr<int>(),
        thin_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int hidden = static_cast<int>(x.size(1));
    const int intermediate = static_cast<int>(h2.size(1));
    int64_t row_cap = h13_gate.size(0);
    int gx = static_cast<int>((row_cap + E3_WARPS - 1) / E3_WARPS);
    if (gx > 1024) gx = 1024;
    if (gx < 1) gx = 1;
    e3_gather_kernel<<<dim3(gx, hidden / 128), E3_THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        row_token.data_ptr<int64_t>(),
        row_expert.data_ptr<int>(),
        reinterpret_cast<const half* const*>(gate_suh.data_ptr()),
        reinterpret_cast<const half* const*>(up_suh.data_ptr()),
        reinterpret_cast<half*>(h13_gate.data_ptr()),
        reinterpret_cast<half*>(h13_up.data_ptr()),
        num_rows.data_ptr<int>(),
        hidden);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int gy = e3_grid_y(seg_expert.numel());
    const E3GateUpLaunch gateup_args = {
        reinterpret_cast<const half*>(h13_gate.data_ptr()),
        reinterpret_cast<const half*>(h13_up.data_ptr()),
        reinterpret_cast<const uint16_t* const*>(gate_trellis.data_ptr()),
        reinterpret_cast<const uint16_t* const*>(up_trellis.data_ptr()),
        reinterpret_cast<const half* const*>(gate_svh.data_ptr()),
        reinterpret_cast<const half* const*>(up_svh.data_ptr()),
        reinterpret_cast<const half* const*>(down_suh.data_ptr()),
        reinterpret_cast<half*>(h2.data_ptr()),
        seg_expert.data_ptr<int>(), seg_row0.data_ptr<int>(), seg_rows.data_ptr<int>(),
        num_segs.data_ptr<int>(), hidden, intermediate, act_limit, gy, stream,
    };
    e3_dispatch_gateup(gate_bits, up_bits, gateup_args);

    const E3DownLaunch down_args = {
        reinterpret_cast<const half*>(h2.data_ptr()),
        reinterpret_cast<const uint16_t* const*>(down_trellis.data_ptr()),
        reinterpret_cast<const half* const*>(down_svh.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()),
        row_token.data_ptr<int64_t>(),
        reinterpret_cast<const half*>(row_weight.data_ptr()),
        seg_expert.data_ptr<int>(), seg_row0.data_ptr<int>(), seg_rows.data_ptr<int>(),
        num_segs.data_ptr<int>(), intermediate, hidden, gy, stream,
    };
    e3_dispatch_down(down_bits, down_args);
}
