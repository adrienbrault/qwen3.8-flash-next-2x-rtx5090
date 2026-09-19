#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mutex>
#include "hc_mix.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "graph.cuh"

/*

Fused mHC HyperConnection mix() kernel

mix(streams (R, H, D) fp32) -> post (R, H), comb (R, H, H), collapsed (R, D):

  flat = rmsnorm_unweighted(streams.flatten)          (row of H * D values)
  mixv = flat @ fn.T                                  (M = 2H + H^2 outputs)
  pre  = sigmoid(mixv[0:H]  * s0 + base[0:H]) + eps
  post = 2 sigmoid(mixv[H:2H] * s1 + base[H:2H])
  comb = sinkhorn(softmax(mixv[2H:] * s2 + base[2H:]))   (iters alternating row/col norms)
  collapsed = sum_h pre[h] * streams[h]

Two launches, no grid-wide sync, deterministic (fixed reduction order, no atomics):
  K1 partials: grid (chunksA, R); each block reduces its column chunk to M + 1 partials.
  K2 finalize: grid (chunksC, R); EVERY block re-reduces the tiny partial matrix and
    derives rmr + pre redundantly (removes the cross-block dependency), then streams its
    chunk of collapsed; the chunk-0 block also runs the sinkhorn on H^2 lanes of warp 0
    (row sums: shfl_xor 1|2, col sums: shfl_xor 4|8 for H = 4) and writes post/comb.

*/

// Partials blocks are small: at R = 1 the grid is the only parallelism, so favor many
// blocks (row_len / (4 * 64) chunks) over wide ones; the M + 1 block reduce also shrinks
#define NUM_THREADS 256
#define NUM_THREADS_A 64

__device__ __forceinline__ float sigmoidf_(float x)
{
    return 1.0f / (1.0f + __expf(-x));
}

template <int H, int M_, typename FN_T>
__global__ __launch_bounds__(NUM_THREADS_A)
void hc_mix_partials_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const FN_T* __restrict__ fn,         // (M, H * D) float, or half (opt-in, halves traffic)
    float* __restrict__ partials,        // (R, chunksA, M + 1)
    const int row_len,
    const int chunk_cols                 // multiple of 4 * NUM_THREADS_A
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, row_len);

    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int row_len4 = row_len / 4;

    float acc[M + 1];
    #pragma unroll
    for (int k = 0; k <= M; ++k) acc[k] = 0.0f;

    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS_A)
    {
        float4 s = s4[c];
        acc[M] = fmaf(s.x, s.x, acc[M]);
        acc[M] = fmaf(s.y, s.y, acc[M]);
        acc[M] = fmaf(s.z, s.z, acc[M]);
        acc[M] = fmaf(s.w, s.w, acc[M]);
        #pragma unroll
        for (int j = 0; j < M; ++j)
        {
            float4 w;
            if constexpr (std::is_same_v<FN_T, half>)
            {
                // Single vectorized 8-byte load (two half2 loads would double the LDGs)
                int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
                float2 lo = __half22float2(*(const half2*) &pk.x);
                float2 hi = __half22float2(*(const half2*) &pk.y);
                w = make_float4(lo.x, lo.y, hi.x, hi.y);
            }
            else
                w = ((const float4*) fn)[(size_t) j * row_len4 + c];
            float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
            acc[j] += d;
        }
    }

    // Block reduce M + 1 lanes' accumulators
    __shared__ float red[NUM_THREADS_A / 32][M + 1];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    #pragma unroll
    for (int k = 0; k <= M; ++k)
    {
        float v = acc[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();

    if (warp == 0)
    {
        float* out = partials + ((size_t) r * gridDim.x + blockIdx.x) * (M + 1);
        for (int k = lane; k <= M; k += 32)
        {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_THREADS_A / 32; ++w)
                v += red[w][k];
            out[k] = v;
        }
    }
}

template <int H, int M_, bool HEAD, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void hc_mix_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H * D)
    const float* __restrict__ partials,  // (R, chunksA, M + 1)
    const float* __restrict__ base,      // (M)
    const float* __restrict__ scale,     // (3)
    float* __restrict__ post,            // (R, H)
    float* __restrict__ comb,            // (R, H, H)
    void* __restrict__ collapsed,        // (R, D) float or half
    const int D,
    const int chunksA,
    const int chunk_cols_c,              // multiple of 4
    const float rms_eps,
    const float hc_eps,
    const int sinkhorn_iters
)
{
    constexpr int M = M_;
    const int r = blockIdx.y;
    const int row_len = H * D;

    // Re-reduce this row's partials (tiny, L2-resident). Every block does this to avoid
    // cross-block dependencies. Warp-split over the chunk axis: the serial loop sits
    // at the head of the kernel's critical path, so with many partials chunks (small
    // NUM_THREADS_A blocks) a single-thread-per-quantity loop is too long
    __shared__ float mix_s[M + 1];
    __shared__ float pre_s[H];
    __shared__ float red_s[NUM_THREADS / 32][M + 1];
    {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane <= M)
        {
            const float* p = partials + (size_t) r * chunksA * (M + 1) + lane;
            float v = 0.0f;
            for (int i = warp; i < chunksA; i += NUM_THREADS / 32)
                v += p[(size_t) i * (M + 1)];
            red_s[warp][lane] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= M)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_THREADS / 32; ++w)
            v += red_s[w][threadIdx.x];
        mix_s[threadIdx.x] = v;
    }
    __syncthreads();
    float rmr = rsqrtf(mix_s[M] / (float) row_len + rms_eps);
    if (threadIdx.x < H)
        pre_s[threadIdx.x] = sigmoidf_(fmaf(mix_s[threadIdx.x] * rmr, scale[0], base[threadIdx.x])) + hc_eps;
    __syncthreads();

    // Sinkhorn + post/comb writes: chunk-0 block only, one lane per comb element. Runs in
    // a dedicated warp, concurrent with the other warps' phase C: the ~20-iteration
    // normalization is a serial shfl/div latency chain that only needs the M + 1 reduced
    // scalars, so at small R it sets the kernel's critical path; don't stack phase C
    // work in front of it.
    //
    // Lane l = i * H + j; row sums reduce over j (xor 1..H/2), col sums over i (xor H..).
    const bool sink_warp = !HEAD && blockIdx.x == 0 && threadIdx.x < 32;
    if (sink_warp)
    {
        if (threadIdx.x < H)
            post[(size_t) r * H + threadIdx.x] =
                2.0f * sigmoidf_(fmaf(mix_s[H + threadIdx.x] * rmr, scale[1], base[H + threadIdx.x]));

        if (threadIdx.x < H * H)
        {
            const unsigned mask = (H * H == 32) ? 0xffffffffu : ((1u << (H * H)) - 1u);
            float v = fmaf(mix_s[2 * H + threadIdx.x] * rmr, scale[2], base[2 * H + threadIdx.x]);

            // softmax over rows
            float m = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) m = fmaxf(m, __shfl_xor_sync(mask, m, o));
            v = __expf(v - m);
            float s = v;
            #pragma unroll
            for (int o = 1; o < H; o <<= 1) s += __shfl_xor_sync(mask, s, o);
            v = __fdividef(v, s) + hc_eps;

            // column normalize, then (iters - 1) x (row, column)
            float cs = v;
            #pragma unroll
            for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = v;
                #pragma unroll
                for (int o = 1; o < H; o <<= 1) rs += __shfl_xor_sync(mask, rs, o);
                v = __fdividef(v, rs + hc_eps);
                cs = v;
                #pragma unroll
                for (int o = H; o < H * H; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * H * H + threadIdx.x] = v;
        }
        return;
    }

    // Phase C: collapsed chunk, weighted sum over the H stream rows. In the sinkhorn
    // block the first warp is excluded, so the remaining threads re-cover its lanes
    float pre_r[H];
    #pragma unroll
    for (int h = 0; h < H; ++h) pre_r[h] = pre_s[h];

    const bool shrunk = !HEAD && blockIdx.x == 0;
    const int tid = shrunk ? threadIdx.x - 32 : threadIdx.x;
    const int nth = shrunk ? NUM_THREADS - 32 : NUM_THREADS;
    const int c0 = blockIdx.x * chunk_cols_c;
    const int c1 = min(c0 + chunk_cols_c, D);
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    const int D4 = D / 4;
    for (int c = c0 / 4 + tid; c < c1 / 4; c += nth)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) collapsed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) collapsed + (size_t) r * D))[c] = o;
    }
}


/*

GatedResidual (Qwen4Exp, the low-rank elementwise cousin of mHC) fused mix, decode form:

  mix(streams (R, H, D) fp32) -> post (R, H), mixed (R, D):
    normed[h] = rmsnorm(streams[h]) * w[h]           (per-STREAM norm, weighted, w incl +1)
    dots      = cat(down, inject) @ normed.flatten   (M = LR + H outputs, LR = low rank ~320)
    t         = silu(dots[:LR] / H)
    post      = 2 sigmoid(dots[LR:] / H)
    g[h, d]   = sigmoid(up[h * D + d, :] @ t)        (per-CHANNEL gate through the low rank)
    mixed[d]  = mean_h g[h, d] * normed[h, d]

  Same two-launch, no-grid-sync, deterministic shape as hc_mix, restructured for the wide low
  rank (LR >> mHC's M = 24, so per-thread accumulator arrays don't fit):
    K1 gr_dots: one block per (fn row | sum-of-squares), computing per-STREAM partial dots
      against the RAW streams -- by linearity the per-stream rms scale applies in K2, and the
      norm WEIGHT is folded into the fn rows at load time.
    K2 gr_finalize: every block re-derives rmr / t redundantly from the K1 output (the mHC
      finalize pattern), then streams its chunk of mixed, evaluating the per-channel up-gate
      inline -- upT is laid out (LR, H * D) so the serial rank loop reads coalesced and stays
      L2-resident at decode R. NOT for large R (the untiled up/down reads defeat the L2);
      the python side runs a plain half-GEMM path for prefill.

  apply_ is hc_apply with no comb: x[h] += post[h] * y.

*/

#define GR_THREADS_A 128

template <int H>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_dots_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const half* __restrict__ fn,         // (M, H * D) half, norm weight folded in
    float* __restrict__ dots,            // (R, M + 1, H): per-stream dots, row M = sum sq
    const int M,
    const int D
)
{
    const int r = blockIdx.y;
    const int j = blockIdx.x;            // fn row, or M for the sum-of-squares row
    const int D4 = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);

    __shared__ float red[H][GR_THREADS_A / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        float a = 0.0f;
        if (j < M)
        {
            // 16-byte fn loads (8 halves) against two 16-byte stream quads
            const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
            for (int c = threadIdx.x; c < D4 / 2; c += GR_THREADS_A)
            {
                float4 s0 = s4[(size_t) h * D4 + 2 * c];
                float4 s1 = s4[(size_t) h * D4 + 2 * c + 1];
                int4 pk = f8[c];
                half4 w0 = *(half4*) &pk.x;
                half4 w1 = *(half4*) &pk.z;
                a = fmaf(s0.x, LOW_TO_FLOAT(w0.x), a);
                a = fmaf(s0.y, HIGH_TO_FLOAT(w0.x), a);
                a = fmaf(s0.z, LOW_TO_FLOAT(w0.y), a);
                a = fmaf(s0.w, HIGH_TO_FLOAT(w0.y), a);
                a = fmaf(s1.x, LOW_TO_FLOAT(w1.x), a);
                a = fmaf(s1.y, HIGH_TO_FLOAT(w1.x), a);
                a = fmaf(s1.z, LOW_TO_FLOAT(w1.y), a);
                a = fmaf(s1.w, HIGH_TO_FLOAT(w1.y), a);
            }
        }
        else
        {
            for (int c = threadIdx.x; c < D4; c += GR_THREADS_A)
            {
                float4 s = s4[(size_t) h * D4 + c];
                a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xffffffffu, a, offset);
        if (lane == 0) red[h][warp] = a;
    }
    __syncthreads();
    if (threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < GR_THREADS_A / 32; ++w)
            v += red[threadIdx.x][w];
        dots[((size_t) r * (M + 1) + j) * H + threadIdx.x] = v;
    }
}

template <int H, bool HALF_OUT>
__global__ __launch_bounds__(NUM_THREADS)
void gr_finalize_kernel
(
    const float* __restrict__ streams,   // (R, H, D)
    const float* __restrict__ dots,      // (R, M + 1, H) from gr_dots
    const half* __restrict__ upt,        // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const half* __restrict__ w,          // (H * D) half norm weight (incl +1)
    float* __restrict__ post,            // (R, H) or nullptr (final-mixer form)
    void* __restrict__ mixed,            // (R, D) half or float
    const int D,
    const int LR,                        // low rank; M = LR + (post ? H : 0)
    const int chunk_cols,                // multiple of 4
    const float rms_eps
)
{
    const int r = blockIdx.y;
    const int M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;

    // Redundant per-block head derivation (no cross-block dependency): rmr from the sumsq
    // row, then the silu'd low-rank activations into shared memory
    __shared__ float rmr_s[H];
    extern __shared__ float t_s[];
    if (threadIdx.x < H)
        rmr_s[threadIdx.x] = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    for (int i = threadIdx.x; i < LR; i += NUM_THREADS)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) i * H + h], v);
        v *= inv_h;
        t_s[i] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && threadIdx.x < H)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            v = fmaf(rmr_s[h], dr[(size_t) (LR + threadIdx.x) * H + h], v);
        post[(size_t) r * H + threadIdx.x] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();

    // Streamed mixed chunk: one WARP per column quad, the LR-long up-gate dots split across
    // the lanes (lane l covers ranks l, l+32, ...) and shfl-reduced -- at decode R the total
    // column count is small (R * D / 4), so per-thread columns would leave the GPU nearly
    // idle with each thread serializing the rank loop
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const float4* s4 = (const float4*) (streams + (size_t) r * H * D);
    for (int c = c0 / 4 + warp; c < c1 / 4; c += NUM_THREADS / 32)
    {
        float4 g[H];
        #pragma unroll
        for (int h = 0; h < H; ++h) g[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int i = lane; i < LR; i += 32)
        {
            float ti = t_s[i];
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                // upx layout (H, D / 4, LR, 4): consecutive lanes (consecutive i) read
                // consecutive 8-byte quads, so the rank loop is fully coalesced
                half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
                g[h].x = fmaf(ti, LOW_TO_FLOAT(u.x), g[h].x);
                g[h].y = fmaf(ti, HIGH_TO_FLOAT(u.x), g[h].y);
                g[h].z = fmaf(ti, LOW_TO_FLOAT(u.y), g[h].z);
                g[h].w = fmaf(ti, HIGH_TO_FLOAT(u.y), g[h].w);
            }
        }
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[h].x += __shfl_xor_sync(0xffffffffu, g[h].x, offset);
                g[h].y += __shfl_xor_sync(0xffffffffu, g[h].y, offset);
                g[h].z += __shfl_xor_sync(0xffffffffu, g[h].z, offset);
                g[h].w += __shfl_xor_sync(0xffffffffu, g[h].w, offset);
            }
        if (lane != 0) continue;
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 s = s4[(size_t) h * D4 + c];
            half4 wq = *(const half4*) (w + (size_t) h * D + 4 * c);
            float coef = rmr_s[h] * inv_h;
            o.x = fmaf(sigmoidf_(g[h].x) * coef * LOW_TO_FLOAT(wq.x),  s.x, o.x);
            o.y = fmaf(sigmoidf_(g[h].y) * coef * HIGH_TO_FLOAT(wq.x), s.y, o.y);
            o.z = fmaf(sigmoidf_(g[h].z) * coef * LOW_TO_FLOAT(wq.y),  s.z, o.z);
            o.w = fmaf(sigmoidf_(g[h].w) * coef * HIGH_TO_FLOAT(wq.y), s.w, o.w);
        }
        if (HALF_OUT)
        {
            half2* out2 = (half2*) ((half*) mixed + (size_t) r * D);
            out2[c * 2] = __floats2half2_rn(o.x, o.y);
            out2[c * 2 + 1] = __floats2half2_rn(o.z, o.w);
        }
        else
            ((float4*) ((float*) mixed + (size_t) r * D))[c] = o;
    }
}


// Residual update for one sublayer site: x[h, d] <- post[h] * y[d] + sum_h' comb[h', h] *
// x[h', d]. Without comb (GatedResidual): x[h, d] <- post[h] * y[d] + x[h, d]. Pure per-column
// mix of the H stream rows, so it runs in place: each thread loads all H values of its columns
// into registers before writing any back.

template <int H, typename Y_T, bool HAS_COMB>
__global__ __launch_bounds__(NUM_THREADS)
void hc_apply_kernel
(
    float* __restrict__ x,               // (R, H, D), updated in place
    const Y_T* __restrict__ y,           // (R, D) float or half
    const float* __restrict__ post,      // (R, H)
    const float* __restrict__ comb,      // (R, H, H), or null
    const int D,
    const int chunk_cols                 // multiple of 4
)
{
    const int r = blockIdx.y;
    const int c0 = blockIdx.x * chunk_cols;
    const int c1 = min(c0 + chunk_cols, D);
    const int D4 = D / 4;

    float post_r[H];
    float comb_r[H][H];
    #pragma unroll
    for (int h = 0; h < H; ++h)
    {
        post_r[h] = __ldg(post + (size_t) r * H + h);
        if (HAS_COMB)
        {
            #pragma unroll
            for (int g = 0; g < H; ++g)
                comb_r[h][g] = __ldg(comb + ((size_t) r * H + h) * H + g);
        }
    }

    float4* x4 = (float4*) (x + (size_t) r * H * D);
    for (int c = c0 / 4 + threadIdx.x; c < c1 / 4; c += NUM_THREADS)
    {
        float4 xv[H];
        #pragma unroll
        for (int h = 0; h < H; ++h)
            xv[h] = x4[(size_t) h * D4 + c];

        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[c * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[c * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[c];

        #pragma unroll
        for (int h = 0; h < H; ++h)
        {
            float4 o;
            o.x = post_r[h] * yv.x;
            o.y = post_r[h] * yv.y;
            o.z = post_r[h] * yv.z;
            o.w = post_r[h] * yv.w;
            if (HAS_COMB)
            {
                #pragma unroll
                for (int g = 0; g < H; ++g)
                {
                    o.x = fmaf(comb_r[g][h], xv[g].x, o.x);
                    o.y = fmaf(comb_r[g][h], xv[g].y, o.y);
                    o.z = fmaf(comb_r[g][h], xv[g].z, o.z);
                    o.w = fmaf(comb_r[g][h], xv[g].w, o.w);
                }
            }
            else
            {
                o.x += xv[h].x;
                o.y += xv[h].y;
                o.z += xv[h].z;
                o.w += xv[h].w;
            }
            x4[(size_t) h * D4 + c] = o;
        }
    }
}

// Shared launch logic. mode: fn rows M = 2H + H^2 (mix) or H (head)

static void hc_mix_launch
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    float rms_eps,
    float hc_eps,
    int sinkhorn_iters,
    at::Tensor& partials,
    at::Tensor* post,
    at::Tensor* comb,
    at::Tensor& collapsed,
    Graph* graph
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = graph ? graph->capture_stream : at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous(), "hc_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int row_len = H * D;
    int M = fn.size(0);
    bool head = M == H;
    TORCH_CHECK(H == 4, "hc_mix: H = 4 only");
    TORCH_CHECK(head || M == 2 * H + H * H, "hc_mix: fn rows must be H or 2H + H^2");
    TORCH_CHECK(fn.size(1) == row_len && D % 4 == 0, "hc_mix: dims");
    bool fn_half = fn.dtype() == at::kHalf;
    if (!fn_half) TORCH_CHECK_DTYPE(fn, kFloat);
    TORCH_CHECK_DTYPE(base, kFloat);
    TORCH_CHECK_DTYPE(scale, kFloat);

    int n_chunks_a = partials.size(1);
    int chunk_cols = ((row_len / n_chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    n_chunks_a = (row_len + chunk_cols - 1) / chunk_cols;
    TORCH_CHECK(n_chunks_a <= partials.size(1), "hc_mix: partials workspace too small");

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols_c = ((D / chunks_c + 4 * NUM_THREADS - 1) / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks_c = (D + chunk_cols_c - 1) / chunk_cols_c;

    bool half_out = collapsed.dtype() == at::kHalf;

    dim3 grid_a(n_chunks_a, R);
    dim3 grid_c(n_chunks_c, R);
    #define ARGS_A(FN_T) \
        (const float*) streams.data_ptr(), (const FN_T*) fn.data_ptr(), \
        (float*) partials.data_ptr(), row_len, chunk_cols
    #define ARGS_C(POST, COMB) \
        (const float*) streams.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), \
        POST, COMB, collapsed.data_ptr(), \
        D, n_chunks_a, chunk_cols_c, rms_eps, hc_eps, sinkhorn_iters
    if (!head)
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 24, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 24, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        float* post_p = (float*) post->data_ptr();
        float* comb_p = (float*) comb->data_ptr();
        if (half_out)
            hc_mix_finalize_kernel<4, 24, false, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
        else
            hc_mix_finalize_kernel<4, 24, false, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(post_p, comb_p));
    }
    else
    {
        if (fn_half)
            hc_mix_partials_kernel<4, 4, half><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(half));
        else
            hc_mix_partials_kernel<4, 4, float><<<grid_a, NUM_THREADS_A, 0, stream>>>(ARGS_A(float));
        cuda_check(cudaPeekAtLastError());
        if (half_out)
            hc_mix_finalize_kernel<4, 4, true, true><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
        else
            hc_mix_finalize_kernel<4, 4, true, false><<<grid_c, NUM_THREADS, 0, stream>>>(ARGS_C(nullptr, nullptr));
    }
    #undef ARGS_A
    #undef ARGS_C
    cuda_check(cudaPeekAtLastError());
}

int hc_mix_num_chunks(int R, int row_len)
{
    int chunks_a = std::min(128, std::max(1, 512 / std::max(R, 1)));
    int chunk_cols = ((row_len / chunks_a + 4 * NUM_THREADS_A - 1) / (4 * NUM_THREADS_A)) * (4 * NUM_THREADS_A);
    return (row_len + chunk_cols - 1) / chunk_cols;
}

void hc_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (2H + H^2, H * D) float
    const at::Tensor& base,              // (2H + H^2) float
    const at::Tensor& scale,             // (3) float
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,                 // (R, chunks, M + 1) float workspace
    at::Tensor post,                     // (R, H) float out
    at::Tensor comb,                     // (R, H, H) float out
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        (int) sinkhorn_iters,
        partials,
        &post,
        &comb,
        collapsed,
        nullptr
    );
}

void hc_head
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (H, H * D) float
    const at::Tensor& base,              // (H) float
    const at::Tensor& scale,             // (1) float
    double rms_eps,
    double hc_eps,
    at::Tensor partials,                 // (R, chunks, H + 1) float workspace
    at::Tensor collapsed                 // (R, D) float or half out
)
{
    hc_mix_launch
    (
        streams,
        fn,
        base,
        scale,
        (float) rms_eps,
        (float) hc_eps,
        0,
        partials,
        nullptr,
        nullptr,
        collapsed,
        nullptr
    );
}

void hc_apply
(
    at::Tensor x,                        // (R, H, D) float, updated IN PLACE
    const at::Tensor& y,                 // (R, D) float or half
    const at::Tensor& post,              // (R, H) float
    const c10::optional<at::Tensor>& comb   // (R, H, H) float, or none (x[h] += post[h] * y)
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(x, kFloat);
    TORCH_CHECK_DTYPE(post, kFloat);
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous() && post.is_contiguous(), "hc_apply: contiguous inputs required");
    int R = x.size(0);
    int H = x.size(1);
    int D = x.size(2);
    TORCH_CHECK(H == 4, "hc_apply: H = 4 only");
    TORCH_CHECK(D % 4 == 0, "hc_apply: dims");
    TORCH_CHECK(y.size(0) == R && y.size(-1) == D, "hc_apply: y shape");
    TORCH_CHECK(post.size(0) == R, "hc_apply: gate shapes");
    const float* comb_p = nullptr;
    if (comb)
    {
        TORCH_CHECK_DTYPE(comb.value(), kFloat);
        TORCH_CHECK(comb.value().is_contiguous() && comb.value().size(0) == R, "hc_apply: comb shape");
        comb_p = (const float*) comb.value().data_ptr();
    }

    int chunks_c = std::min(32, std::max(1, 256 / R));
    int chunk_cols = ((D / chunks_c + 4 * NUM_THREADS - 1)
                      / (4 * NUM_THREADS)) * (4 * NUM_THREADS);
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;

    dim3 grid(n_chunks, R);
    #define ARGS(Y_T) \
        (float*) x.data_ptr(), (const Y_T*) y.data_ptr(), \
        (const float*) post.data_ptr(), comb_p, D, chunk_cols
    if (y.dtype() == at::kHalf)
    {
        if (comb_p) hc_apply_kernel<4, half, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
        else        hc_apply_kernel<4, half, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(half));
    }
    else
    {
        if (comb_p) hc_apply_kernel<4, float, true><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
        else        hc_apply_kernel<4, float, false><<<grid, NUM_THREADS, 0, stream>>>(ARGS(float));
    }
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

void gr_mix
(
    const at::Tensor& streams,           // (R, H, D) float
    const at::Tensor& fn,                // (M, H * D) half: cat(down, inject) * w
    const at::Tensor& upt,               // (H, D / 4, LR, 4) half: up repacked lane-contiguous
    const at::Tensor& w,                 // (H * D) half norm weight (incl +1)
    double rms_eps,
    at::Tensor dots,                     // (R, M + 1, H) float workspace
    c10::optional<at::Tensor> post,      // (R, H) float out, or none (final-mixer form)
    at::Tensor mixed                     // (R, D) half or float out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn, kHalf);
    TORCH_CHECK_DTYPE(upt, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && upt.is_contiguous() &&
                w.is_contiguous() && dots.is_contiguous(), "gr_mix: contiguous inputs required");
    int R = streams.size(0);
    int H = streams.size(1);
    int D = streams.size(2);
    int M = fn.size(0);
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix: upt must be the (H, D / 4, LR, 4) repacked layout");
    int LR = upt.size(2);
    TORCH_CHECK(H == 4, "gr_mix: H = 4 only");
    TORCH_CHECK(D % 8 == 0, "gr_mix: dims");
    TORCH_CHECK(M == LR + (post ? H : 0), "gr_mix: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == H * D && w.numel() == H * D, "gr_mix: dims");
    TORCH_CHECK(dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H, "gr_mix: dots shape");

    dim3 grid_a(M + 1, R);
    gr_dots_kernel<4><<<grid_a, GR_THREADS_A, 0, stream>>>
    (
        (const float*) streams.data_ptr(), (const half*) fn.data_ptr(),
        (float*) dots.data_ptr(), M, D
    );
    cuda_check(cudaPeekAtLastError());

    // Phase C is warp-per-column-quad: chunk at warp granularity (4 * NUM_THREADS / 32
    // columns) so small R still fills the device
    const int gran = 4 * (NUM_THREADS / 32);
    int chunks_c = std::max(1, std::min((D + gran - 1) / gran, 512 / R));
    int chunk_cols = ((D / chunks_c + gran - 1) / gran) * gran;
    int n_chunks = (D + chunk_cols - 1) / chunk_cols;
    dim3 grid_c(n_chunks, R);
    int smem = LR * sizeof(float);
    float* post_p = post ? (float*) post.value().data_ptr() : nullptr;
    #define ARGS \
        (const float*) streams.data_ptr(), (const float*) dots.data_ptr(), \
        (const half*) upt.data_ptr(), (const half*) w.data_ptr(), \
        post_p, mixed.data_ptr(), D, LR, chunk_cols, (float) rms_eps
    if (mixed.dtype() == at::kHalf)
        gr_finalize_kernel<4, true><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    else
        gr_finalize_kernel<4, false><<<grid_c, NUM_THREADS, smem, stream>>>(ARGS);
    #undef ARGS
    cuda_check(cudaPeekAtLastError());
}

// R2 exact-order SIMT mixer. Three launches trade one small state kernel for
// eliminating r1's per-rank-tile RMS, virtual warps and stream-staging barriers.
constexpr int GR_V2_H = 4;
constexpr int GR_V2_WARP = 32;
constexpr int GR_V2_J = 4;
constexpr int GR_V2_ROWS = 8;
constexpr int GR_V2_QUADS = 2;
constexpr int GR_V2_UP_THREADS = GR_V2_H * GR_V2_QUADS * GR_V2_WARP;

// Each physical thread is the SAME thread as in gr_dots_kernel. Independent
// rows/ranks share loads; no dot accumulator is split, combined or reordered.
template <int B>
__global__ __launch_bounds__(GR_THREADS_A)
void gr_v2_dots_kernel
(
    const float* __restrict__ streams, const half* __restrict__ fn,
    float* __restrict__ dots, const int R, const int D, const int M
)
{
    constexpr int H = GR_V2_H, J = GR_V2_J, V = GR_THREADS_A / GR_V2_WARP;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int h = blockIdx.z, r0 = blockIdx.y * B;
    const int j0 = blockIdx.x * J;
    __shared__ float red[B][J][V];
    // A separate block (in this same launch) owns each row/stream's sumsq.
    if (blockIdx.x == (M + J - 1) / J)
    {
        float a[B] = {};
        for (int c = tid; c < D / 4; c += GR_THREADS_A)
        {
            #pragma unroll
            for (int b = 0; b < B; ++b)
                if (r0 + b < R)
                {
                    float4 s = ((const float4*) streams)[((size_t) (r0 + b) * H + h) * (D / 4) + c];
                    a[b] = fmaf(s.x, s.x, fmaf(s.y, s.y,
                           fmaf(s.z, s.z, fmaf(s.w, s.w, a[b]))));
                }
        }
        #pragma unroll
        for (int b = 0; b < B; ++b)
        {
            for (int offset = 16; offset > 0; offset >>= 1)
                a[b] += __shfl_down_sync(0xffffffffu, a[b], offset);
            if (lane == 0) red[b][0][warp] = a[b];
        }
        __syncthreads();
        if (tid < B && r0 + tid < R)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int v = 0; v < V; ++v) sum += red[tid][0][v];
            dots[((size_t) (r0 + tid) * (M + 1) + M) * H + h] = sum;
        }
        return;  // block-uniform; no other block needs these results in K1
    }
    float a[B][J] = {};
    for (int c = tid; c < D / 8; c += GR_THREADS_A)
    {
        float4 f0[J], f1[J];
        #pragma unroll
        for (int j = 0; j < J; ++j)
            if (j0 + j < M)
            {
                int4 pk = ((const int4*) (fn + ((size_t) (j0 + j) * H + h) * D))[c];
                half4 w0 = *(half4*) &pk.x, w1 = *(half4*) &pk.z;
                f0[j] = make_float4(LOW_TO_FLOAT(w0.x), HIGH_TO_FLOAT(w0.x),
                                   LOW_TO_FLOAT(w0.y), HIGH_TO_FLOAT(w0.y));
                f1[j] = make_float4(LOW_TO_FLOAT(w1.x), HIGH_TO_FLOAT(w1.x),
                                   LOW_TO_FLOAT(w1.y), HIGH_TO_FLOAT(w1.y));
            }
        #pragma unroll
        for (int b = 0; b < B; ++b)
            if (r0 + b < R)
            {
                const float4* s4 = (const float4*) (streams + ((size_t) (r0 + b) * H + h) * D);
                float4 s0 = s4[2 * c], s1 = s4[2 * c + 1];
                #pragma unroll
                for (int j = 0; j < J; ++j)
                    if (j0 + j < M)
                    {
                        a[b][j] = fmaf(s0.x, f0[j].x, a[b][j]);
                        a[b][j] = fmaf(s0.y, f0[j].y, a[b][j]);
                        a[b][j] = fmaf(s0.z, f0[j].z, a[b][j]);
                        a[b][j] = fmaf(s0.w, f0[j].w, a[b][j]);
                        a[b][j] = fmaf(s1.x, f1[j].x, a[b][j]);
                        a[b][j] = fmaf(s1.y, f1[j].y, a[b][j]);
                        a[b][j] = fmaf(s1.z, f1[j].z, a[b][j]);
                        a[b][j] = fmaf(s1.w, f1[j].w, a[b][j]);
                    }
            }
    }
    #pragma unroll
    for (int b = 0; b < B; ++b)
        #pragma unroll
        for (int j = 0; j < J; ++j)
        {
            for (int offset = 16; offset > 0; offset >>= 1)
                a[b][j] += __shfl_down_sync(0xffffffffu, a[b][j], offset);
            if (lane == 0) red[b][j][warp] = a[b][j];
        }
    __syncthreads();
    if (tid < B * J)
    {
        const int b = tid / J, j = tid % J;
        if (r0 + b < R && j0 + j < M)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int v = 0; v < V; ++v) sum += red[b][j][v];
            dots[((size_t) (r0 + b) * (M + 1) + j0 + j) * H + h] = sum;
        }
    }
}

__global__ __launch_bounds__(GR_THREADS_A)
void gr_v2_state_kernel
(
    const float* __restrict__ dots, float* __restrict__ state,
    float* __restrict__ post, const int D, const int LR, const float rms_eps
)
{
    constexpr int H = GR_V2_H;
    const int r = blockIdx.x, M = LR + (post ? H : 0);
    const float* dr = dots + (size_t) r * (M + 1) * H;
    __shared__ float rmr[H];
    if (threadIdx.x < H)
    {
        float rm = rsqrtf(dr[(size_t) M * H + threadIdx.x] / (float) D + rms_eps);
        rmr[threadIdx.x] = rm;
        state[(size_t) r * (LR + H) + LR + threadIdx.x] = rm;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < M; i += GR_THREADS_A)
    {
        float v = 0.0f;
        #pragma unroll
        for (int h = 0; h < H; ++h) v = fmaf(rmr[h], dr[(size_t) i * H + h], v);
        v *= 1.0f / (float) H;
        if (i < LR) state[(size_t) r * (LR + H) + i] = v * sigmoidf_(v);
        else post[(size_t) r * H + i - LR] = 2.0f * sigmoidf_(v);
    }
}

template <int B, bool HALF_OUT>
__global__ __launch_bounds__(GR_V2_UP_THREADS)
void gr_v2_up_kernel
(
    const float* __restrict__ streams, const float* __restrict__ state,
    const half* __restrict__ upt, const half* __restrict__ w,
    void* __restrict__ mixed, const int R, const int D, const int LR
)
{
    constexpr int H = GR_V2_H, Q = GR_V2_QUADS;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int h = warp % H, cq = warp / H;
    const int c = blockIdx.x * Q + cq, r0 = blockIdx.y * B;
    extern __shared__ float head[];
    __shared__ float gates[B][Q][H][4];
    for (int k = threadIdx.x; k < B * (LR + H); k += GR_V2_UP_THREADS)
    {
        int b = k / (LR + H);
        head[k] = r0 + b < R ? state[(size_t) r0 * (LR + H) + k] : 0.0f;
    }
    __syncthreads();
    float4 g[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) g[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (c < D / 4)
        for (int i = lane; i < LR; i += GR_V2_WARP)
        {
            half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
            float4 uf = make_float4(LOW_TO_FLOAT(u.x), HIGH_TO_FLOAT(u.x),
                                   LOW_TO_FLOAT(u.y), HIGH_TO_FLOAT(u.y));
            #pragma unroll
            for (int b = 0; b < B; ++b)
            {
                float ti = head[b * (LR + H) + i];
                g[b].x = fmaf(ti, uf.x, g[b].x);
                g[b].y = fmaf(ti, uf.y, g[b].y);
                g[b].z = fmaf(ti, uf.z, g[b].z);
                g[b].w = fmaf(ti, uf.w, g[b].w);
            }
        }
    #pragma unroll
    for (int b = 0; b < B; ++b)
    {
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            g[b].x += __shfl_xor_sync(0xffffffffu, g[b].x, offset);
            g[b].y += __shfl_xor_sync(0xffffffffu, g[b].y, offset);
            g[b].z += __shfl_xor_sync(0xffffffffu, g[b].z, offset);
            g[b].w += __shfl_xor_sync(0xffffffffu, g[b].w, offset);
        }
        if (lane == 0)
        {
            gates[b][cq][h][0] = g[b].x;
            gates[b][cq][h][1] = g[b].y;
            gates[b][cq][h][2] = g[b].z;
            gates[b][cq][h][3] = g[b].w;
        }
    }
    __syncthreads();
    if (threadIdx.x < B * Q * 4)
    {
        const int b = threadIdx.x / (Q * 4), q = (threadIdx.x / 4) % Q, k = threadIdx.x % 4;
        const int d = (blockIdx.x * Q + q) * 4 + k;
        if (r0 + b < R && d < D)
        {
            float o = 0.0f;
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float s = streams[((size_t) (r0 + b) * H + h) * D + d];
                float coef = head[b * (LR + H) + LR + h] * (1.0f / (float) H);
                o = fmaf(sigmoidf_(gates[b][q][h][k]) * coef * __half2float(w[h * D + d]), s, o);
            }
            if (HALF_OUT) ((half*) mixed)[(size_t) (r0 + b) * D + d] = __float2half_rn(o);
            else ((float*) mixed)[(size_t) (r0 + b) * D + d] = o;
        }
    }
}

// Exact state phase followed by the served up phase in one cooperative launch. Only the
// x-block at column zero derives state for its B-row group; grid.sync() replaces the launch
// boundary before every column block reads that state. Each scalar keeps the served state
// kernel's H-ordered FMA chain, rsqrt and sigmoid operations.
template <int B, bool HALF_OUT>
__global__ __launch_bounds__(GR_V2_UP_THREADS)
void gr_v2_state_up_kernel
(
    const float* __restrict__ streams, const float* __restrict__ dots,
    float* __restrict__ state, float* __restrict__ post,
    const half* __restrict__ upt, const half* __restrict__ w,
    void* __restrict__ mixed, const int R, const int D, const int LR,
    const float rms_eps
)
{
    constexpr int H = GR_V2_H, Q = GR_V2_QUADS;
    const int r0 = blockIdx.y * B;
    if (blockIdx.x == 0)
    {
        const int M = LR + (post ? H : 0);
        for (int k = threadIdx.x; k < B * H; k += GR_V2_UP_THREADS)
        {
            const int b = k / H, h = k % H, r = r0 + b;
            if (r < R)
            {
                const float* dr = dots + (size_t) r * (M + 1) * H;
                const float rm = rsqrtf(dr[(size_t) M * H + h] / (float) D + rms_eps);
                state[(size_t) r * (LR + H) + LR + h] = rm;
            }
        }
        __syncthreads();
        for (int k = threadIdx.x; k < B * M; k += GR_V2_UP_THREADS)
        {
            const int b = k / M, i = k % M, r = r0 + b;
            if (r < R)
            {
                const float* dr = dots + (size_t) r * (M + 1) * H;
                const float* sr = state + (size_t) r * (LR + H);
                float v = 0.0f;
                #pragma unroll
                for (int h = 0; h < H; ++h)
                    v = fmaf(sr[LR + h], dr[(size_t) i * H + h], v);
                v *= 1.0f / (float) H;
                if (i < LR) state[(size_t) r * (LR + H) + i] = v * sigmoidf_(v);
                else post[(size_t) r * H + i - LR] = 2.0f * sigmoidf_(v);
            }
        }
    }
    cooperative_groups::this_grid().sync();

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int h = warp % H, cq = warp / H;
    const int c = blockIdx.x * Q + cq;
    extern __shared__ float head[];
    __shared__ float gates[B][Q][H][4];
    for (int k = threadIdx.x; k < B * (LR + H); k += GR_V2_UP_THREADS)
    {
        int b = k / (LR + H);
        head[k] = r0 + b < R ? state[(size_t) r0 * (LR + H) + k] : 0.0f;
    }
    __syncthreads();
    float4 g[B];
    #pragma unroll
    for (int b = 0; b < B; ++b) g[b] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (c < D / 4)
        for (int i = lane; i < LR; i += GR_V2_WARP)
        {
            half4 u = *(const half4*) (upt + ((((size_t) h * (D / 4) + c) * LR + i) * 4));
            float4 uf = make_float4(LOW_TO_FLOAT(u.x), HIGH_TO_FLOAT(u.x),
                                   LOW_TO_FLOAT(u.y), HIGH_TO_FLOAT(u.y));
            #pragma unroll
            for (int b = 0; b < B; ++b)
            {
                float ti = head[b * (LR + H) + i];
                g[b].x = fmaf(ti, uf.x, g[b].x);
                g[b].y = fmaf(ti, uf.y, g[b].y);
                g[b].z = fmaf(ti, uf.z, g[b].z);
                g[b].w = fmaf(ti, uf.w, g[b].w);
            }
        }
    #pragma unroll
    for (int b = 0; b < B; ++b)
    {
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            g[b].x += __shfl_xor_sync(0xffffffffu, g[b].x, offset);
            g[b].y += __shfl_xor_sync(0xffffffffu, g[b].y, offset);
            g[b].z += __shfl_xor_sync(0xffffffffu, g[b].z, offset);
            g[b].w += __shfl_xor_sync(0xffffffffu, g[b].w, offset);
        }
        if (lane == 0)
        {
            gates[b][cq][h][0] = g[b].x;
            gates[b][cq][h][1] = g[b].y;
            gates[b][cq][h][2] = g[b].z;
            gates[b][cq][h][3] = g[b].w;
        }
    }
    __syncthreads();
    if (threadIdx.x < B * Q * 4)
    {
        const int b = threadIdx.x / (Q * 4), q = (threadIdx.x / 4) % Q, k = threadIdx.x % 4;
        const int d = (blockIdx.x * Q + q) * 4 + k;
        if (r0 + b < R && d < D)
        {
            float o = 0.0f;
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                float s = streams[((size_t) (r0 + b) * H + h) * D + d];
                float coef = head[b * (LR + H) + LR + h] * (1.0f / (float) H);
                o = fmaf(sigmoidf_(gates[b][q][h][k]) * coef * __half2float(w[h * D + d]), s, o);
            }
            if (HALF_OUT) ((half*) mixed)[(size_t) (r0 + b) * D + d] = __float2half_rn(o);
            else ((float*) mixed)[(size_t) (r0 + b) * D + d] = o;
        }
    }
}

template <int B, bool HALF_OUT>
static bool gr_v2_state_up_launch
(
    const at::Tensor& streams, const at::Tensor& dots, at::Tensor& state,
    float* post, const at::Tensor& upt, const at::Tensor& w, at::Tensor& mixed,
    int R, int D, int LR, float rms_eps, cudaStream_t stream
)
{
    dim3 grid((D / 4 + GR_V2_QUADS - 1) / GR_V2_QUADS, (R + B - 1) / B);
    const int smem = B * (LR + GR_V2_H) * sizeof(float);
    // Cache the cooperative capacity per template/device. These CUDA runtime queries would
    // otherwise run at every one of the ~100 mixer sites in a decode step and overwhelm the
    // launch-gap saving being tested.
    constexpr int MAX_CACHED_DEVICES = 32;
    static std::once_flag capacity_once[MAX_CACHED_DEVICES];
    static int capacity[MAX_CACHED_DEVICES] = {};
    static int capacity_lr[MAX_CACHED_DEVICES] = {};
    const int dev = streams.get_device();
    TORCH_CHECK(dev >= 0 && dev < MAX_CACHED_DEVICES, "gr_mix_v2_fused: device index out of cache range");
    std::call_once(capacity_once[dev], [&]()
    {
        int blocks_per_sm = 0, sms = 0, coop = 0;
        cuda_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        cuda_check(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev));
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, gr_v2_state_up_kernel<B, HALF_OUT>, GR_V2_UP_THREADS, smem));
        capacity_lr[dev] = LR;
        capacity[dev] = coop ? blocks_per_sm * sms : -1;
    });
    if (capacity_lr[dev] != LR || capacity[dev] < 0 ||
        (int64_t) grid.x * grid.y > capacity[dev]) return false;

    const float* streams_p = (const float*) streams.data_ptr();
    const float* dots_p = (const float*) dots.data_ptr();
    float* state_p = (float*) state.data_ptr();
    const half* upt_p = (const half*) upt.data_ptr();
    const half* w_p = (const half*) w.data_ptr();
    void* mixed_p = mixed.data_ptr();
    void* args[] = {
        &streams_p, &dots_p, &state_p, &post, &upt_p, &w_p, &mixed_p,
        &R, &D, &LR, &rms_eps
    };
    cuda_check(cudaLaunchCooperativeKernel(
        (void*) gr_v2_state_up_kernel<B, HALF_OUT>, grid, GR_V2_UP_THREADS,
        args, smem, stream));
    return true;
}

template <int B>
static void gr_v2_launch
(
    const at::Tensor& streams, const at::Tensor& fn, const at::Tensor& upt,
    const at::Tensor& w, float rms_eps, at::Tensor& dots, at::Tensor& state,
    float* post, at::Tensor& mixed, int R, int D, int LR, cudaStream_t stream
)
{
    const int M = LR + (post ? GR_V2_H : 0);
    dim3 grid_a((M + GR_V2_J - 1) / GR_V2_J + 1, (R + B - 1) / B, GR_V2_H);
    gr_v2_dots_kernel<B><<<grid_a, GR_THREADS_A, 0, stream>>>
    ((const float*) streams.data_ptr(), (const half*) fn.data_ptr(), (float*) dots.data_ptr(), R, D, M);
    cuda_check(cudaPeekAtLastError());
    gr_v2_state_kernel<<<R, GR_THREADS_A, 0, stream>>>
    ((const float*) dots.data_ptr(), (float*) state.data_ptr(), post, D, LR, rms_eps);
    cuda_check(cudaPeekAtLastError());
    dim3 grid_b((D / 4 + GR_V2_QUADS - 1) / GR_V2_QUADS, (R + B - 1) / B);
    const int smem = B * (LR + GR_V2_H) * sizeof(float);
    #define GR_V2_ARGS \
        (const float*) streams.data_ptr(), (const float*) state.data_ptr(), \
        (const half*) upt.data_ptr(), (const half*) w.data_ptr(), mixed.data_ptr(), R, D, LR
    if (mixed.dtype() == at::kHalf)
        gr_v2_up_kernel<B, true><<<grid_b, GR_V2_UP_THREADS, smem, stream>>>(GR_V2_ARGS);
    else
        gr_v2_up_kernel<B, false><<<grid_b, GR_V2_UP_THREADS, smem, stream>>>(GR_V2_ARGS);
    #undef GR_V2_ARGS
    cuda_check(cudaPeekAtLastError());
}

template <int B>
static bool gr_v2_launch_fused
(
    const at::Tensor& streams, const at::Tensor& fn, const at::Tensor& upt,
    const at::Tensor& w, float rms_eps, at::Tensor& dots, at::Tensor& state,
    float* post, at::Tensor& mixed, int R, int D, int LR, cudaStream_t stream
)
{
    const int M = LR + (post ? GR_V2_H : 0);
    dim3 grid_a((M + GR_V2_J - 1) / GR_V2_J + 1, (R + B - 1) / B, GR_V2_H);
    gr_v2_dots_kernel<B><<<grid_a, GR_THREADS_A, 0, stream>>>
    ((const float*) streams.data_ptr(), (const half*) fn.data_ptr(), (float*) dots.data_ptr(), R, D, M);
    cuda_check(cudaPeekAtLastError());

    bool launched = mixed.dtype() == at::kHalf
        ? gr_v2_state_up_launch<B, true>(streams, dots, state, post, upt, w, mixed,
                                        R, D, LR, rms_eps, stream)
        : gr_v2_state_up_launch<B, false>(streams, dots, state, post, upt, w, mixed,
                                         R, D, LR, rms_eps, stream);
    if (!launched)
    {
        // Conservative fallback for devices/shapes whose full grid cannot be co-resident.
        gr_v2_state_kernel<<<R, GR_THREADS_A, 0, stream>>>
        ((const float*) dots.data_ptr(), (float*) state.data_ptr(), post, D, LR, rms_eps);
        cuda_check(cudaPeekAtLastError());
        dim3 grid_b((D / 4 + GR_V2_QUADS - 1) / GR_V2_QUADS, (R + B - 1) / B);
        const int smem = B * (LR + GR_V2_H) * sizeof(float);
        #define GR_V2_FALLBACK_ARGS \
            (const float*) streams.data_ptr(), (const float*) state.data_ptr(), \
            (const half*) upt.data_ptr(), (const half*) w.data_ptr(), mixed.data_ptr(), R, D, LR
        if (mixed.dtype() == at::kHalf)
            gr_v2_up_kernel<B, true><<<grid_b, GR_V2_UP_THREADS, smem, stream>>>(GR_V2_FALLBACK_ARGS);
        else
            gr_v2_up_kernel<B, false><<<grid_b, GR_V2_UP_THREADS, smem, stream>>>(GR_V2_FALLBACK_ARGS);
        #undef GR_V2_FALLBACK_ARGS
        cuda_check(cudaPeekAtLastError());
    }
    return launched;
}

void gr_mix_v2
(
    const at::Tensor& streams, const at::Tensor& fn, const at::Tensor& upt,
    const at::Tensor& w, double rms_eps, at::Tensor dots, at::Tensor state,
    c10::optional<at::Tensor> post, at::Tensor mixed
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn, kHalf);
    TORCH_CHECK_DTYPE(upt, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(state, kFloat);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.dim() == 3 && fn.dim() == 2 && state.dim() == 2,
                "gr_mix_v2: input/workspace rank");
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && upt.is_contiguous() &&
                w.is_contiguous() && state.is_contiguous() && dots.is_contiguous(), "gr_mix_v2: contiguous inputs required");
    const int R = streams.size(0), H = streams.size(1), D = streams.size(2), M = fn.size(0);
    TORCH_CHECK(R >= 1 && R <= 32 && H == GR_V2_H && D > 0 && D % 8 == 0, "gr_mix_v2: dims");
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix_v2: upt must be the (H, D / 4, LR, 4) repacked layout");
    const int LR = upt.size(2);
    TORCH_CHECK(LR > 0 && M == LR + (post ? H : 0), "gr_mix_v2: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == H * D && w.numel() == H * D, "gr_mix_v2: dims");
    TORCH_CHECK(dots.dim() == 3 && dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H,
                "gr_mix_v2: dots shape");
    TORCH_CHECK(state.size(0) == R && state.size(1) == LR + H, "gr_mix_v2: state shape");
    TORCH_CHECK(mixed.is_contiguous() && mixed.dim() == 2 && mixed.size(0) == R && mixed.size(1) == D &&
                (mixed.dtype() == at::kHalf || mixed.dtype() == at::kFloat), "gr_mix_v2: mixed shape/dtype");
    TORCH_CHECK(streams.is_cuda() && fn.device() == streams.device() && upt.device() == streams.device() &&
                w.device() == streams.device() && state.device() == streams.device() && dots.device() == streams.device() && mixed.device() == streams.device(),
                "gr_mix_v2: tensors must share a CUDA device");
    float* post_p = nullptr;
    if (post)
    {
        const auto& p = post.value();
        TORCH_CHECK_DTYPE(p, kFloat);
        TORCH_CHECK(p.is_contiguous() && p.dim() == 2 && p.size(0) == R && p.size(1) == H &&
                    p.device() == streams.device(), "gr_mix_v2: post shape/device");
        post_p = (float*) p.data_ptr();
    }
    // Stay inside the default 48 KiB CTA allocation, no device-wide attribute mutation.
    TORCH_CHECK(GR_V2_ROWS * (LR + H) * sizeof(float) + GR_V2_ROWS * GR_V2_QUADS * H * 4 * sizeof(float)
                <= 48 * 1024, "gr_mix_v2: rank exceeds shared-memory limit");
    #define GR_V2_LAUNCH(B) gr_v2_launch<B>(streams, fn, upt, w, (float) rms_eps, dots, state, post_p, mixed, R, D, LR, stream)
    if (R == 1) GR_V2_LAUNCH(1);
    else if (R == 2) GR_V2_LAUNCH(2);
    else if (R <= 4) GR_V2_LAUNCH(4);
    else GR_V2_LAUNCH(GR_V2_ROWS);
    #undef GR_V2_LAUNCH
}

bool gr_mix_v2_fused
(
    const at::Tensor& streams, const at::Tensor& fn, const at::Tensor& upt,
    const at::Tensor& w, double rms_eps, at::Tensor dots, at::Tensor state,
    c10::optional<at::Tensor> post, at::Tensor mixed
)
{
    const at::cuda::OptionalCUDAGuard device_guard(streams.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK_DTYPE(streams, kFloat);
    TORCH_CHECK_DTYPE(fn, kHalf);
    TORCH_CHECK_DTYPE(upt, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(state, kFloat);
    TORCH_CHECK_DTYPE(dots, kFloat);
    TORCH_CHECK(streams.dim() == 3 && fn.dim() == 2 && state.dim() == 2,
                "gr_mix_v2_fused: input/workspace rank");
    TORCH_CHECK(streams.is_contiguous() && fn.is_contiguous() && upt.is_contiguous() &&
                w.is_contiguous() && state.is_contiguous() && dots.is_contiguous(),
                "gr_mix_v2_fused: contiguous inputs required");
    const int R = streams.size(0), H = streams.size(1), D = streams.size(2), M = fn.size(0);
    TORCH_CHECK(R >= 1 && R <= 32 && H == GR_V2_H && D > 0 && D % 8 == 0,
                "gr_mix_v2_fused: dims");
    TORCH_CHECK(upt.dim() == 4 && upt.size(0) == H && upt.size(1) == D / 4 && upt.size(3) == 4,
                "gr_mix_v2_fused: upt must be the (H, D / 4, LR, 4) repacked layout");
    const int LR = upt.size(2);
    TORCH_CHECK(LR > 0 && M == LR + (post ? H : 0),
                "gr_mix_v2_fused: fn rows must be LR (+ H with post)");
    TORCH_CHECK(fn.size(1) == H * D && w.numel() == H * D, "gr_mix_v2_fused: dims");
    TORCH_CHECK(dots.dim() == 3 && dots.size(0) == R && dots.size(1) == M + 1 && dots.size(2) == H,
                "gr_mix_v2_fused: dots shape");
    TORCH_CHECK(state.size(0) == R && state.size(1) == LR + H, "gr_mix_v2_fused: state shape");
    TORCH_CHECK(mixed.is_contiguous() && mixed.dim() == 2 && mixed.size(0) == R && mixed.size(1) == D &&
                (mixed.dtype() == at::kHalf || mixed.dtype() == at::kFloat),
                "gr_mix_v2_fused: mixed shape/dtype");
    TORCH_CHECK(streams.is_cuda() && fn.device() == streams.device() && upt.device() == streams.device() &&
                w.device() == streams.device() && state.device() == streams.device() &&
                dots.device() == streams.device() && mixed.device() == streams.device(),
                "gr_mix_v2_fused: tensors must share a CUDA device");
    float* post_p = nullptr;
    if (post)
    {
        const auto& p = post.value();
        TORCH_CHECK_DTYPE(p, kFloat);
        TORCH_CHECK(p.is_contiguous() && p.dim() == 2 && p.size(0) == R && p.size(1) == H &&
                    p.device() == streams.device(), "gr_mix_v2_fused: post shape/device");
        post_p = (float*) p.data_ptr();
    }
    TORCH_CHECK(GR_V2_ROWS * (LR + H) * sizeof(float) +
                GR_V2_ROWS * GR_V2_QUADS * H * 4 * sizeof(float) <= 48 * 1024,
                "gr_mix_v2_fused: rank exceeds shared-memory limit");
    bool fused_launched = false;
    #define GR_V2_FUSED_LAUNCH(B) \
        fused_launched = gr_v2_launch_fused<B>(streams, fn, upt, w, (float) rms_eps, dots, state, post_p, mixed, R, D, LR, stream)
    if (R == 1) GR_V2_FUSED_LAUNCH(1);
    else if (R == 2) GR_V2_FUSED_LAUNCH(2);
    else if (R <= 4) GR_V2_FUSED_LAUNCH(4);
    else GR_V2_FUSED_LAUNCH(GR_V2_ROWS);
    #undef GR_V2_FUSED_LAUNCH
    return fused_launched;
}
