#pragma once

// Opt-in sm_120 scheduling/resource variant. The served kernel header is unchanged.
// Arithmetic is copied from V1: do not change K partitions, FOLD, MMA instructions,
// casts, Hadamards or ordered sums here without moving the change to mode 2.
#include "exl3_moe_coop_kernel.cuh"

namespace exl3_moe_coop_v2_ns {
using namespace exl3_moe_coop_ns;

// R=1 uses one MMA row; R>1 retains eight because raw callers may repeat picks.
// Register-dequant K=2/3/4 on sm_120 needs no staging. Other widths retain 8 KiB.
// Host selects this namespace only on sm_120, so host/device layouts agree.
constexpr int SM120_RED_ROW_BYTES = WK * COLS * sizeof(float); // 16*32*4 = 2048
constexpr int SM120_PART_BYTES = WK * 128 * sizeof(float);     // 16*128*4 = 8192
constexpr int SM120_BUSY_SLOTS = 32;
constexpr int SM120_SMALL_WAVES = 2; // two resident waves hide uneven last arrivals
constexpr int SM120_LARGE_WAVES = 1; // loop at occupancy capacity; no thousands of empty CTAs
constexpr int SM120_EXPERIMENTAL_WIDE_B_SLOTS = 128; // R>=ceil(128/topk), 13 at topk=10

template <int bits>
constexpr int stage_bytes_v2() { return bits == 2 || bits == 3 || bits == 4 ? 0 : WK * STAGE_WORDS * 4; }
template <int bits>
int smem_a_v2(int Hi, bool global)
{
    return (global ? 0 : Hi * 2) + (global ? ROWS : 1) * SM120_RED_ROW_BYTES + stage_bytes_v2<bits>();
}
template <int bits>
int smem_b_v2(bool global)
{
    const int gemv = (global ? ROWS : 1) * SM120_RED_ROW_BYTES + stage_bytes_v2<bits>();
    return gemv > SM120_PART_BYTES ? gemv : SM120_PART_BYTES;
}

template <int bits, int cb, bool WIDE>
__device__ __forceinline__ void gemv_tile_v2
(
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    size_t a_stride2,
    const int* __restrict__ rows,
    int nrows,
    int red_rows,
    void* __restrict__ C,
    size_t c_stride,
    bool c_f32,
    int k_begin,                        // k-slice range of this block (split-k across blocks)
    int k_end,
    int ntiles,
    int group,                          // block column group of tile_cols<WIDE>() columns
    float* __restrict__ sh_red,         // [WKK][ROWS][TCOLS]
    uint32_t* __restrict__ sh_stage     // [WK][STAGE_WORDS], staged widths only
)
{
    constexpr bool REG = tile_reg<bits>();
    constexpr int TWORDS = 8 * bits;                                            // uint32 per 16x16 tile
    constexpr int GWORDS = WNT * TWORDS;                                        // uint32 per warp per k-slice
    constexpr int LOADS = REG ? (bits == 2 ? WNT / 2 : WNT) : CEIL_DIVIDE(GWORDS, 32);
    constexpr int LSTRIDE = (REG && bits == 3) ? 24 : 32;
    constexpr int WN = tile_wn<WIDE>();
    constexpr int WKK = tile_wk<WIDE>();
    constexpr int TCOLS = tile_cols<WIDE>();

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int wn = warp % WN;                   // adjacent warps read adjacent column pieces
    const int wk = warp / WN;

    const int kslices = k_end - k_begin;
    const int chunk = CEIL_DIVIDE(kslices, WKK);
    const int ks0 = k_begin + wk * chunk;
    const int myn = max(0, min(chunk, k_end - ks0));
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    // A fragment row of this lane: rows[lane / 4] of the run
    const int r0 = lane >> 2;
    const bool r0_ok = r0 < nrows;
    const size_t a_row0 = r0_ok ? (size_t) rows[r0] * a_stride2 : 0;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + (group * WN + wn) * GWORDS + lane;
    [[maybe_unused]] uint32_t* stage = sh_stage + warp * STAGE_WORDS;

    auto ld_b = [&] (int i, int l) -> uint32_t
    {
        if constexpr (REG && bits == 3)
            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else if constexpr (!REG)
            return (l * 32 + lane < GWORDS) ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };

    uint32_t pf[PF][LOADS];
    #pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < myn)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);

    FragC_h ch[WNT][2] = {};
    float2 acc0[WNT][2] = {};

    for (int ib = 0; ib < myn; ib += PF)
    {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(i + PF, l);
            }

            if constexpr (!REG)
            {
                __syncwarp();
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    if (l * 32 + lane < GWORDS)
                        stage[l * 32 + lane] = bw[l];
                __syncwarp();
            }

            // A fragment: lane covers row lane/4, k pairs (2(lane%4), +1) and (+8, +9)
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            FragB a01, a23;
            a01[0] = r0_ok ? A2[a_row0 + a_col] : hzero;
            a23[0] = r0_ok ? A2[a_row0 + a_col + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (!REG)
                {
                    dq_dispatch<bits, cb>(stage + t * TWORDS, lane << 3, f0, f1);
                }
                else if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                    {
                        acc0[t][f].x += __low2float(ch[t][f][0]);
                        acc0[t][f].y += __high2float(ch[t][f][0]);
                        ch[t][f][0] = hzero;
                    }
            }
        }
    }

    // Cross-warp reduction over the k splits. Lane l holds row l/4, cols
    // tile*16 + frag*8 + 2*(l%4) (+1)
    if (r0_ok)
    {
        const int c0 = 2 * (lane & 3);
        float* red = sh_red + (wk * red_rows + r0) * TCOLS + wn * COLS;
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int col = t * 16 + f * 8 + c0;
                red[col + 0] = acc0[t][f].x;
                red[col + 1] = acc0[t][f].y;
            }
    }
    __syncthreads();

    for (int o = threadIdx.x; o < TCOLS * nrows; o += THREADS)     // up to 128 x 8 outputs per block
    {
        const int r = o / TCOLS;
        const int c = o % TCOLS;
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < WKK; ++j)
            sum += sh_red[(j * red_rows + r) * TCOLS + c];
        const size_t idx = (size_t) rows[r] * c_stride + group * TCOLS + c;
        if (c_f32) ((float*) C)[idx] = sum;
        else       ((half*) C)[idx] = __float2half_rn(sum);
    }
    __syncthreads();
}

template <int bits, int cb, bool WIDE>
__device__ __forceinline__
void tile_a(const MoeCoopParams& p, int item)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    static_assert(CPB == 1, "V2 batches one completion per row");
    const int red_rows = p.a_global ? ROWS : 1;
    extern __shared__ uint32_t smem_dyn[];
    half* sh_A = (half*) smem_dyn;
    const int input_words = p.a_global ? 0 : p.Hi / 2;
    float* sh_red = (float*) (smem_dyn + input_words);
    uint32_t* sh_stage = smem_dyn + input_words + WK * red_rows * COLS;
    __shared__ int sh_last[ROWS];
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int ng = p.I / TCOLS;

    const bool is_gate = p.gated && (item % nproj) == 0;
    const int rem = item / nproj;
    const int ks = rem % p.ksplit_a;                 // split-k index: partial output rows
    const int rem2 = rem / p.ksplit_a;
    const int run_idx = rem2 / ng;
    const int group = rem2 % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res))
    {
        // bsz 1: an inactive slot's blocks (up projection, split 0) write the output chunks of an
        // empty token row, one warp per chunk, strided by the number of gate/up groups
        if (!p.a_global && run_idx < p.bsz * p.topk && !is_gate && ks == 0 && run_idx % p.topk == 0)
        {
            const int row = run_idx / p.topk;
            if (token_active_slots(p, row) == 0)
                for (int oc = group + warp * ng; oc < p.Ho / 128; oc += ng * WK)
                    write_empty_row_chunk(p, row, oc, lane);
        }
        return;
    }
    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.Hi / 16;
    const int k_begin = kslices_all * ks / p.ksplit_a;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_a;

    const half* suh = (const half*) (is_gate ? p.g_suh : p.u_suh)[local];
    const half2* A2;
    size_t a_stride2;
    if (p.a_global)
    {
        A2 = (const half2*) (is_gate ? p.had_g : p.had_u);
        a_stride2 = (size_t) p.Hi / 2;
    }
    else
    {
        // bsz 1: every run is a single slot; rotate its input into shared memory here. A zero
        // row stride makes the tile read it whatever the slot index (which still selects the C row)
        const SlotInfo si = slot_info(p, sh_rows[0]);
        for (int c = warp; c < p.Hi / 128; c += WK)
            rotate_chunk(p, si.row, suh, c, sh_A, lane);
        __syncthreads();
        A2 = (const half2*) sh_A;
        a_stride2 = 0;
    }

    {
        const uint32_t* B32 = (const uint32_t*) (is_gate ? p.g_trellis : p.u_trellis)[local];
        // Partial output rows of split ks live at slot + ks * slots (ksplit * slots <= slots_max)
        const size_t part = (size_t) ks * (p.bsz * p.topk) * p.I * (p.gu_f32 ? 4 : 2);
        void* C = (void*) (((char*) (is_gate ? p.gu_g : p.gu_u)) + part);
        gemv_tile_v2<bits, cb, WIDE>(B32, A2, a_stride2, sh_rows, nrows, red_rows, C, p.I, p.gu_f32, k_begin, k_end, p.I / 16, group, sh_red, sh_stage);
    }

    // Every producer fences all its row stores once. Each row still has its own
    // integer counter; no output arithmetic is performed in arrival order.
    __threadfence();
    __syncthreads();
    const int chunk = group / GPC;
    if (threadIdx.x < nrows)
    {
        const int s = sh_rows[threadIdx.x];
        sh_last[threadIdx.x] = atomicAdd(p.ctr_a + (size_t) s * (p.I / 128) + chunk, 1)
                              == GPC * nproj * p.ksplit_a - 1;
    }
    __syncthreads();
    __threadfence();
    // Independent rows use independent warps; arithmetic within each warp is V1.
    if (warp < nrows && sh_last[warp] && !(p.dbg & 1))
    {
        const int s = sh_rows[warp];
        const int col = chunk * 128 + lane * 4;
        const size_t off = (size_t) s * p.I + col;
        const size_t pstride = (size_t) (p.bsz * p.topk) * p.I;

        // Sum of the split-k partials (fixed order)
        auto load_sum = [&] (const void* buf, float& v0, float& v1, float& v2, float& v3)
        {
            v0 = v1 = v2 = v3 = 0.0f;
            for (int q = 0; q < p.ksplit_a; ++q)
            {
                float t0, t1, t2, t3;
                if (p.gu_f32) load_f4_cg(((const float*) buf) + q * pstride + off, t0, t1, t2, t3);
                else          load_h4_cg(((const half*) buf) + q * pstride + off, t0, t1, t2, t3);
                v0 += t0; v1 += t1; v2 += t2; v3 += t3;
            }
        };
        float u0, u1, u2, u3;
        load_sum(p.gu_u, u0, u1, u2, u3);
        had128(u0, u1, u2, u3, lane);
        scale_h4(((const half*) p.u_svh[local]) + col, u0, u1, u2, u3);
        if (p.u_bias) add_h4(((const half*) p.u_bias[local]) + col, u0, u1, u2, u3);

        float g0 = u0, g1 = u1, g2 = u2, g3 = u3;
        if (p.gated)
        {
            load_sum(p.gu_g, g0, g1, g2, g3);
            had128(g0, g1, g2, g3, lane);
            scale_h4(((const half*) p.g_svh[local]) + col, g0, g1, g2, g3);
            if (p.g_bias) add_h4(((const half*) p.g_bias[local]) + col, g0, g1, g2, g3);
        }

        float a0 = act_gate(p.act, p.gated, g0, u0, p.act_limit);
        float a1 = act_gate(p.act, p.gated, g1, u1, p.act_limit);
        float a2 = act_gate(p.act, p.gated, g2, u2, p.act_limit);
        float a3 = act_gate(p.act, p.gated, g3, u3, p.act_limit);

        scale_h4(((const half*) p.d_suh[local]) + col, a0, a1, a2, a3);
        had128(a0, a1, a2, a3, lane);
        store_h4(p.act_out + off, a0, a1, a2, a3);
    }
    __syncthreads();
}

template <int bits, int cb, bool WIDE>
__device__ __forceinline__
void tile_b(const MoeCoopParams& p, int item)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    static_assert(CPB == 1, "V2 batches one completion per row");
    const int red_rows = p.a_global ? ROWS : 1;
    extern __shared__ uint32_t smem_dyn[];
    float* sh_red = (float*) smem_dyn;
    uint32_t* sh_stage = smem_dyn + WK * red_rows * COLS;
    // gemv_tile_v2 ends with a block barrier; its reduction/staging storage is dead.
    float* sh_part = (float*) smem_dyn;
    __shared__ int sh_last[ROWS];
    __shared__ float sh_gate;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ng = p.Ho / TCOLS;

    const int ks = item % p.ksplit_b;
    const int rem = item / p.ksplit_b;
    const int run_idx = rem / ng;
    const int group = rem % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res)) return;
    const int local = slot_info(p, sh_rows[0]).local;
    const int kslices_all = p.I / 16;
    const int k_begin = kslices_all * ks / p.ksplit_b;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_b;

    {
        const uint32_t* B32 = (const uint32_t*) p.d_trellis[local];
        float* C = p.d_out + (size_t) ks * (p.bsz * p.topk) * p.Ho;
        gemv_tile_v2<bits, cb, WIDE>(B32, (const half2*) p.act_out, (size_t) p.I / 2, sh_rows, nrows, red_rows, C, p.Ho, true,
                                  k_begin, k_end, p.Ho / 16, group, sh_red, sh_stage);
    }

    __threadfence();
    __syncthreads();
    const int chunk = group / GPC;
    if (threadIdx.x < nrows)
    {
        const int row = sh_rows[threadIdx.x] / p.topk;
        const int n = token_active_slots(p, row);
        sh_last[threadIdx.x] = atomicAdd(p.ctr_b + (size_t) row * (p.Ho / 128) + chunk, 1)
                              == n * GPC * p.ksplit_b - 1;
    }
    __syncthreads();
    __threadfence();
    for (int r = 0; r < nrows; ++r)
    {
        if (!sh_last[r] || (p.dbg & 2)) continue;
        const int row = sh_rows[r] / p.topk;
        const int col = chunk * 128 + lane * 4;

        // Shared-expert gate: sigmoid(x_row . w), fp32 across the block
        if (p.sh_gate_w)
        {
            const half* xr = p.x + (size_t) row * p.x_stride;
            float dot = 0.0f;
            for (int i = threadIdx.x * 2; i < p.H; i += THREADS * 2)
            {
                half2 xv = *((const half2*) (xr + i));
                half2 wv = *((const half2*) (p.sh_gate_w + i));
                dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) sh_part[warp] = dot;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                float tt = 0.0f;
                for (int w = 0; w < WK; ++w) tt += sh_part[w];
                sh_gate = 1.0f / (1.0f + __expf(-tt));
            }
            __syncthreads();
        }

        // Each warp takes slots warp, warp + WK, ...; per-warp partials are summed in warp order
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int k = warp; k < p.topk; k += WK)
        {
            const int sk = row * p.topk + k;
            const SlotInfo sj = slot_info(p, sk);
            if (!sj.active) continue;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            for (int q = 0; q < p.ksplit_b; ++q)     // split-k partials, fixed order
            {
                float t0, t1, t2, t3;
                load_f4_cg(p.d_out + ((size_t) q * (p.bsz * p.topk) + sk) * p.Ho + col, t0, t1, t2, t3);
                v0 += t0; v1 += t1; v2 += t2; v3 += t3;
            }
            had128(v0, v1, v2, v3, lane);
            scale_h4(((const half*) p.d_svh[sj.local]) + col, v0, v1, v2, v3);
            if (p.d_bias) add_h4(((const half*) p.d_bias[sj.local]) + col, v0, v1, v2, v3);
            o0 += sj.w * v0;
            o1 += sj.w * v1;
            o2 += sj.w * v2;
            o3 += sj.w * v3;
        }
        float* part = sh_part + warp * 128 + lane * 4;
        part[0] = o0; part[1] = o1; part[2] = o2; part[3] = o3;
        __syncthreads();
        if (warp == 0)
        {
            o0 = 0.0f; o1 = 0.0f; o2 = 0.0f; o3 = 0.0f;
            #pragma unroll
            for (int w = 0; w < WK; ++w)
            {
                const float* q = sh_part + w * 128 + lane * 4;
                o0 += q[0]; o1 += q[1]; o2 += q[2]; o3 += q[3];
            }

            if (p.sh_out)
            {
                const float gv = p.sh_gate_w ? sh_gate : 1.0f;
                const float* sh = p.sh_out + (size_t) row * p.H + col;
                if (col + 0 < p.H) o0 += gv * sh[0];
                if (col + 1 < p.H) o1 += gv * sh[1];
                if (col + 2 < p.H) o2 += gv * sh[2];
                if (col + 3 < p.H) o3 += gv * sh[3];
            }

            float* dst = p.out + (size_t) row * p.out_stride + col;
            if (col + 3 < p.H_out)
                *((float4*) dst) = make_float4(o0, o1, o2, o3);
            else
            {
                if (col + 0 < p.H_out) dst[0] = o0;
                if (col + 1 < p.H_out) dst[1] = o1;
                if (col + 2 < p.H_out) dst[2] = o2;
            }
        }
        __syncthreads();
    }
}


// Ordinary bounded work loops, not a cooperative launch: there is no waiting for
// another CTA. Stream order publishes rot -> A -> B. Last-arrival counters only
// elect an epilogue owner, so any grid size is safe even when oversubscribed.
template <int bits, int cb, bool WIDE>
__global__ __launch_bounds__(THREADS)
void exl3_moe_coop_a_kernel(const MoeCoopParams p)
{
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS)
        p.ctr_b[i] = 0;
    const int runs = p.a_global ? p.runs[0] : p.bsz * p.topk;
    const int count = runs * (p.I / tile_cols<WIDE>()) * p.ksplit_a * (p.gated ? 2 : 1);
    for (int item = blockIdx.x; item < count; item += gridDim.x)
    {
        tile_a<bits, cb, WIDE>(p, item);
        __syncthreads(); // reuse this CTA's shared rows/flags only after all warps finish
    }
}

template <int bits, int cb, bool WIDE>
__global__ __launch_bounds__(THREADS)
void exl3_moe_coop_b_kernel(const MoeCoopParams p)
{
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * THREADS)
        p.ctr_a[i] = 0;
    const int runs = p.a_global ? p.runs[0] : p.bsz * p.topk;
    const int count = runs * (p.Ho / tile_cols<WIDE>()) * p.ksplit_b;
    for (int item = blockIdx.x; item < count; item += gridDim.x)
    {
        tile_b<bits, cb, WIDE>(p, item);
        __syncthreads();
    }
}
} // namespace exl3_moe_coop_v2_ns
