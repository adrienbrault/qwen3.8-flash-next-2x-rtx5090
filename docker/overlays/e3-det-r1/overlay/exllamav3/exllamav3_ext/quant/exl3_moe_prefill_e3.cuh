#pragma once

#include <torch/extension.h>

// K in {2,3,4} / mul1 grouped routed-MoE prefill. Gate, up, and down K are
// dispatched independently for mixed-width layers. This remains pinned to the
// served Qwen4-Exp geometry; Python keeps every other shape on the served path.
void exl3_moe_prefill_e3(
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
    int64_t gate_bits,
    int64_t up_bits,
    int64_t down_bits,
    int64_t thin_rows,
    double act_limit);

int64_t exl3_moe_prefill_e3_tile_rows();

// CUDA implementation called only after the C++ wrapper has validated every
// tensor and selected the current PyTorch stream.
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
    float act_limit);

// Deterministic E3 (EXL3_MOE_PREFILL_E3_DET=1, e3-det r1). Phase 1 runs metadata, gather and
// gate/up exactly as exl3_moe_prefill_e3, fills row_pos / expert_start / inv_order from the
// counts and the sort permutation `order`, then stores every fat assignment's half-rounded,
// pre-Hadamard down output into `slots` (fp32 [assignments, 2560], row = expert-sorted
// position, may alias h13_gate|h13_up) instead of adding into the output. The caller then runs
// the thin tier through exl3_moe's slot mode (fused_base = expert_start) into the same slots and
// calls phase 2, which sums each token's 10 slots in router top-k order and stores the result.
// No float atomics.
void exl3_moe_prefill_e3_det(
    const at::Tensor& x,
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
    const at::Tensor& h13_gate,
    const at::Tensor& h13_up,
    const at::Tensor& h2,
    const at::Tensor& row_token,
    const at::Tensor& row_weight,
    const at::Tensor& row_expert,
    const at::Tensor& row_pos,
    const at::Tensor& order,
    const at::Tensor& expert_start,
    const at::Tensor& inv_order,
    const at::Tensor& seg_expert,
    const at::Tensor& seg_row0,
    const at::Tensor& seg_rows,
    const at::Tensor& num_rows,
    const at::Tensor& num_segs,
    const at::Tensor& slots,
    int64_t gate_bits,
    int64_t up_bits,
    int64_t down_bits,
    int64_t thin_rows,
    double act_limit);

void exl3_moe_prefill_e3_det_reduce(
    const at::Tensor& out,
    const at::Tensor& slots,
    const at::Tensor& flat_expert,
    const at::Tensor& inv_order,
    const at::Tensor& flat_weight,
    const at::Tensor& expert_count,
    const at::Tensor& down_svh,
    int64_t thin_rows);

void exl3_moe_prefill_e3_det_cuda(
    const at::Tensor& x,
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
    const at::Tensor& h13_gate,
    const at::Tensor& h13_up,
    const at::Tensor& h2,
    const at::Tensor& row_token,
    const at::Tensor& row_weight,
    const at::Tensor& row_expert,
    const at::Tensor& row_pos,
    const at::Tensor& order,
    const at::Tensor& expert_start,
    const at::Tensor& inv_order,
    const at::Tensor& seg_expert,
    const at::Tensor& seg_row0,
    const at::Tensor& seg_rows,
    const at::Tensor& num_rows,
    const at::Tensor& num_segs,
    const at::Tensor& slots,
    int gate_bits,
    int up_bits,
    int down_bits,
    int thin_rows,
    float act_limit);

void exl3_moe_prefill_e3_det_reduce_cuda(
    const at::Tensor& out,
    const at::Tensor& slots,
    const at::Tensor& flat_expert,
    const at::Tensor& inv_order,
    const at::Tensor& flat_weight,
    const at::Tensor& expert_count,
    const at::Tensor& down_svh,
    int thin_rows);
