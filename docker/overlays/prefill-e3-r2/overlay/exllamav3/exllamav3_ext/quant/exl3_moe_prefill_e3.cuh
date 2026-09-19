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
