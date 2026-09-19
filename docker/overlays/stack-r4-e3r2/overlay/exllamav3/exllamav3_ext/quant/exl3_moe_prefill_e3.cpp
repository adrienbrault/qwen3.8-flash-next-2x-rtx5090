#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime_api.h>

#include "exl3_moe_prefill_e3.cuh"

namespace {

void check_cuda_contiguous(const at::Tensor& t, at::ScalarType dtype, const char* name)
{
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
}

void check_ptrs(const at::Tensor& t, int64_t experts, const char* name)
{
    check_cuda_contiguous(t, at::kLong, name);
    TORCH_CHECK(t.dim() == 1 && t.numel() == experts, name, " must be int64[num_experts]");
}

void check_i32_vector(const at::Tensor& t, const char* name)
{
    check_cuda_contiguous(t, at::kInt, name);
    TORCH_CHECK(t.dim() == 1, name, " must be a vector");
}

}  // namespace

int64_t exl3_moe_prefill_e3_tile_rows() { return 64; }

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
    double act_limit)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    int current_device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&current_device));
    TORCH_CHECK(current_device == x.get_device(), "E3 current-device guard failed");

    check_cuda_contiguous(x, at::kHalf, "x");
    check_cuda_contiguous(out, at::kFloat, "out");
    check_cuda_contiguous(expert_count, at::kLong, "expert_count");
    check_cuda_contiguous(token_sorted, at::kLong, "token_sorted");
    check_cuda_contiguous(weight_sorted, at::kHalf, "weight_sorted");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "x/out must be matrices");
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == x.size(1), "out must match x");
    TORCH_CHECK(x.size(1) == 2560, "E3 r2 is pinned to hidden_size=2560");
    TORCH_CHECK(expert_count.dim() == 1 && expert_count.numel() == 513,
                "E3 r2 is pinned to 512 experts plus sentinel");
    TORCH_CHECK(token_sorted.dim() == 1 && weight_sorted.sizes() == token_sorted.sizes(),
                "sorted token/weight vectors must match");
    TORCH_CHECK(token_sorted.numel() == x.size(0) * 10,
                "E3 r2 is pinned to top_k=10");
    auto check_bits = [](int64_t bits, const char* name)
    {
        TORCH_CHECK(bits >= 2 && bits <= 4, name, " K must be 2, 3, or 4");
    };
    check_bits(gate_bits, "gate");
    check_bits(up_bits, "up");
    check_bits(down_bits, "down");
    TORCH_CHECK(thin_rows >= 1 && thin_rows <= 256, "thin_rows must be in [1, 256]");

    constexpr int64_t experts = 512;
    check_ptrs(gate_trellis, experts, "gate_trellis");
    check_ptrs(gate_suh, experts, "gate_suh");
    check_ptrs(gate_svh, experts, "gate_svh");
    check_ptrs(up_trellis, experts, "up_trellis");
    check_ptrs(up_suh, experts, "up_suh");
    check_ptrs(up_svh, experts, "up_svh");
    check_ptrs(down_trellis, experts, "down_trellis");
    check_ptrs(down_suh, experts, "down_suh");
    check_ptrs(down_svh, experts, "down_svh");

    const int64_t cap = token_sorted.numel();
    auto check_half_matrix = [cap](const at::Tensor& t, int64_t width, const char* name)
    {
        check_cuda_contiguous(t, at::kHalf, name);
        TORCH_CHECK(t.dim() == 2 && t.size(0) == cap && t.size(1) == width,
                    name, " has the wrong workspace shape");
    };
    check_half_matrix(h13_gate, 2560, "h13_gate");
    check_half_matrix(h13_up, 2560, "h13_up");
    check_half_matrix(h2, 640, "h2");
    check_cuda_contiguous(row_token, at::kLong, "row_token");
    check_cuda_contiguous(row_weight, at::kHalf, "row_weight");
    check_cuda_contiguous(row_expert, at::kInt, "row_expert");
    TORCH_CHECK(row_token.numel() == cap && row_weight.numel() == cap && row_expert.numel() == cap,
                "row workspaces must have assignment capacity");
    check_i32_vector(seg_expert, "seg_expert");
    check_i32_vector(seg_row0, "seg_row0");
    check_i32_vector(seg_rows, "seg_rows");
    TORCH_CHECK(seg_expert.numel() == seg_row0.numel() && seg_row0.numel() == seg_rows.numel(),
                "segment workspaces must match");
    TORCH_CHECK(seg_expert.numel() >= (cap + 63) / 64 + experts,
                "segment workspace is smaller than the no-sync upper bound");
    check_i32_vector(num_rows, "num_rows");
    check_i32_vector(num_segs, "num_segs");
    TORCH_CHECK(num_rows.numel() == 1 && num_segs.numel() == 1, "live counts must have one element");

    exl3_moe_prefill_e3_cuda(
        x, out, expert_count, token_sorted, weight_sorted,
        gate_trellis, gate_suh, gate_svh,
        up_trellis, up_suh, up_svh,
        down_trellis, down_suh, down_svh,
        h13_gate, h13_up, h2,
        row_token, row_weight, row_expert,
        seg_expert, seg_row0, seg_rows, num_rows, num_segs,
        static_cast<int>(gate_bits), static_cast<int>(up_bits), static_cast<int>(down_bits),
        static_cast<int>(thin_rows), static_cast<float>(act_limit));
}
