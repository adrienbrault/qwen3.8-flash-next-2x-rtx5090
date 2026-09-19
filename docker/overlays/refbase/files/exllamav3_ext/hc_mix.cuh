#pragma once

#include <ATen/Tensor.h>

class Graph;

// Fused mHC HyperConnection mix / HyperHead collapse

int hc_mix_num_chunks(int R, int row_len);

void hc_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    int64_t sinkhorn_iters,
    at::Tensor partials,
    at::Tensor post,
    at::Tensor comb,
    at::Tensor collapsed
);

void hc_head
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& base,
    const at::Tensor& scale,
    double rms_eps,
    double hc_eps,
    at::Tensor partials,
    at::Tensor collapsed
);

void hc_apply
(
    at::Tensor x,
    const at::Tensor& y,
    const at::Tensor& post,
    const c10::optional<at::Tensor>& comb
);

void gr_mix
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& upt,
    const at::Tensor& w,
    double rms_eps,
    at::Tensor dots,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);

// R2 exact-order decode mixer: dots (R, M + 1, H), state (R, LR + H), both fp32.
void gr_mix_v2
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& upt,
    const at::Tensor& w,
    double rms_eps,
    at::Tensor dots,
    at::Tensor state,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);

// Decode-only experimental variant. It preserves gr_mix_v2's dots/state arithmetic but
// combines the state and up phases in one cooperative launch. The caller gates this path;
// gr_mix_v2 itself remains the served implementation.
bool gr_mix_v2_fused
(
    const at::Tensor& streams,
    const at::Tensor& fn,
    const at::Tensor& upt,
    const at::Tensor& w,
    double rms_eps,
    at::Tensor dots,
    at::Tensor state,
    c10::optional<at::Tensor> post,
    at::Tensor mixed
);
