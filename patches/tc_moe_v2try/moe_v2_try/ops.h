#pragma once

#include <torch/types.h>
#include <cstdint>

namespace exl3_try {

// FP32 workspace elements moe_out / moe_out_fp32 need for up to max_rows rows.
int64_t moe_workspace_size(int64_t max_rows, int64_t hidden_size,
                           int64_t intermediate_size, int64_t topk,
                           int64_t local_experts);

// out [rows, hidden] BF16 = this rank's routed experts, combined; every routed
// matrix of the layer is stored with bits_x2 and codebook 2 (mul1);
// swiglu_limit > 0 clamps the gate from above and up on both sides before
// SiLU(gate) * up.
void moe_out(const torch::Tensor& x, const torch::Tensor& topk_ids,
             const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
             const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
             int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
             torch::Tensor& workspace, double swiglu_limit);

// The same computation; out [rows, hidden] FP32 = the combine sums whose
// round-to-nearest BF16 is moe_out's output.
void moe_out_fp32(const torch::Tensor& x, const torch::Tensor& topk_ids,
                  const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
                  const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
                  int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
                  torch::Tensor& workspace, double swiglu_limit);

}  // namespace exl3_try
