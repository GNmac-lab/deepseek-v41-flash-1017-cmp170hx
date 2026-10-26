#pragma once
// EXL3 routed-expert MoE for prefill-sized calls, second schedule (moe_prefill_try.cu): the arguments,
// output and arithmetic of moe_out_prefill_v2 (moe_prefill_v2.h); calls of up to 2048 rows run as one
// launch sequence, so the workspace grows to 2048 rows. Registered into _exl3_C by its own library fragment.

#include <torch/types.h>

#include <cstdint>

namespace exl3 {

// FP32 workspace elements one call of up to max_rows rows needs.
int64_t moe_workspace_size_prefill_try(int64_t max_rows, int64_t hidden_size,
                                       int64_t intermediate_size, int64_t topk,
                                       int64_t local_experts);

void moe_out_prefill_try(const torch::Tensor& x, const torch::Tensor& topk_ids,
                         const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
                         const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
                         int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
                         torch::Tensor& workspace, double swiglu_limit);

}  // namespace exl3
