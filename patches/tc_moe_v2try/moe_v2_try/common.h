#pragma once

#include <ATen/MemoryOverlap.h>
#include <torch/types.h>
#ifdef __CUDACC__
#include <cuda_bf16.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace exl3_try {

inline int64_t checked_add(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 && a <= std::numeric_limits<int64_t>::max() - b,
              "EXL3 workspace size overflow");
  return a + b;
}
inline int64_t checked_mul(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 &&
                  (b == 0 || a <= std::numeric_limits<int64_t>::max() / b),
              "EXL3 workspace size overflow");
  return a * b;
}
inline int64_t workspace_elements(int64_t n) {
  TORCH_CHECK(n <= std::numeric_limits<int64_t>::max() / sizeof(float),
              "EXL3 workspace byte size overflow");
  return n;
}
inline int64_t padded_size(int64_t n) {
  TORCH_CHECK(n > 0 && n <= std::numeric_limits<int>::max() - 127,
              "EXL3 logical dimensions must be positive and fit int32 padding");
  return (n + 127) / 128 * 128;
}
inline void check_padded(int64_t n) {
  TORCH_CHECK(n > 0 && n % 128 == 0 &&
                  n <= std::numeric_limits<int>::max(),
              "EXL3 padded dimensions must be positive H128 multiples fitting int32");
}
inline void check_tensor(const torch::Tensor& t, const torch::Tensor& x,
                         at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == x.device(), name,
              " must be on the input CUDA device");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has incorrect dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
inline void check_integer(const torch::Tensor& t, const torch::Tensor& x,
                          const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
              name, " must have int32 or int64 dtype");
  check_tensor(t, x, t.scalar_type(), name);
}
// The dense linear's out: BF16 (the FP32 result rounded once to nearest) or
// FP32 (the result itself).
inline void check_linear_out(const torch::Tensor& out, const torch::Tensor& x) {
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kFloat,
              "EXL3 linear out must be BF16 or FP32");
  check_tensor(out, x, out.scalar_type(), "out");
}
inline void check_mutations(const torch::Tensor& out,
                            const torch::Tensor& workspace,
                            std::initializer_list<const torch::Tensor*> inputs) {
  at::assert_no_internal_overlap(out);
  at::assert_no_internal_overlap(workspace);
  at::assert_no_overlap(out, workspace);
  for (const auto& input : inputs) {
    at::assert_no_overlap(out, *input);
    at::assert_no_overlap(workspace, *input);
  }
}
inline unsigned int launch_blocks(int64_t tasks) {
  return static_cast<unsigned int>(std::min<int64_t>(tasks, 65535));
}
// The MoE operators' swiglu_limit: finite and nonnegative, 0 for no clamp.
inline float checked_swiglu_limit(double limit) {
  TORCH_CHECK(std::isfinite(limit) && limit >= 0.0,
              "EXL3 swiglu_limit must be finite and nonnegative");
  return static_cast<float>(limit);
}
#ifdef __CUDACC__
// SwiGLU clamp before SiLU(gate) * up: the gate from above, up on both sides
// (limit 0: unclamped).
__device__ __forceinline__ void swiglu_clamp(float& gate, float& up, float limit) {
  if (limit > 0.0f) {
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
  }
}
__device__ __forceinline__ void swiglu_clamp(double& gate, double& up, float limit) {
  if (limit > 0.0f) {
    const double bound = limit;
    gate = fmin(gate, bound);
    up = fmin(fmax(up, -bound), bound);
  }
}
// The dense linear's store of one FP32 result (check_linear_out).
__device__ __forceinline__ void store_output(__nv_bfloat16* out, float value) {
  *out = __float2bfloat16_rn(value);
}
__device__ __forceinline__ void store_output(float* out, float value) { *out = value; }
// Calls launch(p) with p the data of out as __nv_bfloat16* or float*.
template <typename Launch>
void with_linear_out(torch::Tensor& out, Launch&& launch) {
  if (out.scalar_type() == at::kFloat) {
    launch(out.data_ptr<float>());
  } else {
    launch(reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  }
}
#endif

}  // namespace exl3_try
