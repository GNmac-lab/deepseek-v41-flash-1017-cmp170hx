#pragma once

// Tensor-core forms of the EXL3 codec for the v2 MoE kernels (moe_v2.cu): a
// lane's decoded tile values as an m16n8k16 A fragment, and the H128
// transform in FP64. Built on codec.cuh, which this header leaves unchanged.

#include "codec.cuh"

namespace exl3_try {
namespace moe_v2 {

// Two mul1 codebook values at once, as an FP16 pair (low half = s0): the byte
// sum of the hash placed in the FP16 mantissa of 1024.0 (0x6400, the dp4a
// accumulator; exactly 1024 + sum < 0x10000, so one byte permute packs the
// pair), then the format's FP16 fused multiply-add with round-to-nearest.
// The same values as codebook_pair<2> before its widening to FP32.
__device__ __forceinline__ uint32_t mul1_half2(uint32_t s0, uint32_t s1) {
  const uint32_t sum0 = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
  const uint32_t sum1 = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
  const uint32_t packed = __byte_perm(sum0, sum1, 0x5410);
  const uint32_t scale = 0x1eee1eeeu;
  const uint32_t bias = 0xc931c931u;
  const __half2 value = __hfma2(*reinterpret_cast<const __half2*>(&packed),
                                *reinterpret_cast<const __half2*>(&scale),
                                *reinterpret_cast<const __half2*>(&bias));
  return *reinterpret_cast<const uint32_t*>(&value);
}

// Encoder lane `lane`'s eight 16-bit states from its loaded window words.
template <int kBitsX2>
__device__ __forceinline__ void lane_states(
    const uint32_t (&words)[TileRing<kBitsX2>::kLoadWords], int shift, uint32_t (&state)[8]) {
  using Ring = TileRing<kBitsX2>;
  uint32_t u[Ring::kWindowWords];
#pragma unroll
  for (int i = 0; i < Ring::kWindowWords; ++i) {
    const uint32_t next = i + 1 < Ring::kLoadWords ? words[i + 1] : 0u;
    u[i] = __funnelshift_l(next, words[i], shift);
  }
  state[0] = window_state<Ring::rel(0), Ring::kWindowWords>(u);
  state[1] = window_state<Ring::rel(1), Ring::kWindowWords>(u);
  state[2] = window_state<Ring::rel(2), Ring::kWindowWords>(u);
  state[3] = window_state<Ring::rel(3), Ring::kWindowWords>(u);
  state[4] = window_state<Ring::rel(4), Ring::kWindowWords>(u);
  state[5] = window_state<Ring::rel(5), Ring::kWindowWords>(u);
  state[6] = window_state<Ring::rel(6), Ring::kWindowWords>(u);
  state[7] = window_state<Ring::rel(7), Ring::kWindowWords>(u);
}

// A lane's eight mul1 values as the m16n8k16 A fragment of W^T (fragment
// row = tile column = output column, fragment column = tile row = k):
// {a0a1, a2a3, a4a5, a6a7} = value pairs (0,1), (4,5), (2,3), (6,7) of
// decode_lane's order.
template <int kBitsX2>
__device__ __forceinline__ void decode_lane_fragment_mul1(
    const uint32_t (&words)[TileRing<kBitsX2>::kLoadWords], int shift, uint32_t (&a)[4]) {
  uint32_t state[8];
  lane_states<kBitsX2>(words, shift, state);
  a[0] = mul1_half2(state[0], state[1]);
  a[1] = mul1_half2(state[4], state[5]);
  a[2] = mul1_half2(state[2], state[3]);
  a[3] = mul1_half2(state[6], state[7]);
}

// hadamard128_warp in FP64 (scale 1/sqrt(128) rounded to FP64).
__device__ __forceinline__ void hadamard128_warp_f64(double (&v)[4], int lane) {
#pragma unroll
  for (int distance = 1; distance < 32; distance <<= 1) {
    const bool upper = (lane & distance) != 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const double other = __shfl_xor_sync(0xffffffffu, v[i], distance);
      v[i] = upper ? __dsub_rn(other, v[i]) : __dadd_rn(v[i], other);
    }
  }
#pragma unroll
  for (int distance = 1; distance < 4; distance <<= 1) {
    double next[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const double other = v[i ^ distance];
      next[i] = (i & distance) ? __dsub_rn(other, v[i]) : __dadd_rn(v[i], other);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = next[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __dmul_rn(v[i], 0.08838834764831844055);
}

}  // namespace moe_v2
}  // namespace exl3_try
