#pragma once

/*
 * EXL3 codec adapted from turboderp-org/exllamav3, revision
 * d3739fd393337b1ff4d6c2a342b12f0c87a9592f:
 * quant/{pack.cu,frac.cu,codebook.cuh,exl3_dq.cuh,hadamard_inner.cuh}
 * and modules/quant/exl3_lib/quantize.py.
 * Modifications: per-lane eight-state ring window decoder, inverse lane
 * permutation, BF16 operator integration, and FP32 normalized H128
 * arithmetic.
 *
 * MIT License
 *
 * Copyright (c) 2025 Turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace exl3 {

constexpr float kH128Scale = 0.08838834764831845f;

// Every supported (bits_x2, codebook) pair, as one dispatch key.
constexpr int format_key(int bits_x2, int codebook) { return bits_x2 * 4 + codebook; }

// One 16x16 tile is a ring of 256 * bits_x2 / 2 bits held in 4 * bits_x2
// MSB-first uint32 words. Encoder lane L (0..31) owns ring states 8L..8L+7;
// state i is the 16 bits ending at end(i) = (i + 1) * whole + [odd] (i + 1) / 2.
// A lane's eight states lie in one window that starts at end(8L) - 16, so a
// lane loads a few consecutive words once and every state offset inside the
// window is a compile-time constant.
template <int kBitsX2>
struct TileRing {
  static constexpr int kWhole = kBitsX2 / 2;
  static constexpr bool kOdd = (kBitsX2 & 1) != 0;
  static constexpr int kRingBits = 128 * kBitsX2;
  static constexpr int kWords = 4 * kBitsX2;
  static constexpr int kTileHalfwords = 8 * kBitsX2;
  // end(8L + j) - end(8L); 8L is even, so the odd-rate term is (j + 1) / 2.
  static constexpr int rel(int j) { return j * kWhole + (kOdd ? (j + 1) / 2 : 0); }
  static constexpr int kWindowBits = rel(7) + 16;
  static constexpr int kWindowWords = (kWindowBits + 31) / 32;
  // Words to load for the worst in-word start offset of 31 bits.
  static constexpr int kLoadWords = (31 + kWindowBits + 31) / 32;
  static_assert(kLoadWords <= kWords, "window exceeds the ring");
};

// Lane-constant part of the window: first word (relative to the tile) and bit
// offset inside it. Wrapping uses compare-and-subtract, not integer modulo.
template <int kBitsX2>
struct LaneWindow {
  using Ring = TileRing<kBitsX2>;
  int word[Ring::kLoadWords];
  int shift;

  __device__ __forceinline__ explicit LaneWindow(int lane) {
    int start = (8 * lane + 1) * Ring::kWhole + (Ring::kOdd ? 4 * lane : 0) - 16;
    if (start < 0) start += Ring::kRingBits;
    shift = start & 31;
    int w = start >> 5;
#pragma unroll
    for (int i = 0; i < Ring::kLoadWords; ++i) {
      word[i] = w;
      w = w + 1 == Ring::kWords ? 0 : w + 1;
    }
  }
};

// pack_trellis swaps 16-bit pairs, leaving MSB-first uint32 ring words made
// of halfwords 2w (low) and 2w + 1 (high). A tile starts at a multiple of
// 16 * bits_x2 bytes, so 32-bit loads are aligned whenever the base is.
__device__ __forceinline__ uint32_t ring_word(const uint16_t* tile, int w, bool aligned) {
  if (aligned) return __ldg(reinterpret_cast<const uint32_t*>(tile) + w);
  return uint32_t(__ldg(tile + 2 * w)) | (uint32_t(__ldg(tile + 2 * w + 1)) << 16);
}

template <int kBitsX2>
__device__ __forceinline__ void load_lane_words(
    const uint16_t* tile, const LaneWindow<kBitsX2>& window, bool aligned,
    uint32_t (&words)[TileRing<kBitsX2>::kLoadWords]) {
#pragma unroll
  for (int i = 0; i < TileRing<kBitsX2>::kLoadWords; ++i) {
    words[i] = ring_word(tile, window.word[i], aligned);
  }
}

// The 16-bit state at compile-time bit offset kOffset of the realigned window.
template <int kOffset, int kWindowWords>
__device__ __forceinline__ uint32_t window_state(const uint32_t (&u)[kWindowWords]) {
  constexpr int q = kOffset >> 5;
  constexpr int r = kOffset & 31;
  if constexpr (r <= 16) {
    return (u[q] >> (16 - r)) & 0xffffu;
  } else {
    return __funnelshift_l(u[q + 1], u[q], r) >> 16;
  }
}

// Two codebook values at once. Each lane computes exactly the scalar
// codebook arithmetic of the format: FP16 add (mcg) or FP16 fused
// multiply-add (mul1) with round-to-nearest, then widening to FP32.
template <int kCodebook>
__device__ __forceinline__ float2 codebook_pair(uint32_t s0, uint32_t s1) {
  if constexpr (kCodebook == 1) {
    const uint32_t m0 = ((s0 * 0xCBAC1FEDu) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint32_t m1 = ((s1 * 0xCBAC1FEDu) & 0x8fff8fffu) ^ 0x3b603b60u;
    const uint32_t lo = __byte_perm(m0, m1, 0x5410);
    const uint32_t hi = __byte_perm(m0, m1, 0x7632);
    return __half22float2(__hadd2(*reinterpret_cast<const __half2*>(&lo),
                                  *reinterpret_cast<const __half2*>(&hi)));
  } else {
    // Sum of the four product bytes, 0..1020, placed in the FP16 mantissa
    // of 1024.0 (0x6400): exactly 1024 + sum.
    const uint32_t sum0 = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0u);
    const uint32_t sum1 = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0u);
    const uint32_t packed = (sum0 | (sum1 << 16)) + 0x64006400u;
    const uint32_t scale = 0x1eee1eeeu;
    const uint32_t bias = 0xc931c931u;
    return __half22float2(__hfma2(*reinterpret_cast<const __half2*>(&packed),
                                  *reinterpret_cast<const __half2*>(&scale),
                                  *reinterpret_cast<const __half2*>(&bias)));
  }
}

// Decodes encoder lane `lane`'s eight states. Value j sits at tile row
// 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1) and tile column
// (lane >> 2) + 8 * (j >> 2): the inverse of the tensor-core lane permutation.
template <int kBitsX2, int kCodebook>
__device__ __forceinline__ void decode_lane(
    const uint32_t (&words)[TileRing<kBitsX2>::kLoadWords], int shift, float (&value)[8]) {
  using Ring = TileRing<kBitsX2>;
  uint32_t u[Ring::kWindowWords];
#pragma unroll
  for (int i = 0; i < Ring::kWindowWords; ++i) {
    const uint32_t next = i + 1 < Ring::kLoadWords ? words[i + 1] : 0u;
    u[i] = __funnelshift_l(next, words[i], shift);
  }
  uint32_t state[8];
  state[0] = window_state<Ring::rel(0), Ring::kWindowWords>(u);
  state[1] = window_state<Ring::rel(1), Ring::kWindowWords>(u);
  state[2] = window_state<Ring::rel(2), Ring::kWindowWords>(u);
  state[3] = window_state<Ring::rel(3), Ring::kWindowWords>(u);
  state[4] = window_state<Ring::rel(4), Ring::kWindowWords>(u);
  state[5] = window_state<Ring::rel(5), Ring::kWindowWords>(u);
  state[6] = window_state<Ring::rel(6), Ring::kWindowWords>(u);
  state[7] = window_state<Ring::rel(7), Ring::kWindowWords>(u);
#pragma unroll
  for (int j = 0; j < 8; j += 2) {
    const float2 pair = codebook_pair<kCodebook>(state[j], state[j + 1]);
    value[j] = pair.x;
    value[j + 1] = pair.y;
  }
}

__device__ __forceinline__ int lane_tile_row(int lane, int j) {
  return 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);
}
__device__ __forceinline__ int lane_tile_col(int lane, int j) {
  return (lane >> 2) + 8 * (j >> 2);
}

// H128 over one 128-value block held by a warp: lane holds elements
// lane + 32 * i. Standard symmetric Sylvester ordering, distances 1..64 in
// increasing order, each element computing exactly value + other (lower) or
// other - value (upper), then the 1/sqrt(128) scale.
__device__ __forceinline__ void hadamard128_warp(float (&v)[4], int lane) {
#pragma unroll
  for (int distance = 1; distance < 32; distance <<= 1) {
    const bool upper = (lane & distance) != 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float other = __shfl_xor_sync(0xffffffffu, v[i], distance);
      v[i] = upper ? __fsub_rn(other, v[i]) : __fadd_rn(v[i], other);
    }
  }
#pragma unroll
  for (int distance = 1; distance < 4; distance <<= 1) {
    float next[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float other = v[i ^ distance];
      next[i] = (i & distance) ? __fsub_rn(other, v[i]) : __fadd_rn(v[i], other);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = next[i];
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], kH128Scale);
}

}  // namespace exl3
