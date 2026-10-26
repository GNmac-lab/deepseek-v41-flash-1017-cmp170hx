// dsv41 port: Morrowmake/vllm-cmp170hx@a0c57b52376efd67c57735bc949d4bafa22ae29e csrc/exl3/moe_v2_try.cu
// (Apache-2.0). Changes: LaneTiling for hidden 5120 with intermediate 640 / 512 (TP expert chunks, the old
// moe_out_v2 slice layout); registered as its own library _dsv41_mm_v2try.
// moe_out_v2_try (VLLM_EXL3_MOE_V2_RING): a faster moe_out_v2 for the target model's
// expert-parallel routed experts, beside moe_out_v2.
// Same arguments, workspace contract and output bits as moe_out_v2 of
// csrc/exl3/moe_v2.cu at 7b31d6b252 (bitwise equal on the FP64-checked cases),
// faster lane GEMMs:
//   - lane GEMM trellis words through a per-warp cp.async shared ring where a
//     k step holds >= 384 B, with fixed per-lane LDS bases;
//   - column groups of eight n16 tiles, ring words loaded one tile ahead;
//   - activation partial loads in 8-slice chunks, one input block per
//     (assignment, projection);
//   - only the window words the lanes' start offsets need (codec.cuh).
// Its own namespace (exl3_try) and header copies (moe_v2_try/), so every
// other kernel of the library, moe_out_v2 and moe_out_v2_fold included, is
// compiled exactly as without this file. The caller
// (vllm/sm80/kernels/exl3/ops.py moe_v2_try_open) takes it only for hidden
// 5120, intermediate 2304, top-6, 96 of 384 experts local, bits_x2 6 or 4,
// 1..64 rows.
//
// Base: moe_v2.cu of 7b31d6b252, whose hidden-5120 shape follows Curtis
// White's port (github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE at
// 88eb6e1, Apache-2.0); see plugins/sm80/licenses/exl3/NOTICE.
#include "moe_v2_try/ops.h"
#include "moe_v2_try/common.h"
#include "moe_v2_try/codec_moe_v2.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <type_traits>
#include <utility>

namespace exl3_try {
namespace moe_v2 {
namespace {

// The route + gate/up operand launch (block 0 routes with these threads; the
// other blocks run one warp per H128 block of a row).
constexpr int kRouteInputThreads = 640;
// Route kernel shared memory: three int arrays per local expert.
constexpr int64_t kMaxRouteExperts = 4096;

// Integers the route kernel writes into the workspace each call:
//   [0] local assignment count, [1] row-tile count, [2..3] unused,
//   sorted[assignments]: local assignment ids grouped by local expert,
//   tiles[assignments] (16-byte aligned): {expert, first sorted index, rows, 0},
//   experts[assignments]: the local expert of each sorted entry,
//   by_assignment[assignments]: the local expert of each assignment (-1: none),
//   tile_trellis[assignments][3] (int64), tile_columns[assignments][3]: per row
//   tile and projection the trellis address and padded_out / 16.
struct RouteLayout {
  static constexpr int64_t kHeader = 4;
  __host__ __device__ static constexpr int64_t sorted_offset() { return kHeader; }
  __host__ __device__ static constexpr int64_t tiles_offset(int64_t assignments) {
    return kHeader + (assignments + 3) / 4 * 4;
  }
  __host__ __device__ static constexpr int64_t experts_offset(int64_t assignments) {
    return tiles_offset(assignments) + 4 * assignments;
  }
  __host__ __device__ static constexpr int64_t by_assignment_offset(int64_t assignments) {
    return experts_offset(assignments) + assignments;
  }
  // Per row tile and projection: the expert's trellis address (int64, 8-byte
  // aligned) and its tile columns (padded_out / 16).
  __host__ __device__ static constexpr int64_t tile_trellis_offset(int64_t assignments) {
    return (by_assignment_offset(assignments) + assignments + 1) / 2 * 2;
  }
  __host__ __device__ static constexpr int64_t tile_columns_offset(int64_t assignments) {
    return tile_trellis_offset(assignments) + 2 * 3 * assignments;
  }
  __host__ __device__ static constexpr int64_t ints(int64_t assignments) {
    return tile_columns_offset(assignments) + 3 * assignments;
  }
};

struct Routing {
  const void* ids;
  const void* expert_map;
  const int64_t* pointers;
  const int64_t* metadata;
  int64_t global_experts;
  int64_t local_experts;
  int64_t topk;
  bool ids64;
  bool map64;
};

__device__ __forceinline__ int64_t integer_at(const void* ptr, bool wide, int64_t i) {
  return wide ? static_cast<const int64_t*>(ptr)[i]
              : static_cast<const int32_t*>(ptr)[i];
}
__device__ __forceinline__ int64_t local_expert(const Routing& routing, int64_t assignment) {
  const int64_t global = integer_at(routing.ids, routing.ids64, assignment);
  if (global < 0 || global >= routing.global_experts) return -1;
  const int64_t local = integer_at(routing.expert_map, routing.map64, global);
  return local >= 0 && local < routing.local_experts ? local : -1;
}
// [trellis, suh, svh] of one projection of one local expert.
__device__ __forceinline__ const int64_t* expert_pointers(const Routing& routing,
                                                          int64_t expert, int projection) {
  return routing.pointers + expert * 9 + projection * 3;
}
__device__ __forceinline__ const half* expert_suh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[1]);
}
__device__ __forceinline__ const half* expert_svh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[2]);
}

// Exclusive block-wide scan of two counters; also returns the totals.
__device__ __forceinline__ void block_scan2(int a, int b, int& a_before, int& b_before,
                                            int& a_total, int& b_total) {
  __shared__ int warp_a[32];
  __shared__ int warp_b[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  int x = a;
  int y = b;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int xu = __shfl_up_sync(0xffffffffu, x, d);
    const int yu = __shfl_up_sync(0xffffffffu, y, d);
    if (lane >= d) {
      x += xu;
      y += yu;
    }
  }
  if (lane == 31) {
    warp_a[warp] = x;
    warp_b[warp] = y;
  }
  __syncthreads();
  if (warp == 0) {
    int wx = lane < warps ? warp_a[lane] : 0;
    int wy = lane < warps ? warp_b[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int xu = __shfl_up_sync(0xffffffffu, wx, d);
      const int yu = __shfl_up_sync(0xffffffffu, wy, d);
      if (lane >= d) {
        wx += xu;
        wy += yu;
      }
    }
    warp_a[lane] = wx;
    warp_b[lane] = wy;
  }
  __syncthreads();
  a_before = x - a + (warp > 0 ? warp_a[warp - 1] : 0);
  b_before = y - b + (warp > 0 ? warp_b[warp - 1] : 0);
  a_total = warp_a[warps - 1];
  b_total = warp_b[warps - 1];
}

// Groups the local assignments by local expert into tiles of up to
// block_rows rows (one block, every call), records each sorted entry's
// expert and each assignment's expert, and each row tile's trellis address
// and tile columns per projection. Order inside an expert is arbitrary: each
// assignment's result depends only on its own row, never on its position or
// neighbours. A matrix not stored as bits_x2 kBitsX2 / codebook 2 traps.
template <int kBitsX2>
__device__ __forceinline__ void route_block(const Routing& routing, int64_t assignments,
                                            int block_rows, int* table) {
  extern __shared__ int route_shared[];
  const int experts = static_cast<int>(routing.local_experts);
  int* cursor = route_shared;              // counts, then fill cursors
  int* row_begin = cursor + experts;
  int* tile_begin = row_begin + experts;   // experts + 1 entries
  for (int e = threadIdx.x; e < experts; e += blockDim.x) cursor[e] = 0;
  __syncthreads();
  int* by_assignment = table + RouteLayout::by_assignment_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    by_assignment[a] = static_cast<int>(e);
    if (e >= 0) atomicAdd(&cursor[e], 1);
  }
  __syncthreads();
  const int per = (experts + blockDim.x - 1) / blockDim.x;
  const int e0 = min(experts, static_cast<int>(threadIdx.x) * per);
  const int e1 = min(experts, e0 + per);
  int rows = 0;
  int tiles = 0;
  for (int e = e0; e < e1; ++e) {
    rows += cursor[e];
    tiles += (cursor[e] + block_rows - 1) / block_rows;
  }
  int row_at, tile_at, row_total, tile_total;
  block_scan2(rows, tiles, row_at, tile_at, row_total, tile_total);
  for (int e = e0; e < e1; ++e) {
    const int count = cursor[e];
    row_begin[e] = row_at;
    tile_begin[e] = tile_at;
    cursor[e] = row_at;
    row_at += count;
    tile_at += (count + block_rows - 1) / block_rows;
  }
  if (threadIdx.x == 0) {
    tile_begin[experts] = tile_total;
    table[0] = row_total;
    table[1] = tile_total;
  }
  __syncthreads();
  int4* tile_table = reinterpret_cast<int4*>(table + RouteLayout::tiles_offset(assignments));
  for (int t = threadIdx.x; t < tile_total; t += blockDim.x) {
    // Last expert whose first tile is at or before t (empty experts share
    // their successor's start, so the search lands on the owner).
    int lo = 0;
    int hi = experts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (tile_begin[mid] <= t) lo = mid; else hi = mid;
    }
    const int first = row_begin[lo] + (t - tile_begin[lo]) * block_rows;
    const int end = lo + 1 < experts ? row_begin[lo + 1] : row_total;
    tile_table[t] = make_int4(lo, first, min(block_rows, end - first), 0);
    int64_t* tile_trellis = reinterpret_cast<int64_t*>(
        table + RouteLayout::tile_trellis_offset(assignments));
    int* tile_columns = table + RouteLayout::tile_columns_offset(assignments);
    for (int q = 0; q < 3; ++q) {
      const int64_t* meta = routing.metadata + int64_t(lo) * 12 + q * 4;
      // The launch's format only (kBitsX2, codebook 2); anything else traps.
      if (meta[2] != kBitsX2 || meta[3] != 2) __trap();
      tile_trellis[t * 3 + q] = routing.pointers[int64_t(lo) * 9 + q * 3];
      tile_columns[t * 3 + q] = static_cast<int>(meta[1] / 16);
    }
  }
  int* sorted = table + RouteLayout::sorted_offset();
  int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    if (e >= 0) {
      const int at = atomicAdd(&cursor[e], 1);
      sorted[at] = static_cast<int>(a);
      sorted_experts[at] = static_cast<int>(e);
    }
  }
}

__device__ __forceinline__ double silu(double g) { return g / __dadd_rn(1.0, exp(-g)); }

// Lane order of a K-long row (MMA B operand): k step k / 16, then lane t's k
// values 2t, 2t + 1, 2t + 8, 2t + 9 (t = (k % 8) / 2).
__device__ __forceinline__ int64_t lane_position(int64_t k) {
  const int kk = static_cast<int>(k & 15);
  return (k & ~int64_t(15)) + ((kk & 7) >> 1) * 4 + ((kk >> 3) << 1) + (kk & 1);
}

// The tensor-core B operand of one row: e per scale group (the peak over the
// group's warps) puts max |v| 2^e in [2^14, 2^15) (2^e and 2^-e stay normal
// FP32), hi = f16(v 2^e) (exact scaling), lo = f16(v 2^e - hi), stored at
// lane_position(k) of the row's high and low FP16 rows.
template <int kVals>
__device__ __forceinline__ float warp_peak(const float (&v)[kVals]) {
  float peak = 0.0f;
#pragma unroll
  for (int i = 0; i < kVals; ++i) peak = fmaxf(peak, fabsf(v[i]));
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
  return peak;
}
__device__ __forceinline__ int halves_exponent(float peak) {
  const int biased = static_cast<int>(__float_as_uint(peak) >> 23);
  return peak == 0.0f ? 0 : min(141 - biased, 126);
}
// The lane's values at k = first + lane + 32 i, scaled by 2^e.
template <int kVals>
__device__ __forceinline__ void store_scaled_halves(const float (&v)[kVals], int lane, int first,
                                                    int e, __half* hi, __half* lo) {
  const float scale = __uint_as_float(uint32_t(127 + e) << 23);
#pragma unroll
  for (int i = 0; i < kVals; ++i) {
    const int64_t at = lane_position(first + lane + 32 * i);
    const float y = __fmul_rn(v[i], scale);
    const __half h = __float2half_rn(y);
    hi[at] = h;
    lo[at] = __float2half_rn(__fsub_rn(y, __half2float(h)));
  }
}

// Where the down GEMM wants its input: per assignment [hi | lo] FP16 rows of
// ipad halves, and one 2^-e per (assignment, down K slice) (the same value
// for every slice: one exponent per row).
struct DownHalves {
  __half* halves;
  float* unscale;
};

// One task: the routing entry at sorted position pos, 128-column block
// `block`; returns false past the routed count, else the transformed values
// (lane holds columns block * 128 + lane + 32 i) and the assignment. Gate and
// up arrive as kSlices K-slice partials (slice_stride apart), summed in
// increasing slice order; then the gate and up output transforms (H128, svh),
// SiLU(gate) * up, down suh and the down H128, all in FP64. Three dependent
// memory round trips: the routing table (count, sorted entry and its expert
// together), then the expert's svh/suh values with the first chunk of up to 8
// slices of partials, then each further chunk (all of a chunk's loads in
// flight at once, summed after it lands: the same additions in the same
// order).
template <int kSlices>
__device__ __forceinline__ bool activation_block(const Routing& routing, const int* table,
                                                 int64_t assignments, int64_t intermediate,
                                                 int64_t ipad, const float* gate, const float* up,
                                                 int64_t slice_stride, int64_t pos, int64_t block,
                                                 int lane, float limit, double (&v)[4],
                                                 int64_t& assignment) {
  const int* sorted = table + RouteLayout::sorted_offset();
  const int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  const int count = __ldg(table);
  assignment = __ldg(sorted + pos);
  const int64_t expert = __ldg(sorted_experts + pos);
  if (pos >= count) return false;
  const int64_t base = block * 128 + lane;
  const half* gate_svh = expert_svh(routing, expert, 0);
  const half* up_svh = expert_svh(routing, expert, 1);
  const half* down_suh = expert_suh(routing, expert, 2);
  float gate_scale[4];
  float up_scale[4];
  float down_scale[4];
  double g[4];
  double u[4];
  constexpr int kChunk = kSlices < 8 ? kSlices : 8;
  static_assert(kSlices % kChunk == 0, "whole chunks of slices");
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    gate_scale[i] = __half2float(gate_svh[base + 32 * i]);
    up_scale[i] = __half2float(up_svh[base + 32 * i]);
    down_scale[i] = __half2float(down_suh[base + 32 * i]);
  }
#pragma unroll
  for (int s0 = 0; s0 < kSlices; s0 += kChunk) {
    float gs[kChunk][4];
    float us[kChunk][4];
#pragma unroll
    for (int s = 0; s < kChunk; ++s) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        gs[s][i] = __ldg(gate + (s0 + s) * slice_stride + assignment * ipad + base + 32 * i);
        us[s][i] = __ldg(up + (s0 + s) * slice_stride + assignment * ipad + base + 32 * i);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int s = 0; s < kChunk; ++s) {
        if (s0 + s == 0) {
          g[i] = gs[0][i];
          u[i] = us[0][i];
        } else {
          g[i] = __dadd_rn(g[i], double(gs[s][i]));
          u[i] = __dadd_rn(u[i], double(us[s][i]));
        }
      }
    }
  }
  hadamard128_warp_f64(g, lane);
  hadamard128_warp_f64(u, lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t col = base + 32 * i;
    double gv = __dmul_rn(g[i], double(gate_scale[i]));
    double uv = __dmul_rn(u[i], double(up_scale[i]));
    v[i] = 0.0;
    if (col < intermediate) {
      swiglu_clamp(gv, uv, limit);
      v[i] = __dmul_rn(__dmul_rn(silu(gv), uv), double(down_scale[i]));
    }
  }
  hadamard128_warp_f64(v, lane);
  return true;
}

// The activation stage: rounded once to FP32 and written as the down GEMM's
// tensor-core operand with one 2^e per row (down_slices copies of 2^-e). One
// thread block per sorted position (kRowBlocks warps = ipad / 128, one per
// H128 block), the row peak from the warps' peaks in shared memory.
template <int kSlices, int kRowBlocks>
__global__ void __launch_bounds__(32 * kRowBlocks)
activate_down_row(Routing routing, const int* table, int64_t assignments, int64_t intermediate,
                  int64_t ipad, const float* gate, const float* up, int64_t slice_stride,
                  DownHalves halves, int down_slices, float limit) {
  __shared__ float warp_peaks[kRowBlocks];
  const int lane = threadIdx.x & 31;
  const int block = threadIdx.x >> 5;
  // Uniform trip count per thread block (barriers inside).
  for (int64_t pos = blockIdx.x; pos < assignments; pos += gridDim.x) {
    double v[4];
    int64_t assignment;
    if (!activation_block<kSlices>(routing, table, assignments, intermediate, ipad, gate, up,
                                   slice_stride, pos, block, lane, limit, v, assignment)) {
      break;
    }
    float f[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __double2float_rn(v[i]);
    const float peak = warp_peak(f);
    if (lane == 0) warp_peaks[block] = peak;
    __syncthreads();
    float row_peak = 0.0f;
#pragma unroll
    for (int w = 0; w < kRowBlocks; ++w) row_peak = fmaxf(row_peak, warp_peaks[w]);
    const int e = halves_exponent(row_peak);
    __half* hi = halves.halves + assignment * 2 * ipad;
    store_scaled_halves(f, lane, block * 128, e, hi, hi + ipad);
    if (block == 0 && lane < down_slices) {
      halves.unscale[assignment * down_slices + lane] = __uint_as_float(uint32_t(127 - e) << 23);
    }
    __syncthreads();
  }
}

// One block per (token, 128-column block), one warp per route (blockDim =
// 32 * min(topk, 32), at least 4 warps). Warp w handles routes w, w + warps,
// ...: the local route's down output (the sum of kSlices K-slice partials,
// slice_stride apart, in increasing slice order) transform (H128, svh) in
// FP64 into shared memory, every route's loads in flight at once. Then
// threads 0..127 apply exactly one routing-weight FMA per local route in
// route order (FP64) and round to FP32, then to BF16. Local experts come from
// the routing table's by-assignment array. Dynamic shared memory:
// combine_shared_bytes(topk).
// The combine output: the FP64 route sum rounded to FP32 (moe_out_fp32), and
// that FP32 value rounded to BF16 (moe_out).
__device__ __forceinline__ void store_out(float* out, double sum) {
  *out = __double2float_rn(sum);
}
__device__ __forceinline__ void store_out(__nv_bfloat16* out, double sum) {
  *out = __float2bfloat16_rn(__double2float_rn(sum));
}
constexpr int kCombineMaxThreads = 1024;
size_t combine_shared_bytes(int64_t topk) {
  return size_t(topk) * (128 * sizeof(double) + sizeof(float) + sizeof(int));
}
template <int kSlices, class OutT>
__global__ void __launch_bounds__(kCombineMaxThreads)
combine_routes(const float* down, const float* weights, Routing routing, const int* table,
               int64_t rows, int64_t hidden, int64_t hpad, int64_t slice_stride,
               OutT* out) {
  extern __shared__ __align__(16) unsigned char combine_shared[];
  const int64_t topk = routing.topk;
  double* values = reinterpret_cast<double*>(combine_shared);      // [topk][128]
  float* route_weight = reinterpret_cast<float*>(values + topk * 128);
  int* route_local = reinterpret_cast<int*>(route_weight + topk);
  const int* by_assignment = table + RouteLayout::by_assignment_offset(rows * topk);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t blocks = hpad / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col0 = (task % blocks) * 128;
    for (int64_t r = warp; r < topk; r += blockDim.x / 32) {
      const int64_t assignment = row * topk + r;
      const int64_t expert = __ldg(by_assignment + assignment);
      if (lane == 0) {
        route_local[r] = expert >= 0;
        route_weight[r] = weights[assignment];
      }
      if (expert < 0) continue;
      const half* svh = expert_svh(routing, expert, 2);
      float v[kSlices][4];
#pragma unroll
      for (int s = 0; s < kSlices; ++s) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          v[s][i] = down[s * slice_stride + assignment * hpad + col0 + lane + 32 * i];
        }
      }
      float scale[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) scale[i] = __half2float(svh[col0 + lane + 32 * i]);
      double x[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[i] = v[0][i];
#pragma unroll
        for (int s = 1; s < kSlices; ++s) x[i] = __dadd_rn(x[i], double(v[s][i]));
      }
      hadamard128_warp_f64(x, lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[r * 128 + lane + 32 * i] = __dmul_rn(x[i], double(scale[i]));
      }
    }
    __syncthreads();
    if (threadIdx.x < 128) {
      double sum = 0.0;
      for (int64_t r = 0; r < topk; ++r) {
        if (route_local[r]) {
          sum = __fma_rn(values[r * 128 + threadIdx.x], double(route_weight[r]), sum);
        }
      }
      const int64_t col = col0 + threadIdx.x;
      if (col < hidden) store_out(out + row * hidden + col, sum);
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Lane-tile expert GEMM (codebook 2, one bits_x2 per instance).
//
// An item is one warp's kNT output n16 tiles x one K slice of one route tile;
// no shared memory and no block barriers. The warp decodes its own 16x16
// trellis tiles straight into m16n8k16 A fragments (A = W^T: 16 output
// columns x 16 k; decode_lane_fragment_mul1) and multiplies them on the tensor
// cores (FP16 in, FP32 out, C = 0 per k16 step) with B = 16 k x 8 columns: per
// group of four route rows, columns 0..3 hold the rows' high FP16 parts and
// 4..7 the low parts of their activations scaled by a power of two per row
// and scale group (store_scaled_halves; produced once per call by
// route_and_gate_up_input and activate_down_row, read straight from global
// memory). The k16 results are summed in FP32 in increasing k per K slice,
// high and low apart; the slice partial (hi + lo) 2^-e goes to the workspace
// and the consumer kernel sums the slices in increasing slice order (FP64).
// Everything is fixed-order: a row's result never depends on other rows.
constexpr int kLaneRows = 8;                    // route rows per tile
constexpr int64_t kLaneMaxAssignments = 1024;   // routed rows the workspace layout covers
constexpr int kLaneThreads = 256;
constexpr int kLaneAhead = 3;                   // k steps prefetched to L2 (register loads)
// How a warp gets its trellis words: through a shared ring (cp.async, 16-byte
// aligned trellis), as per-lane 32-bit loads (4-byte aligned) or as per-lane
// halfword loads (any other address).
enum LaneLoad { kLaneHalves, kLaneWords, kLaneRing };
constexpr int lane_step_bytes(int nt, int bits_x2) { return nt * 16 * bits_x2; }
// The ring pays where a k step holds at least 384 bytes (bits_x2 6: 9-10 %
// faster); at 256 bytes (bits_x2 4) the register path with L2 prefetch is ~1 %
// faster.
constexpr bool lane_uses_ring(int nt, int bits_x2) { return lane_step_bytes(nt, bits_x2) >= 384; }
// A warp's ring: 4 k steps (3 in flight while one decodes; 8 or 16 measured
// no faster), halved while a block's rings exceed 64 KB.
constexpr int lane_stages(int nt, int bits_x2) {
  int stages = 4;
  while (stages > 2 && lane_step_bytes(nt, bits_x2) * stages * (kLaneThreads / 32) > 65536) {
    stages /= 2;
  }
  return stages;
}
constexpr int lane_ring_bytes(int nt, int bits_x2) {
  return lane_uses_ring(nt, bits_x2)
             ? lane_stages(nt, bits_x2) * lane_step_bytes(nt, bits_x2) * (kLaneThreads / 32)
             : 0;
}

struct LaneProblem {
  const int* table;
  int64_t assignments;       // routing table capacity
  int k_tiles;               // K / 16
  int n_tiles;               // N / 16
  int projections;           // 2: gate (out0) and up (out1); 1: down (out0)
  int first_projection;
  float* out0;               // [slice][assignment][N] partials
  float* out1;
  int64_t slice_stride;
  // B operand: per (assignment, projection) row [hi | lo] FP16 rows of K
  // halves in lane order, and 2^-e per (assignment, projection, K slice).
  const __half* halves;
  const float* unscale;
};

__device__ __forceinline__ void mma_f16_f32(const uint32_t (&a)[4], uint2 b, float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "f"(0.0f));
}

// One warp: kNT output n16 tiles x kKT k16 steps of one route tile with
// kGroups n8 row groups (row_of: lane r holds tile row r's assignment), k
// outer: the kNT tiles of a k step are adjacent in the trellis; the next k
// step's B fragments are in flight while this one decodes. kLaneRing: each k
// step's trellis bytes go through the warp's shared ring (cp.async),
// kLaneStages - 1 steps ahead of the decode. Otherwise: per-lane loads one
// step ahead in registers (immediate offsets from per-lane word pointers for
// kLaneWords), L2 prefetch kLaneAhead steps ahead.
template <int kGroups, int kNT, int kKT, int kBitsX2, int kLoad>
__device__ __forceinline__ void lane_tiles(const LaneProblem& p, int pi, int slice, int row_of,
                                           int rows, const uint16_t* trellis, int64_t tiles_n,
                                           int n0, int lane, const LaneWindow<kBitsX2>& window,
                                           unsigned char* ring, float* out) {
  using Ring = TileRing<kBitsX2>;
  constexpr int kTileWords = Ring::kTileHalfwords / 2;
  const int kt0 = slice * kKT;
  const int slices = p.k_tiles / kKT;
  const int64_t row_halves = 2 * int64_t(p.k_tiles) * 16;
  const int t = lane & 3;
  const int gq = lane >> 2;
  // B source of this lane: column gq of each group = hi (gq < 4) or lo of
  // row 4g + gq % 4, k values 2t, 2t + 1, 2t + 8, 2t + 9 of each k step.
  const uint2* bsrc[kGroups];
  bool bok[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    const int r = 4 * g + (gq & 3);
    bok[g] = r < rows;
    const int64_t assignment = __shfl_sync(0xffffffffu, row_of, bok[g] ? r : 0);
    bsrc[g] = reinterpret_cast<const uint2*>(
                  p.halves + (assignment * p.projections + pi) * row_halves +
                  (gq >> 2) * (row_halves / 2) + kt0 * 16) + t;
  }
  // 2^-e of the rows this lane stores: 4g + 2t + c (lanes t < 2).
  float unscale[kGroups][2];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 4 * g + 2 * (t & 1) + c;
      const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
      unscale[g][c] = __ldg(p.unscale + (assignment * p.projections + pi) * slices + slice);
    }
  }
  const uint16_t* step = trellis + (int64_t(kt0) * tiles_n + n0) * Ring::kTileHalfwords;
  const int64_t stride = tiles_n * Ring::kTileHalfwords;
  constexpr int kStepBytes = lane_step_bytes(kNT, kBitsX2);
  constexpr int kLaneStages = lane_stages(kNT, kBitsX2);
  uint32_t words[kNT][Ring::kLoadWords];
  // Ring stage: the step's kStepBytes of trellis in 16-byte chunks, lane
  // strided. Lane addresses are set up once: the copy source advances by one
  // stride per step, the window words are read at fixed per-lane shared
  // offsets (rd) plus the stage and the tile offset.
  const char* fill = reinterpret_cast<const char*>(step) + lane * 16;
  const uint32_t ring_at =
      kLoad == kLaneRing ? static_cast<uint32_t>(__cvta_generic_to_shared(ring)) : 0u;
  auto issue = [&](int s) {
    if (s < kKT) {
      const uint32_t stage = ring_at + lane * 16 + (s & (kLaneStages - 1)) * kStepBytes;
#pragma unroll
      for (int c = 0; c < kStepBytes / 16; c += 32) {
        if (c + lane < kStepBytes / 16) {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(stage + c * 16),
                       "l"(fill + c * 16));
        }
      }
    }
    asm volatile("cp.async.commit_group;");
    fill += stride * 2;
  };
  uint32_t rd[Ring::kLoadWords];
#pragma unroll
  for (int i = 0; i < Ring::kLoadWords; ++i) rd[i] = ring_at + 4 * window.word[i];
  uint32_t slot = 0;   // the current k step's ring stage offset
  auto load_ring_words = [&](int j) {
#pragma unroll
    for (int i = 0; i < Ring::kLoadWords; ++i) {
      asm volatile("ld.shared.u32 %0, [%1];"
                   : "=r"(words[j][i])
                   : "r"(rd[i] + slot + j * kTileWords * 4));
    }
  };
  const uint32_t* wp[Ring::kLoadWords];
#pragma unroll
  for (int i = 0; i < Ring::kLoadWords; ++i) {
    wp[i] = reinterpret_cast<const uint32_t*>(step) + window.word[i];
  }
  auto fetch = [&](int j) {
    if constexpr (kLoad == kLaneWords) {
#pragma unroll
      for (int i = 0; i < Ring::kLoadWords; ++i) words[j][i] = __ldg(wp[i] + j * kTileWords);
    } else {
      load_lane_words<kBitsX2>(step + j * Ring::kTileHalfwords, window, false, words[j]);
    }
  };
  if constexpr (kLoad == kLaneRing) {
#pragma unroll
    for (int s = 0; s < kLaneStages - 1; ++s) issue(s);
  } else {
#pragma unroll
    for (int j = 0; j < kNT; ++j) fetch(j);
  }
  uint2 bnext[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) bnext[g] = bok[g] ? __ldg(bsrc[g]) : make_uint2(0u, 0u);
  float acc[kNT][kGroups][4];
#pragma unroll
  for (int j = 0; j < kNT; ++j) {
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[j][g][i] = 0.0f;
    }
  }
  // L2 prefetch kLaneAhead k steps ahead: lane = 128-byte line of the step.
  const char* ahead = reinterpret_cast<const char*>(step + kLaneAhead * stride) +
                      min(lane * 128, kStepBytes - 1);
#pragma unroll 1
  for (int kt = 0; kt < kKT; ++kt) {
    const bool more = kt + 1 < kKT;
    uint2 b[kGroups];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      b[g] = bnext[g];
      if (more && bok[g]) bnext[g] = __ldg(bsrc[g] + 4 * (kt + 1));
    }
    if constexpr (kLoad == kLaneRing) {
      asm volatile("cp.async.wait_group %0;" ::"n"(kLaneStages - 2));
      asm volatile("bar.warp.sync -1;" ::: "memory");
      slot = (kt & (kLaneStages - 1)) * kStepBytes;
      // Tile 0's window words now; tile j + 1's while tile j decodes (two
      // tiles' words live, not kNT).
      load_ring_words(0);
      // Every lane has read step kt - 1's stage (before the barrier): refill it.
      issue(kt + kLaneStages - 1);
    } else {
      if (lane * 128 < kStepBytes + 127 && kt + kLaneAhead < kKT) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ahead));
      }
      ahead += stride * 2;
      if constexpr (kLoad == kLaneWords) {
#pragma unroll
        for (int i = 0; i < Ring::kLoadWords; ++i) wp[i] += stride / 2;
      } else {
        step += stride;
      }
    }
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
      if constexpr (kLoad == kLaneRing) {
        if (j + 1 < kNT) load_ring_words(j + 1);
      }
      uint32_t a[4];
      decode_lane_fragment_mul1<kBitsX2>(words[j], window.shift, a);
      if constexpr (kLoad != kLaneRing) {
        if (more) fetch(j);
      }
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        float d[4];
        mma_f16_f32(a, b[g], d);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][g][i] = __fadd_rn(acc[j][g][i], d[i]);
      }
    }
  }
  if constexpr (kLoad == kLaneRing) {
    // The next item refills the ring: every lane is past its last read.
    asm volatile("cp.async.wait_all;");
    __syncwarp();
  }
  // Lanes t < 2 hold the high sums of rows 4g + 2t + {0, 1}; lane t + 2 the
  // low sums of the same rows and columns.
  const int64_t n_stride = int64_t(p.n_tiles) * 16;
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 4 * g + 2 * (t & 1) + c;
      const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
#pragma unroll
      for (int j = 0; j < kNT; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int i = 2 * h + c;
          const float low = __shfl_xor_sync(0xffffffffu, acc[j][g][i], 2);
          if (t < 2 && r < rows) {
            out[assignment * n_stride + (n0 + j) * 16 + gq + 8 * h] =
                __fmul_rn(__fadd_rn(acc[j][g][i], low), unscale[g][c]);
          }
        }
      }
    }
  }
}

// The halfword-load variant, out of line: inlining it beside the hot one
// costs the hot loops about 3 % (register allocation and code layout).
template <int kGroups, int kNT, int kKT, int kBitsX2>
__device__ __noinline__ void lane_tiles_unaligned(LaneProblem p, int pi, int slice, int row_of,
                                                  int rows, const uint16_t* trellis,
                                                  int64_t tiles_n, int n0, int lane,
                                                  float* out) {
  const LaneWindow<kBitsX2> window(lane);
  lane_tiles<kGroups, kNT, kKT, kBitsX2, kLaneHalves>(p, pi, slice, row_of, rows, trellis,
                                                      tiles_n, n0, lane, window, nullptr, out);
}

// Warp items: (route tile, projection, K slice, column group of kNT n16
// tiles), column groups fastest so a block's warps share B rows in L1.
template <int kNT, int kKT, int kBitsX2>
__global__ void __launch_bounds__(kLaneThreads, 2)
moe_lane_kernel(LaneProblem p) {
  const int lane = threadIdx.x & 31;
  const LaneWindow<kBitsX2> window(lane);
  extern __shared__ __align__(16) unsigned char lane_rings[];
  unsigned char* ring =
      lane_rings + (threadIdx.x >> 5) * lane_stages(kNT, kBitsX2) * lane_step_bytes(kNT, kBitsX2);
  const int slices = p.k_tiles / kKT;
  const int groups_n = p.n_tiles / kNT;
  const int64_t per_tile = int64_t(p.projections) * slices * groups_n;
  const int64_t items = int64_t(p.table[1]) * per_tile;
  const int* sorted = p.table + RouteLayout::sorted_offset();
  const int4* tiles = reinterpret_cast<const int4*>(p.table + RouteLayout::tiles_offset(p.assignments));
  const long long* tile_trellis = reinterpret_cast<const long long*>(
      p.table + RouteLayout::tile_trellis_offset(p.assignments));
  const int* tile_columns = p.table + RouteLayout::tile_columns_offset(p.assignments);
  const int64_t warps = int64_t(gridDim.x) * (kLaneThreads / 32);
  for (int64_t item = int64_t(blockIdx.x) * (kLaneThreads / 32) + (threadIdx.x >> 5);
       item < items; item += warps) {
    const int64_t tile_index = item / per_tile;
    int rest = static_cast<int>(item % per_tile);
    const int column_group = rest % groups_n;
    rest /= groups_n;
    const int slice = rest % slices;
    const int pi = rest / slices;
    const int projection = p.first_projection + pi;
    // One round trip for the tile, its trellis address and columns (written
    // by the route kernel, which also checked the format).
    const int4 tile = tiles[tile_index];
    const uint16_t* trellis = reinterpret_cast<const uint16_t*>(
        __ldg(tile_trellis + tile_index * 3 + projection));
    const int64_t tiles_n = __ldg(tile_columns + tile_index * 3 + projection);
    const int rows = tile.z;
    const int row_of = lane < rows ? sorted[tile.y + lane] : 0;
    const int n0 = column_group * kNT;
    constexpr bool kRing = lane_uses_ring(kNT, kBitsX2);
    constexpr int kLoad = kRing ? kLaneRing : kLaneWords;
    const bool aligned = (reinterpret_cast<uintptr_t>(trellis) & (kRing ? 15 : 3)) == 0;
    float* out = (pi == 0 ? p.out0 : p.out1) + slice * p.slice_stride;
#define EXL3_LANE_GROUPS(G)                                                                     \
    if (aligned) {                                                                              \
      lane_tiles<G, kNT, kKT, kBitsX2, kLoad>(p, pi, slice, row_of, rows, trellis, tiles_n, n0, \
                                              lane, window, ring, out);                         \
    } else {                                                                                    \
      lane_tiles_unaligned<G, kNT, kKT, kBitsX2>(p, pi, slice, row_of, rows, trellis, tiles_n,  \
                                                 n0, lane, out);                                \
    }
    if (rows <= 4) {
      EXL3_LANE_GROUPS(1);
    } else {
      EXL3_LANE_GROUPS(2);
    }
#undef EXL3_LANE_GROUPS
  }
}

// One launch: block 0 routes (route_block) while the other blocks form the
// gate/up tensor-core B operand straight from topk_ids (independent of the
// routing table). One thread block per local assignment (above one H128 block per
// warp: per (assignment, projection)), each warp kBlocksPerWarp consecutive H128
// blocks of the row (kRouteInputThreads / 32 x kBlocksPerWarp = hpad / 128),
// its projection(s): x * suh and H128 in FP64,
// rounded once to FP32; the 2^e of a (projection, kScaleK-column group) comes
// from the peak over the group's H128 blocks (shared memory), then the halves
// are stored by store_scaled_halves. A group spans kSplit GEMM K slices, each
// given its own copy of 2^-e. One block per warp keeps its own body: the same
// arithmetic in the instruction order the hidden-2560 shape was tuned with.
template <int kBitsX2, int kBlocksPerWarp, int kScaleK, int kSplit>
__global__ void __launch_bounds__(kRouteInputThreads)
route_and_gate_up_input(Routing routing, int64_t assignments, int block_rows, int* table,
                        const __nv_bfloat16* x, int64_t hidden, int64_t hpad, __half* halves,
                        float* unscale) {
  if (blockIdx.x == 0) {
    route_block<kBitsX2>(routing, assignments, block_rows, table);
    return;
  }
  constexpr int kBlocks = kScaleK / 128;
  constexpr int kWarps = kRouteInputThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t slices = hpad / kScaleK * kSplit;
  if constexpr (kBlocksPerWarp == 1) {
    static_assert(kWarps % kBlocks == 0 && kSplit <= 32, "input warps must tile whole groups");
    __shared__ float warp_peaks[2][kWarps];
    const int group = warp / kBlocks;
    // Uniform trip count per thread block (barriers inside).
    for (int64_t assignment = blockIdx.x - 1; assignment < assignments;
         assignment += gridDim.x - 1) {
      const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
      float xs[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = int64_t(warp) * 128 + lane + 32 * i;
        xs[i] = col < hidden ? __bfloat162float(row[col]) : 0.0f;
      }
      const int64_t expert = local_expert(routing, assignment);
      float v[2][4] = {};
      if (expert >= 0) {
        const half* suh[2] = {expert_suh(routing, expert, 0), expert_suh(routing, expert, 1)};
        float scale[2][4];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int64_t col = int64_t(warp) * 128 + lane + 32 * i;
            scale[q][i] = col < hidden ? __half2float(suh[q][col]) : 0.0f;
          }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          double block[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) block[i] = __dmul_rn(xs[i], scale[q][i]);
          hadamard128_warp_f64(block, lane);
#pragma unroll
          for (int i = 0; i < 4; ++i) v[q][i] = __double2float_rn(block[i]);
        }
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const float peak = warp_peak(v[q]);
        if (lane == 0) warp_peaks[q][warp] = peak;
      }
      __syncthreads();
      if (expert >= 0 && warp < hpad / 128) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          float group_peak = 0.0f;
#pragma unroll
          for (int w = 0; w < kBlocks; ++w) {
            group_peak = fmaxf(group_peak, warp_peaks[q][group * kBlocks + w]);
          }
          const int e = halves_exponent(group_peak);
          __half* hi = halves + (assignment * 2 + q) * 2 * hpad + int64_t(group) * kScaleK;
          store_scaled_halves(v[q], lane, (warp % kBlocks) * 128, e, hi, hi + hpad);
          if (warp % kBlocks == 0 && lane < kSplit) {
            unscale[(assignment * 2 + q) * slices + group * kSplit + lane] =
                __uint_as_float(uint32_t(127 - e) << 23);
          }
        }
      }
      __syncthreads();
    }
  } else {
    constexpr int kWarpsPerGroup = kBlocks / kBlocksPerWarp;
    static_assert(kBlocks % kBlocksPerWarp == 0 && kWarps % kWarpsPerGroup == 0 && kSplit <= 32,
                  "input warps must tile whole groups");
    __shared__ float block_peaks[kWarps][kBlocksPerWarp];
    const int group = warp / kWarpsPerGroup;
    const int member = warp % kWarpsPerGroup;
    // One block per (assignment, projection) work item; non-local assignments
    // leave at once (uniform per block, before any barrier).
    for (int64_t work = blockIdx.x - 1; work < 2 * assignments; work += gridDim.x - 1) {
      const int64_t assignment = work >> 1;
      const int q = static_cast<int>(work & 1);
      const int64_t expert = local_expert(routing, assignment);
      if (expert < 0) continue;
      const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
      const half* suh = expert_suh(routing, expert, q);
      float v[kBlocksPerWarp][4];
#pragma unroll
      for (int b = 0; b < kBlocksPerWarp; ++b) {
        const int64_t first = (int64_t(warp) * kBlocksPerWarp + b) * 128;
        double block[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int64_t col = first + lane + 32 * i;
          const float xs = col < hidden ? __bfloat162float(row[col]) : 0.0f;
          const float scale = col < hidden ? __half2float(suh[col]) : 0.0f;
          block[i] = __dmul_rn(xs, scale);
        }
        hadamard128_warp_f64(block, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[b][i] = __double2float_rn(block[i]);
        const float peak = warp_peak(v[b]);
        if (lane == 0) block_peaks[warp][b] = peak;
      }
      __syncthreads();
      if (warp * kBlocksPerWarp < hpad / 128) {
        float group_peak = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarpsPerGroup; ++w) {
#pragma unroll
          for (int b = 0; b < kBlocksPerWarp; ++b) {
            group_peak = fmaxf(group_peak, block_peaks[group * kWarpsPerGroup + w][b]);
          }
        }
        const int e = halves_exponent(group_peak);
        __half* hi = halves + (assignment * 2 + q) * 2 * hpad + int64_t(group) * kScaleK;
#pragma unroll
        for (int b = 0; b < kBlocksPerWarp; ++b) {
          store_scaled_halves(v[b], lane, (member * kBlocksPerWarp + b) * 128, e, hi, hi + hpad);
        }
        if (member == 0 && lane < kSplit) {
          unscale[(assignment * 2 + q) * slices + group * kSplit + lane] =
              __uint_as_float(uint32_t(127 - e) << 23);
        }
      }
      __syncthreads();
    }
  }
}

// The served shapes, by padded hidden and intermediate size. Items cover
// kColumnTiles n16 tiles. Gate/up: calls below kLongSliceAssignments routed
// assignments give each (row, projection) one 2^e over the whole K and split
// it into slices of kGateUpShortSliceTiles k16 steps; larger calls one 2^e per
// kGateUpLongScaleK columns and slices of kGateUpLongSliceTiles steps. The
// down operand has one 2^e per row: slices of kDownShortSliceTiles steps below
// kLongDownAssignments assignments, of kDownLongSliceTiles steps from there
// on. Bits lists the bits_x2 values the lane kernels are built for.
template <int64_t kHidden, int64_t kIntermediate>
struct LaneTiling;

// Hidden 2560, intermediate 640: gate/up 8 slices of 20 steps (4 of 40, one
// 2^e per five H128 blocks), down 2 slices of 20 (the whole K, one item per
// row tile and column group).
template <>
struct LaneTiling<2560, 640> {
  using Bits = std::integer_sequence<int, 10>;
  static constexpr int kColumnTiles = 5;
  static constexpr int kGateUpShortScaleK = 2560;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = 640;
  static constexpr int kGateUpLongSliceTiles = 40;
  static constexpr int kDownShortSliceTiles = 20;
  static constexpr int kDownLongSliceTiles = 40;
  static constexpr int64_t kLongSliceAssignments = 160;
  static constexpr int64_t kLongDownAssignments = 160;
};

// Hidden 5120, intermediate 2304 (whole experts per rank; at expert parallel
// size 4 a quarter of the routes is local): column groups of 8 n16 tiles
// divide 144 and 320 (8 rather than 4: the per-step B, ring and loop work is
// shared by twice the tiles, 6-10 % faster); gate/up 16 slices of 20 steps,
// down 8 slices of 18. One
// slice layout for every call of up to 64 rows (384 assignments; the long
// parameters repeat the short ones and never apply), so a row's result does
// not depend on how many rows share the call.
template <>
struct LaneTiling<5120, 2304> {
  using Bits = std::integer_sequence<int, 6, 4>;
  static constexpr int kColumnTiles = 8;
  static constexpr int kGateUpShortScaleK = 5120;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = kGateUpShortScaleK;
  static constexpr int kGateUpLongSliceTiles = kGateUpShortSliceTiles;
  static constexpr int kDownShortSliceTiles = 18;
  static constexpr int kDownLongSliceTiles = kDownShortSliceTiles;
  static constexpr int64_t kLongSliceAssignments = std::numeric_limits<int64_t>::max();
  static constexpr int64_t kLongDownAssignments = std::numeric_limits<int64_t>::max();
};

// dsv41: hidden 5120 with the tensor-parallel intermediate chunks 640 / 512 (all 384 experts on every rank), with
// the slice layout and scale groups of the moe_out_v2 these ranks served before (DeepSeekv41-Flash-EXL3-TC-MoE
// moe_v2.cu: gate/up 16 slices of 20 steps below 160 assignments, else one 2^e per 1,280 columns and 8 slices of
// 40; down 2 slices below 160 assignments, else the whole K), so the K order of every sum is the same; only the
// column grouping (8 n16 tiles, which does not change any sum) and the trellis loads are this file's.
template <int64_t kI>
struct LaneTilingTp {
  using Bits = std::integer_sequence<int, 6, 4>;
  static constexpr int kColumnTiles = 8;
  static constexpr int kGateUpShortScaleK = 5120;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = 1280;
  static constexpr int kGateUpLongSliceTiles = 40;
  static constexpr int kDownShortSliceTiles = kI == 640 ? 20 : 16;
  static constexpr int kDownLongSliceTiles = int(kI / 16);
  static constexpr int64_t kLongSliceAssignments = 160;
  static constexpr int64_t kLongDownAssignments = 160;
};
template <>
struct LaneTiling<5120, 640> : LaneTilingTp<640> {};
template <>
struct LaneTiling<5120, 512> : LaneTilingTp<512> {};

template <int64_t kH, int64_t kI>
struct LaneShape : LaneTiling<kH, kI> {
  using Tiling = LaneTiling<kH, kI>;
  static constexpr int64_t kHidden = kH;
  static constexpr int64_t kIntermediate = kI;
  static constexpr int kGateUpShortSlices = int(kH / (16 * Tiling::kGateUpShortSliceTiles));
  static constexpr int kGateUpLongSlices = int(kH / (16 * Tiling::kGateUpLongSliceTiles));
  static constexpr int kDownShortSlices = int(kI / (16 * Tiling::kDownShortSliceTiles));
  static constexpr int kDownLongSlices = int(kI / (16 * Tiling::kDownLongSliceTiles));
  // Workspace layout: room for the larger slice counts.
  static constexpr int kGateUpSlices = kGateUpShortSlices;
  static constexpr int kDownSlices = kDownShortSlices;
  // H128 blocks of a row per input warp, and the activation stage's warps.
  static constexpr int kInputBlocksPerWarp = int(kH / 128) / (kRouteInputThreads / 32);
  static constexpr int kRowBlocks = int(kI / 128);
  static_assert(kH % (16 * Tiling::kGateUpShortSliceTiles) == 0 &&
                    kH % (16 * Tiling::kGateUpLongSliceTiles) == 0 &&
                    kGateUpShortSlices >= kGateUpLongSlices &&
                    Tiling::kGateUpShortScaleK % (16 * Tiling::kGateUpShortSliceTiles) == 0 &&
                    Tiling::kGateUpLongScaleK % (16 * Tiling::kGateUpLongSliceTiles) == 0 &&
                    kH % Tiling::kGateUpShortScaleK == 0 && kH % Tiling::kGateUpLongScaleK == 0 &&
                    kI % (16 * Tiling::kDownShortSliceTiles) == 0 &&
                    kI % (16 * Tiling::kDownLongSliceTiles) == 0 &&
                    kDownShortSlices >= kDownLongSlices && kDownShortSlices <= 32 &&
                    kH % (16 * Tiling::kColumnTiles) == 0 &&
                    kI % (16 * Tiling::kColumnTiles) == 0,
                "column groups, K slices and scale groups must divide the shape; a "
                "gate/up scale group holds whole K slices");
  static_assert(kH % (128 * (kRouteInputThreads / 32)) == 0 && kI % 128 == 0 &&
                    kRowBlocks * 32 <= 1024,
                "the input warps cover whole H128 blocks of a row; one activation warp "
                "per H128 block of the intermediate");
};

constexpr const char* kShapeMessage =
    "EXL3 MoE v2 supports padded hidden 2560 with intermediate 640 and padded hidden 5120 "
    "with intermediate 2304, 640 or 512 only";

// f(LaneShape<hpad, ipad>{}) for a served shape; false for any other.
template <class F>
bool with_shape(int64_t hpad, int64_t ipad, F&& f) {
  if (hpad == 2560 && ipad == 640) {
    f(LaneShape<2560, 640>{});
    return true;
  }
  if (hpad == 5120 && ipad == 2304) {
    f(LaneShape<5120, 2304>{});
    return true;
  }
  if (hpad == 5120 && ipad == 640) {
    f(LaneShape<5120, 640>{});
    return true;
  }
  if (hpad == 5120 && ipad == 512) {
    f(LaneShape<5120, 512>{});
    return true;
  }
  return false;
}

// f(std::integral_constant<int, bits_x2>{}) if bits_x2 is one of kBits.
template <int... kBits, class F>
bool with_bits(std::integer_sequence<int, kBits...>, int64_t bits_x2, F&& f) {
  return ((bits_x2 == kBits ? (f(std::integral_constant<int, kBits>{}), true) : false) || ...);
}

void check_assignments(int64_t assignments) {
  TORCH_CHECK(assignments <= kLaneMaxAssignments,
              "EXL3 MoE v2 supports at most 1024 routed assignments per call");
}

template <int kNT, int kKT, int kBitsX2>
void launch_lane(const LaneProblem& p, int64_t max_tiles, cudaStream_t stream) {
  auto* kernel = &moe_lane_kernel<kNT, kKT, kBitsX2>;
  constexpr int kRingBytes = lane_ring_bytes(kNT, kBitsX2);
  static const int64_t resident = [kernel] {
    int device = 0;
    int sms = 0;
    int per_sm = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kRingBytes));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                                 kLaneThreads, kRingBytes));
    return int64_t(std::max(sms, 1)) * std::max(per_sm, 1);
  }();
  const int64_t per_tile = int64_t(p.projections) * (p.k_tiles / kKT) * (p.n_tiles / kNT);
  constexpr int64_t kWarps = kLaneThreads / 32;
  const int64_t grid = std::min((max_tiles * per_tile + kWarps - 1) / kWarps, resident);
  if (grid <= 0) return;
  kernel<<<static_cast<unsigned int>(grid), kLaneThreads, kRingBytes, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// FP32 workspace elements before the routing table: gate, up and down slice
// partials; the gate/up and down B operands (FP16 hi and lo rows) and their
// 2^-e per (row, slice), rounded up to whole 16-byte groups.
template <class S>
int64_t lane_workspace(int64_t assignments, int64_t hpad, int64_t ipad) {
  const int64_t gate_up = checked_mul(2 * S::kGateUpSlices, ipad);
  const int64_t down = checked_mul(S::kDownSlices, hpad);
  const int64_t operands = checked_add(checked_mul(2, hpad), ipad);   // FP16 hi + lo
  const int64_t unscale = (assignments * (2 * S::kGateUpSlices + S::kDownSlices) + 3) / 4 * 4;
  return checked_add(
      checked_mul(assignments, checked_add(checked_add(gate_up, down), operands)), unscale);
}

// One call's arguments, checked by moe_out_v2.
struct LaneCall {
  Routing routing;
  const __nv_bfloat16* x;
  const float* topk_weights;
  void* out;  // __nv_bfloat16 (moe_out) or float (moe_out_fp32)
  bool out_fp32;
  float* workspace;
  int64_t rows;
  int64_t hidden;
  int64_t hpad;
  int64_t ipad;
  int64_t intermediate;
  int64_t experts;
  int64_t assignments;
  float limit;
  cudaStream_t stream;
};

template <class S, int kBitsX2>
void run_lanes(const LaneCall& c) {
  Routing routing = c.routing;
  const int64_t assignments = c.assignments;
  const int64_t hpad = c.hpad;
  const int64_t ipad = c.ipad;
  float* input = c.workspace;
  const float* down_out = input;
  int64_t down_stride = 0;
  const int* route_table = nullptr;   // the routing table combine_routes reads
  if (c.experts > 0 && assignments > 0) {
    const int64_t gate_up_stride = assignments * ipad;
    down_stride = assignments * hpad;
    float* gate = input;
    float* up = gate + S::kGateUpSlices * gate_up_stride;
    float* down = up + S::kGateUpSlices * gate_up_stride;
    // B operands: [assignment][gate, up][hi, lo][hpad] and [assignment][hi, lo][ipad]
    // FP16, then 2^-e per (assignment, projection, slice).
    __half* gate_up_halves = reinterpret_cast<__half*>(down + S::kDownSlices * down_stride);
    __half* down_halves = gate_up_halves + assignments * 2 * 2 * hpad;
    float* gate_up_unscale = reinterpret_cast<float*>(down_halves + assignments * 2 * ipad);
    float* down_unscale = gate_up_unscale + assignments * 2 * S::kGateUpSlices;
    // The int4 route tiles need a 16-byte aligned table.
    int* table = reinterpret_cast<int*>(
        gate_up_unscale + (assignments * (2 * S::kGateUpSlices + S::kDownSlices) + 3) / 4 * 4);
    route_table = table;
    down_out = down;
    // Routing (block 0) and the gate/up B operand (other blocks) in one launch,
    // then the gate/up GEMM and the activation, with short or long K slices.
    const auto gate_up = [&](auto scale_k, auto slice_tiles) {
      constexpr int kScaleK = decltype(scale_k)::value;
      constexpr int kSplit = kScaleK / (16 * decltype(slice_tiles)::value);
      route_and_gate_up_input<kBitsX2, S::kInputBlocksPerWarp, kScaleK, kSplit>
          <<<launch_blocks(1 + 2 * assignments),
             kRouteInputThreads, (3 * c.experts + 1) * sizeof(int), c.stream>>>(
              routing, assignments, kLaneRows, table, c.x, c.hidden, hpad, gate_up_halves,
              gate_up_unscale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    const bool long_slices = assignments >= S::kLongSliceAssignments;
    if (long_slices) {
      gate_up(std::integral_constant<int, S::kGateUpLongScaleK>{},
              std::integral_constant<int, S::kGateUpLongSliceTiles>{});
    } else {
      gate_up(std::integral_constant<int, S::kGateUpShortScaleK>{},
              std::integral_constant<int, S::kGateUpShortSliceTiles>{});
    }
    // Row tiles: sum over experts of ceil(rows_e / kLaneRows).
    const int64_t max_tiles = std::min<int64_t>(
        assignments, assignments / kLaneRows + std::min<int64_t>(c.experts, assignments) + 1);
    LaneProblem p{};
    p.table = table;
    p.assignments = assignments;
    p.k_tiles = static_cast<int>(hpad / 16);
    p.n_tiles = static_cast<int>(ipad / 16);
    p.projections = 2;
    p.first_projection = 0;
    p.out0 = gate;
    p.out1 = up;
    p.slice_stride = gate_up_stride;
    p.halves = gate_up_halves;
    p.unscale = gate_up_unscale;
    // Down operand with one 2^e per row; long or short K slices.
    const bool long_down = assignments >= S::kLongDownAssignments;
    const int down_used = long_down ? S::kDownLongSlices : S::kDownShortSlices;
    const auto activate = [&](auto slices) {
      activate_down_row<decltype(slices)::value, S::kRowBlocks>
          <<<launch_blocks(assignments), 32 * S::kRowBlocks, 0, c.stream>>>(
              routing, table, assignments, c.intermediate, ipad, gate, up, gate_up_stride,
              DownHalves{down_halves, down_unscale}, down_used, c.limit);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (long_slices) {
      launch_lane<S::kColumnTiles, S::kGateUpLongSliceTiles, kBitsX2>(p, max_tiles, c.stream);
      activate(std::integral_constant<int, S::kGateUpLongSlices>{});
    } else {
      launch_lane<S::kColumnTiles, S::kGateUpShortSliceTiles, kBitsX2>(p, max_tiles, c.stream);
      activate(std::integral_constant<int, S::kGateUpShortSlices>{});
    }
    p.k_tiles = static_cast<int>(ipad / 16);
    p.n_tiles = static_cast<int>(hpad / 16);
    p.projections = 1;
    p.first_projection = 2;
    p.out0 = down;
    p.out1 = nullptr;
    p.slice_stride = down_stride;
    p.halves = down_halves;
    p.unscale = down_unscale;
    if (long_down) {
      launch_lane<S::kColumnTiles, S::kDownLongSliceTiles, kBitsX2>(p, max_tiles, c.stream);
    } else {
      launch_lane<S::kColumnTiles, S::kDownShortSliceTiles, kBitsX2>(p, max_tiles, c.stream);
    }
  } else {
    // No local expert or no route: every row is exactly zero.
    routing.topk = 0;
  }
  const size_t shared = combine_shared_bytes(routing.topk);
  TORCH_CHECK(shared <= 48 * 1024, "EXL3 MoE topk too large for the combine stage");
  const auto combine = [&](auto slices) {
    const dim3 grid(launch_blocks(c.rows * (hpad / 128)));
    const dim3 block(
        32 * static_cast<int>(std::min<int64_t>(std::max<int64_t>(routing.topk, 4), 32)));
    if (c.out_fp32) {
      combine_routes<decltype(slices)::value, float><<<grid, block, shared, c.stream>>>(
          down_out, c.topk_weights, routing, route_table, c.rows, c.hidden, hpad, down_stride,
          static_cast<float*>(c.out));
    } else {
      combine_routes<decltype(slices)::value, __nv_bfloat16><<<grid, block, shared, c.stream>>>(
          down_out, c.topk_weights, routing, route_table, c.rows, c.hidden, hpad, down_stride,
          static_cast<__nv_bfloat16*>(c.out));
    }
  };
  if (assignments >= S::kLongDownAssignments) {
    combine(std::integral_constant<int, S::kDownLongSlices>{});
  } else {
    combine(std::integral_constant<int, S::kDownSlices>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
}  // namespace moe_v2

int64_t moe_workspace_size(int64_t max_rows, int64_t hidden_size,
                           int64_t intermediate_size, int64_t topk,
                           int64_t local_experts) {
  using namespace moe_v2;
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  const int64_t hpad = padded_size(hidden_size);
  const int64_t ipad = padded_size(intermediate_size);
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  int64_t lane = 0;
  TORCH_CHECK(with_shape(hpad, ipad,
                         [&](auto shape) {
                           check_assignments(assignments);
                           lane = lane_workspace<decltype(shape)>(assignments, hpad, ipad);
                         }),
              kShapeMessage);
  return workspace_elements(checked_add(lane, RouteLayout::ints(assignments)));
}

namespace {
void moe_out_any(const torch::Tensor& x, const torch::Tensor& topk_ids,
                 const torch::Tensor& topk_weights,
                 const torch::Tensor& packed_ptrs,
                 const torch::Tensor& packed_meta,
                 const torch::Tensor& expert_map, int64_t intermediate_size,
                 int64_t bits_x2, torch::Tensor& out, torch::Tensor& workspace,
                 double swiglu_limit, bool out_fp32) {
  using namespace moe_v2;
  const float limit = checked_swiglu_limit(swiglu_limit);
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_integer(topk_ids, x, "topk_ids");
  check_integer(expert_map, x, "expert_map");
  check_tensor(topk_weights, x, at::kFloat, "topk_weights");
  check_tensor(packed_ptrs, x, at::kLong, "packed_ptrs");
  check_tensor(packed_meta, x, at::kLong, "packed_meta");
  check_tensor(out, x, out_fp32 ? at::kFloat : at::kBFloat16, "out");
  check_tensor(workspace, x, at::kFloat, "workspace");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && topk_ids.dim() == 2 &&
                  topk_weights.dim() == 2 && expert_map.dim() == 1 &&
                  packed_ptrs.dim() == 3 && packed_meta.dim() == 3,
              "EXL3 MoE tensor ranks mismatch");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = packed_ptrs.size(0);
  const int64_t hpad = padded_size(hidden);
  const int64_t ipad = padded_size(intermediate_size);
  TORCH_CHECK(topk_ids.size(0) == rows && topk_weights.sizes() == topk_ids.sizes() &&
                  out.sizes() == x.sizes(), "EXL3 MoE row/topk/output geometry mismatch");
  TORCH_CHECK(packed_ptrs.size(1) == 3 && packed_ptrs.size(2) == 3 &&
                  packed_meta.size(0) == experts && packed_meta.size(1) == 3 &&
                  packed_meta.size(2) == 4, "EXL3 MoE pointer/metadata geometry mismatch");
  TORCH_CHECK(experts <= kMaxRouteExperts, "EXL3 MoE has too many local experts");
  const int64_t assignments = checked_mul(rows, topk);
  bool built = false;
  TORCH_CHECK(with_shape(hpad, ipad,
                         [&](auto shape) {
                           check_assignments(assignments);
                           built = with_bits(typename decltype(shape)::Bits{}, bits_x2,
                                             [](auto) {});
                         }),
              kShapeMessage);
  TORCH_CHECK(built, "EXL3 MoE v2 has no bits_x2 ", bits_x2, " kernels for padded hidden ",
              hpad, " and intermediate ", ipad);
  const int64_t required =
      moe_workspace_size(rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
  check_mutations(out, workspace,
                  {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const LaneCall call{
      Routing{topk_ids.data_ptr(), expert_map.data_ptr(), packed_ptrs.data_ptr<int64_t>(),
              packed_meta.data_ptr<int64_t>(), expert_map.numel(), experts, topk,
              topk_ids.scalar_type() == at::kLong, expert_map.scalar_type() == at::kLong},
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      topk_weights.data_ptr<float>(),
      out.data_ptr(),
      out_fp32,
      workspace.data_ptr<float>(),
      rows,
      hidden,
      hpad,
      ipad,
      intermediate_size,
      experts,
      assignments,
      limit,
      stream};
  with_shape(hpad, ipad, [&](auto shape) {
    with_bits(typename decltype(shape)::Bits{}, bits_x2, [&](auto bits) {
      run_lanes<decltype(shape), decltype(bits)::value>(call);
    });
  });
}
}  // namespace

void moe_out(const torch::Tensor& x, const torch::Tensor& topk_ids,
             const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
             const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
             int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
             torch::Tensor& workspace, double swiglu_limit) {
  moe_out_any(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
              intermediate_size, bits_x2, out, workspace, swiglu_limit, false);
}

void moe_out_fp32(const torch::Tensor& x, const torch::Tensor& topk_ids,
                  const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
                  const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
                  int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
                  torch::Tensor& workspace, double swiglu_limit) {
  moe_out_any(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
              intermediate_size, bits_x2, out, workspace, swiglu_limit, true);
}

}  // namespace exl3_try

TORCH_LIBRARY(_dsv41_mm_v2try, m) {
  m.def("moe_out_v2_try(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, int bits_x2, Tensor(a!) out, "
        "Tensor(b!) workspace, float swiglu_limit) -> ()");
  m.def("moe_workspace_size_v2_try(int max_rows, int hidden_size, "
        "int intermediate_size, int topk, int local_experts) -> int",
        &exl3_try::moe_workspace_size);
}

TORCH_LIBRARY_IMPL(_dsv41_mm_v2try, CUDA, m) {
  m.impl("moe_out_v2_try", &exl3_try::moe_out);
}
