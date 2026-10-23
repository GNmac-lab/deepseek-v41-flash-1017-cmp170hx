// dsv41 wo_a (grouped attention output projection) for M <= 8 tokens on sm_80, straight from the exact FP8 codes
// (ds41-patches/dsv41_woa, patch_wo_a_cuda.py). W: e4m3 [G*R, D] row-major, S: fp32 power-of-two [G, R/32, D/32],
// x: the attention output [M, G*D] before the inverse GPT-J RoPE, which is applied in the activation load with the
// _inverse_rope_gptj_kernel expressions (fp32: even a*cos + b*sin, odd b*cos - a*sin, rounded to bf16). Weights are
// the 16-row mma operand (lane (g, t): rows g, g + 8, k 16t..16t+15 of a 64-k chunk), tokens the n8 operand; e4m3 ->
// bf16 is exact (bit move x 2^120 * s, s a power of two); K is split over the warps of a block and summed in smem in
// fixed order: deterministic. Output [M, G*R] bf16, the contiguous layout wo_b reads.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ uint4 ld_w(const uint4* p)
{
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_x(const uint4* p)
{
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b)
{
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a), *reinterpret_cast<const __nv_bfloat162*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

// natural (k0, k1, k2, k3) e4m3 -> bf16 pairs (k0, k2), (k1, k3), times mult
__device__ __forceinline__ void dq4(uint32_t q, uint32_t mult, uint32_t& lo, uint32_t& hi)
{
    const uint32_t h = ((q & 0x7F007F00u) >> 4) | (q & 0x80008000u);
    const uint32_t u = q << 8;
    const uint32_t l = ((u & 0x7F007F00u) >> 4) | (u & 0x80008000u);
    lo = bmul(l, mult);
    hi = bmul(h, mult);
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int i)
{
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi)
{
    __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&r);
}

// inverse GPT-J RoPE on 16 activations (8 pairs) starting at rope lane 2 * half0, in place
__device__ __forceinline__ void rope16(uint4 (&xa)[2], const float* __restrict__ cs_row, int half0, int HALF)
{
    const float4 c0 = *reinterpret_cast<const float4*>(cs_row + half0);
    const float4 c1 = *reinterpret_cast<const float4*>(cs_row + half0 + 4);
    const float4 s0 = *reinterpret_cast<const float4*>(cs_row + HALF + half0);
    const float4 s1 = *reinterpret_cast<const float4*>(cs_row + HALF + half0 + 4);
    const float cv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    uint32_t* w = reinterpret_cast<uint32_t*>(xa);
    #pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        const __nv_bfloat162 ab = *reinterpret_cast<const __nv_bfloat162*>(&w[p]);
        const float a = __low2float(ab), b = __high2float(ab);
        const float e = a * cv[p] + b * sv[p];
        const float o = b * cv[p] - a * sv[p];
        w[p] = pack_bf16(e, o);
    }
}

template <int DEPTH>
__global__ void wo_a_small_kernel(const uint4* __restrict__ W, const float* __restrict__ S,
                                  const __nv_bfloat16* __restrict__ x, int M, const int64_t* __restrict__ pos,
                                  const float* __restrict__ cs, int cs_stride,
                                  __nv_bfloat16* __restrict__ out, int G, int R, int D, int HEAD, int NOPE, int WPT)
{
    extern __shared__ float red[];   // [warps][4][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int kp = warp % WPT, ts = warp / WPT;
    const int nch = D >> 6;
    const int r = blockIdx.x * (blockDim.x / 32 / WPT) + ts;   // 16-row tile
    const int ntile = G * R / 16;
    const int c_lo = kp * nch / WPT, c_hi = (kp + 1) * nch / WPT;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = r * 16, grp = n0 / R, rl = n0 % R;
    const int m = min(g, M - 1);   // rows >= M read row M - 1: computed, never stored
    const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t) m * G * D + (size_t) grp * D) + 2 * t;
    const float* cs_row = cs + (size_t) pos[m] * cs_stride;
    const int HALF = (HEAD - NOPE) / 2;
    const uint4* wp0 = W + ((size_t) (n0 + g) * D) / 16 + t;
    const uint4* wp1 = wp0 + (size_t) 8 * D / 16;
    const float* sp = S + ((size_t) grp * (R / 32) + rl / 32) * (D / 32) + (t >> 1);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto load = [&] (int c, uint4 (&w)[2], uint4 (&xa)[2], uint32_t& sw)
    {
        w[0] = ld_w(wp0 + c * 4);
        w[1] = ld_w(wp1 + c * 4);
        xa[0] = ld_x(xp + 8 * c);
        xa[1] = ld_x(xp + 8 * c + 1);
        sw = __float_as_uint(__ldg(sp + 2 * c));
    };
    auto compute = [&] (int c, const uint4 (&wc)[2], uint4 (&xc)[2], uint32_t sw)
    {
        const int d = (64 * c + 16 * t) % HEAD;
        if (d >= NOPE) rope16(xc, cs_row, (d - NOPE) / 2, HALF);
        const uint32_t mult = (((sw >> 23) & 0xFFu) + 120u) * 0x00800080u;   // bf16x2 of 2^120 * s
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            const uint32_t X0 = comp(xc[s >> 1], (s & 1) * 2), X1 = comp(xc[s >> 1], (s & 1) * 2 + 1);
            const uint32_t b0 = __byte_perm(X0, X1, 0x5410), b1 = __byte_perm(X0, X1, 0x7632);
            uint32_t a0, a1, a2, a3;
            dq4(comp(wc[0], s), mult, a0, a2);
            dq4(comp(wc[1], s), mult, a1, a3);
            mma_bf16(acc, a0, a1, a2, a3, b0, b1);
        }
    };

    if (r < ntile && c_lo < c_hi)
    {
        uint4 wr[DEPTH][2], xr[DEPTH][2];
        uint32_t sr[DEPTH];
        #pragma unroll
        for (int dd = 0; dd < DEPTH; ++dd)
            if (c_lo + dd < c_hi) load(c_lo + dd, wr[dd], xr[dd], sr[dd]);
        int c = c_lo;
        for (; c + DEPTH <= c_hi; c += DEPTH)
        {
            #pragma unroll
            for (int dd = 0; dd < DEPTH; ++dd)
            {
                compute(c + dd, wr[dd], xr[dd], sr[dd]);
                if (c + dd + DEPTH < c_hi) load(c + dd + DEPTH, wr[dd], xr[dd], sr[dd]);
            }
        }
        #pragma unroll
        for (int dd = 0; dd < DEPTH; ++dd)
            if (c + dd < c_hi) compute(c + dd, wr[dd], xr[dd], sr[dd]);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) red[(warp * 4 + j) * 32 + lane] = acc[j];
    __syncthreads();
    if (kp != 0 || r >= ntile) return;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int w = 0; w < WPT; ++w)
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] += red[((warp + w) * 4 + j) * 32 + lane];
    const int n = n0 + g, ldo = G * R;
    const int m0 = 2 * t, m1 = 2 * t + 1;
    if (m0 < M) { out[(size_t) m0 * ldo + n] = __float2bfloat16_rn(v[0]); out[(size_t) m0 * ldo + n + 8] = __float2bfloat16_rn(v[2]); }
    if (m1 < M) { out[(size_t) m1 * ldo + n] = __float2bfloat16_rn(v[1]); out[(size_t) m1 * ldo + n + 8] = __float2bfloat16_rn(v[3]); }
}

}  // namespace

void wo_a_small(torch::Tensor W, torch::Tensor S, torch::Tensor x, torch::Tensor pos, torch::Tensor cs, torch::Tensor out,
                int64_t G, int64_t R, int64_t D, int64_t HEAD, int64_t NOPE, int64_t WPT, int64_t TPB, int64_t DEPTH)
{
    const int M = x.size(0);
    TORCH_CHECK(M >= 1 && M <= 8 && R % 32 == 0 && D % 64 == 0 && HEAD % 64 == 0 && NOPE % 16 == 0 && NOPE < HEAD
                && D % HEAD == 0, "dsv41 wo_a_small: shape");
    TORCH_CHECK(W.is_contiguous() && W.numel() == G * R * D && W.element_size() == 1, "dsv41 wo_a_small: weight");
    TORCH_CHECK(S.is_contiguous() && S.dtype() == torch::kFloat32 && S.numel() == G * (R / 32) * (D / 32), "dsv41 wo_a_small: scales");
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.is_contiguous() && x.numel() == M * G * D
                && (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0, "dsv41 wo_a_small: x");
    TORCH_CHECK(pos.dtype() == torch::kInt64 && pos.numel() >= M && cs.dtype() == torch::kFloat32 && cs.stride(1) == 1
                && cs.size(1) == HEAD - NOPE && cs.stride(0) % 4 == 0, "dsv41 wo_a_small: rope");
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == M * G * R, "dsv41 wo_a_small: out");
    TORCH_CHECK(WPT >= 1 && TPB >= 1 && WPT * TPB <= 32 && DEPTH >= 1 && DEPTH <= 3, "dsv41 wo_a_small: config");
    const at::cuda::CUDAGuard guard(x.device());
    const int ntile = G * R / 16, grid = (ntile + TPB - 1) / TPB, threads = 32 * WPT * TPB, smem = threads / 32 * 4 * 32 * 4;
    auto st = at::cuda::getCurrentCUDAStream();
#define L(D_) wo_a_small_kernel<D_><<<grid, threads, smem, st>>>((const uint4*) W.data_ptr(), S.data_ptr<float>(), \
        (const __nv_bfloat16*) x.data_ptr(), M, pos.data_ptr<int64_t>(), cs.data_ptr<float>(), cs.stride(0), \
        (__nv_bfloat16*) out.data_ptr(), G, R, D, HEAD, NOPE, WPT)
    if (DEPTH == 1) L(1); else if (DEPTH == 2) L(2); else L(3);
#undef L
}
